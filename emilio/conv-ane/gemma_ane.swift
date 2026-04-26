// Gemma ANE runtime — pure-ANE heavy compute path.
//
// Loads CoreML INT8 shards targeting the Apple Neural Engine. Tokenizer,
// embedding lookup, routing bookkeeping, and sampling run on the host; attention,
// FFN, final norm, and LM head projection must come from ANE CoreML shards.
// Missing LM head shards are fatal so the runtime cannot silently fall back to
// host-side Accelerate projection.
//
// Build:
//   swiftc -O -framework CoreML -framework Accelerate -o gemma_ane gemma_ane.swift
//
// Usage:
//   ./gemma_ane --meta python/moe/out/gemma_swift_head_meta.json \
//       --prompt-ids 2,818,5279,529,7001,563 --n-new 8
//
// Experimental per-expert dispatch is opt-in with --use-per-expert. If a meta
// file contains both `layers` and `per_expert_layers`, the validated 90-shard
// split path is used by default.
//
// Expert-GROUP dispatch: --expert-groups flag. Uses grouped shards (16 experts
// per shard, 8 groups/layer) loaded at startup. Eliminates per-token MLModel
// loading overhead. Group shard input: x (1,D,1,1) + expert_weights (1,16,1,1).

import Accelerate
import CoreML
import Foundation

struct GemmaLayerSpec: Decodable {
    let layer: Int
    let attn: String
    let ffnPartials: [String]
    let ffnLast: String

    enum CodingKeys: String, CodingKey {
        case layer
        case attn
        case ffnPartials = "ffn_partials"
        case ffnLast = "ffn_last"
    }
}

struct LMHeadShardSpec: Decodable {
    let shardIdx: Int
    let vocabStart: Int
    let vocabEnd: Int
    let mlmodelc: String

    enum CodingKeys: String, CodingKey {
        case shardIdx = "shard_idx"
        case vocabStart = "vocab_start"
        case vocabEnd = "vocab_end"
        case mlmodelc
    }
}

// Per-expert MoE dispatch structs
struct PerExpertLayerSpec: Decodable {
    let layer: Int
    let attn: String
    let nExperts: Int
    let topK: Int
    let experts: [String]       // paths to 64 .mlmodelc
    let combine: String         // path to combine .mlmodelc
    let routerProjBin: String   // path to router proj fp16 binary
    let routerPerExpScaleBin: String  // path to per-expert scale binary

    enum CodingKeys: String, CodingKey {
        case layer
        case attn
        case nExperts = "n_experts"
        case topK = "top_k"
        case experts
        case combine
        case routerProjBin = "router_proj_bin"
        case routerPerExpScaleBin = "router_per_expert_scale_bin"
    }
}

// Expert-GROUP MoE dispatch structs (G=8 groups of 16 experts each)
struct ExpertGroupSpec: Decodable {
    let group: Int
    let expertStart: Int
    let expertEnd: Int
    let modelc: String

    enum CodingKeys: String, CodingKey {
        case group
        case expertStart = "expert_start"
        case expertEnd = "expert_end"
        case modelc
    }
}

struct ExpertGroupCombineSpec: Decodable {
    let mlmodelc: String
}

struct ExpertGroupRouterSpec: Decodable {
    let projFp32: String
    let perExpertScaleFp32: String
    let nExperts: Int
    let dModel: Int

    enum CodingKeys: String, CodingKey {
        case projFp32 = "proj_fp32"
        case perExpertScaleFp32 = "per_expert_scale_fp32"
        case nExperts = "n_experts"
        case dModel = "d_model"
    }
}

struct ExpertGroupLayerSpec: Decodable {
    let layer: Int
    let groups: [ExpertGroupSpec]
    let combine: ExpertGroupCombineSpec
    let router: ExpertGroupRouterSpec
    let gateScale: Float
    let attn: String

    enum CodingKeys: String, CodingKey {
        case layer, groups, combine, router
        case gateScale = "gate_scale"
        case attn
    }
}

struct ExpertGroupsMeta: Decodable {
    let mode: String
    let nGroups: Int
    let groupSize: Int
    let nExperts: Int
    let topK: Int
    let layers: [ExpertGroupLayerSpec]
    // Shared fields from GemmaRuntimeMeta
    let artifactsVersion: Int
    let embedBin: String
    let finalNormGammaBin: String
    let vocabSize: Int
    let dModel: Int
    let rmsNormEps: Double
    let softcap: Double
    let tieWordEmbeddings: Bool
    let bosTokenId: Int
    let eosTokenId: Int
    let maxCtx: Int
    let slidingRopeTheta: Double
    let globalRopeTheta: Double
    let slidingDHead: Int
    let globalRotDim: Int
    let tokenizerJson: String
    let lmHeadShards: [LMHeadShardSpec]?

    enum CodingKeys: String, CodingKey {
        case mode
        case nGroups = "n_groups"
        case groupSize = "group_size"
        case nExperts = "n_experts"
        case topK = "top_k"
        case layers
        case artifactsVersion = "artifacts_version"
        case embedBin = "embed_bin"
        case finalNormGammaBin = "final_norm_gamma_bin"
        case vocabSize = "vocab_size"
        case dModel = "d_model"
        case rmsNormEps = "rms_norm_eps"
        case softcap
        case tieWordEmbeddings = "tie_word_embeddings"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case maxCtx = "max_ctx"
        case slidingRopeTheta = "sliding_rope_theta"
        case globalRopeTheta = "global_rope_theta"
        case slidingDHead = "sliding_d_head"
        case globalRotDim = "global_rot_dim"
        case tokenizerJson = "tokenizer_json"
        case lmHeadShards = "lm_head_shards"
    }
}

/// Router weights for one layer — CPU-side top-K computation.
/// Supports both Float32 and Float16 bin files (auto-detected by file size).
struct LayerRouter {
    let nExperts: Int
    let topK: Int
    let projWeights: UnsafeMutablePointer<Float>  // (nExperts, dModel) row-major, Float32
    let perExpertScale: UnsafeMutablePointer<Float>  // (nExperts,), Float32
    private let projCount: Int
    private let scaleCount: Int

    init(projBinPath: String, scaleBinPath: String, nExperts: Int, topK: Int, dModel: Int) throws {
        self.nExperts = nExperts
        self.topK = topK
        self.projCount = nExperts * dModel
        self.scaleCount = nExperts
        let projData = try Data(contentsOf: URL(fileURLWithPath: projBinPath))
        let scaleData = try Data(contentsOf: URL(fileURLWithPath: scaleBinPath))
        // Auto-detect fp16 (2 bytes/elem) vs fp32 (4 bytes/elem)
        let pCount = projCount
        let sCount = scaleCount
        let proj = UnsafeMutablePointer<Float>.allocate(capacity: pCount)
        if projData.count == pCount * 4 {
            // Float32 binary
            projData.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float.self)
                proj.initialize(from: src.baseAddress!, count: pCount)
            }
        } else {
            precondition(projData.count == pCount * 2,
                         "router proj size mismatch: \(projData.count) vs \(pCount * 2) or \(pCount * 4)")
            // Float16 binary — promote to Float32
            projData.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float16.self)
                for i in 0..<pCount { proj[i] = Float(src[i]) }
            }
        }
        self.projWeights = proj
        let scale = UnsafeMutablePointer<Float>.allocate(capacity: sCount)
        if scaleData.count == sCount * 4 {
            scaleData.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float.self)
                scale.initialize(from: src.baseAddress!, count: sCount)
            }
        } else {
            precondition(scaleData.count == sCount * 2,
                         "router scale size mismatch: \(scaleData.count) vs \(sCount * 2) or \(sCount * 4)")
            scaleData.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float16.self)
                for i in 0..<sCount { scale[i] = Float(src[i]) }
            }
        }
        self.perExpertScale = scale
    }

    /// CPU router: input_ln(x) → matmul(proj) → softmax → top-K.
    /// Returns (topIndices, topWeights) of size topK.
    /// The input x is already the post-attention hidden state.
    /// Router proj already has scale fused in (done at build time).
    func route(_ x: UnsafePointer<Float16>, dModel: Int, rmsEps: Float) -> ([Int], [Float]) {
        // 1. RMSNorm on x (without learned scale — that's fused into proj)
        var sumSq: Float = 0
        let K = Float(dModel).squareRoot()
        for i in 0..<dModel {
            let v = Float(x[i]) / K
            sumSq += v * v
        }
        let rms = (sumSq / Float(dModel) + rmsEps / (K * K)).squareRoot()

        // 2. Matmul: scores = proj @ (x / rms), shape (nExperts,)
        var scores = [Float](repeating: 0, count: nExperts)
        let invRmsK = 1.0 / (rms * K)
        for e in 0..<nExperts {
            var dot: Float = 0
            let rowBase = e * dModel
            for d in 0..<dModel {
                dot += projWeights[rowBase + d] * Float(x[d])
            }
            scores[e] = dot * invRmsK
        }

        // 3. Softmax
        var maxScore: Float = scores[0]
        for e in 1..<nExperts { if scores[e] > maxScore { maxScore = scores[e] } }
        var expSum: Float = 0
        for e in 0..<nExperts {
            scores[e] = expf(scores[e] - maxScore)
            expSum += scores[e]
        }
        let invSum = 1.0 / expSum
        for e in 0..<nExperts { scores[e] *= invSum }

        // 4. Top-K selection
        var topIdx = [Int](repeating: 0, count: topK)
        var topW = [Float](repeating: 0, count: topK)
        for k in 0..<topK {
            var bestE = -1
            var bestS: Float = -Float.infinity
            for e in 0..<nExperts {
                if scores[e] > bestS {
                    bestS = scores[e]
                    bestE = e
                }
            }
            topIdx[k] = bestE
            topW[k] = bestS
            scores[bestE] = -Float.infinity  // exclude from future picks
        }

        // 5. Renormalize top-K weights
        var wSum: Float = 0
        for k in 0..<topK { wSum += topW[k] }
        let invWSum = 1.0 / wSum
        for k in 0..<topK {
            topW[k] = topW[k] * invWSum * perExpertScale[topIdx[k]]
        }

        return (topIdx, topW)
    }
}

struct GemmaRuntimeMeta: Decodable {
    let artifactsVersion: Int
    let embedBin: String
    let finalNormGammaBin: String
    let vocabSize: Int
    let dModel: Int
    let rmsNormEps: Double
    let softcap: Double
    let tieWordEmbeddings: Bool
    let bosTokenId: Int
    let eosTokenId: Int
    let maxCtx: Int
    let slidingRopeTheta: Double
    let globalRopeTheta: Double
    let slidingDHead: Int
    let globalRotDim: Int
    let tokenizerJson: String
    let layers: [GemmaLayerSpec]
    let lmHeadShards: [LMHeadShardSpec]?
    let perExpertLayers: [PerExpertLayerSpec]?

    enum CodingKeys: String, CodingKey {
        case artifactsVersion = "artifacts_version"
        case embedBin = "embed_bin"
        case finalNormGammaBin = "final_norm_gamma_bin"
        case vocabSize = "vocab_size"
        case dModel = "d_model"
        case rmsNormEps = "rms_norm_eps"
        case softcap
        case tieWordEmbeddings = "tie_word_embeddings"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case maxCtx = "max_ctx"
        case slidingRopeTheta = "sliding_rope_theta"
        case globalRopeTheta = "global_rope_theta"
        case slidingDHead = "sliding_d_head"
        case globalRotDim = "global_rot_dim"
        case tokenizerJson = "tokenizer_json"
        case layers
        case lmHeadShards = "lm_head_shards"
        case perExpertLayers = "per_expert_layers"
    }
}

struct GemmaTokenizerAddedToken: Decodable {
    let id: Int
    let content: String
    let special: Bool?
}

struct GemmaTokenizerMerge: Decodable {
    let left: String
    let right: String

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let raw = try? container.decode(String.self) {
            let parts = raw.split(separator: " ", maxSplits: 1).map(String.init)
            guard parts.count == 2 else {
                throw DecodingError.dataCorruptedError(in: container,
                                                       debugDescription: "invalid merge string: \(raw)")
            }
            self.left = parts[0]
            self.right = parts[1]
            return
        }
        let pair = try container.decode([String].self)
        guard pair.count == 2 else {
            throw DecodingError.dataCorruptedError(in: container,
                                                   debugDescription: "invalid merge pair: \(pair)")
        }
        self.left = pair[0]
        self.right = pair[1]
    }

    var pairKey: String {
        "\(left) \(right)"
    }
}

struct GemmaTokenizerModel: Decodable {
    let type: String
    let unkToken: String
    let fuseUnk: Bool
    let byteFallback: Bool
    let ignoreMerges: Bool
    let vocab: [String: Int]
    let merges: [GemmaTokenizerMerge]

    enum CodingKeys: String, CodingKey {
        case type
        case unkToken = "unk_token"
        case fuseUnk = "fuse_unk"
        case byteFallback = "byte_fallback"
        case ignoreMerges = "ignore_merges"
        case vocab
        case merges
    }
}

struct GemmaTokenizerRoot: Decodable {
    let addedTokens: [GemmaTokenizerAddedToken]
    let model: GemmaTokenizerModel

    enum CodingKeys: String, CodingKey {
        case addedTokens = "added_tokens"
        case model
    }
}

let defaultPromptIds = [2, 818, 5279, 529, 7001, 563]

func loadMeta(_ path: String) throws -> GemmaRuntimeMeta {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONDecoder().decode(GemmaRuntimeMeta.self, from: data)
}

func parsePromptIds(_ raw: String) -> [Int] {
    raw.split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
        .map { Int($0)! }
}

func parseIntSet(_ raw: String) -> Set<Int> {
    Set(parsePromptIds(raw))
}

func hiddenStageKey(_ raw: String) -> (boundary: Int, suffix: String)? {
    guard raw.hasPrefix("hidden_l") else { return nil }
    let body = raw.dropFirst("hidden_l".count)
    let digits = String(body.prefix { $0.isNumber })
    guard !digits.isEmpty, let boundary = Int(digits) else { return nil }
    let suffixStart = body.index(body.startIndex, offsetBy: digits.count)
    let suffix = String(body[suffixStart...])
    return (boundary, suffix)
}

func hiddenStageBoundary(_ raw: String) -> Int? {
    hiddenStageKey(raw)?.boundary
}

func idsMatch(_ actual: [Int], expected: [Int]) -> Bool {
    actual.count == expected.count && zip(actual, expected).allSatisfy(==)
}

func ensureParentDirectory(for path: String) throws {
    let url = URL(fileURLWithPath: path)
    let parent = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent,
                                            withIntermediateDirectories: true)
}

func writeAtomicData(_ data: Data, to path: String) throws {
    try ensureParentDirectory(for: path)
    try data.write(to: URL(fileURLWithPath: path), options: .atomic)
}

func writeAtomicJSON(_ object: [String: Any], to path: String) throws {
    let data = try JSONSerialization.data(withJSONObject: object,
                                          options: [.prettyPrinted, .sortedKeys])
    try writeAtomicData(data, to: path)
}

func writeLogitsDump(prefix: String,
                     kind: String,
                     rows: Int,
                     promptIds: [Int],
                     promptText: String?,
                     vocabSize: Int,
                     logitsFlat: [Float],
                     extraMeta: [String: Any] = [:],
                     announce: Bool = true) throws {
    let expectedCount = rows * vocabSize
    precondition(logitsFlat.count == expectedCount,
                 "\(kind) logits dump mismatch: \(logitsFlat.count) vs \(expectedCount)")
    let binPath = prefix + "_logits_f32.bin"
    let metaPath = prefix + "_meta.json"
    let logitsData = logitsFlat.withUnsafeBufferPointer { Data(buffer: $0) }
    try writeAtomicData(logitsData, to: binPath)
    var meta: [String: Any] = [
        "kind": kind,
        "rows": rows,
        "cols": vocabSize,
        "dtype": "float32",
        "prompt_ids": promptIds,
        "logits_bin": binPath,
    ]
    if let promptText {
        meta["prompt_text"] = promptText
    }
    for (key, value) in extraMeta {
        meta[key] = value
    }
    try writeAtomicJSON(meta, to: metaPath)
    if announce {
        print("  wrote \(kind) logits: \(binPath)")
        print("  wrote \(kind) meta  : \(metaPath)")
    }
}

func writePromptLogitsDump(prefix: String,
                           promptIds: [Int],
                           promptText: String?,
                           vocabSize: Int,
                           logitsFlat: [Float]) throws {
    try writeLogitsDump(prefix: prefix,
                        kind: "prompt",
                        rows: promptIds.count,
                        promptIds: promptIds,
                        promptText: promptText,
                        vocabSize: vocabSize,
                        logitsFlat: logitsFlat)
}

func writeDecodeLogitsDump(prefix: String,
                           promptIds: [Int],
                           promptText: String?,
                           generatedIds: [Int],
                           nNewRequested: Int,
                           vocabSize: Int,
                           logitsFlat: [Float],
                           announce: Bool = true) throws {
    try writeLogitsDump(prefix: prefix,
                        kind: "decode",
                        rows: generatedIds.count,
                        promptIds: promptIds,
                        promptText: promptText,
                        vocabSize: vocabSize,
                        logitsFlat: logitsFlat,
                        extraMeta: [
                            "generated_ids": generatedIds,
                            "logit_row_semantics": "post_token",
                            "n_new_requested": nNewRequested,
                        ],
                        announce: announce)
}

struct HiddenBoundarySnapshot {
    let shardHidden: [[Float]]
    let projectedHidden: [Float]
}

func flatOffset(for linearIndex: Int,
                shape: [Int],
                strides: [Int]) -> Int {
    precondition(shape.count == strides.count,
                 "shape/stride rank mismatch")
    if shape.isEmpty {
        precondition(linearIndex == 0, "scalar MLMultiArray has one element")
        return 0
    }

    var remaining = linearIndex
    var offset = 0
    for axis in stride(from: shape.count - 1, through: 0, by: -1) {
        let dim = shape[axis]
        precondition(dim > 0, "invalid MLMultiArray dim \(dim) at axis \(axis)")
        let coord = remaining % dim
        remaining /= dim
        offset += coord * strides[axis]
    }
    precondition(remaining == 0,
                 "linear index \(linearIndex) out of bounds for shape \(shape)")
    return offset
}

func float32Array(from source: MLMultiArray, count: Int) -> [Float] {
    precondition(source.count >= count, "source hidden smaller than d_model")
    let sourcePtr = source.dataPointer.assumingMemoryBound(to: Float16.self)
    let shape = source.shape.map { Int(truncating: $0) }
    let strides = source.strides.map { Int(truncating: $0) }
    var result = [Float](repeating: 0, count: count)
    for index in 0..<count {
        result[index] = Float(sourcePtr[flatOffset(for: index,
                                                   shape: shape,
                                                   strides: strides)])
    }
    return result
}

func writeHiddenBoundaryDump(prefix: String,
                            promptIds: [Int],
                            promptText: String?,
                            dim: Int,
                            stageNames: [String],
                            decodeSteps: [Int],
                            absolutePositions: [Int],
                            emittedTokenIds: [Int],
                            generatedIdsPrefixes: [[Int]],
                            hiddenFlat: [Float],
                            announce: Bool = true) throws {
    let expectedCount = decodeSteps.count * stageNames.count * dim
    precondition(hiddenFlat.count == expectedCount,
                 "hidden boundary dump mismatch: \(hiddenFlat.count) vs \(expectedCount)")
    let binPath = prefix + "_hidden_boundaries_f32.bin"
    let metaPath = prefix + "_hidden_boundaries_meta.json"
    let hiddenData = hiddenFlat.withUnsafeBufferPointer { Data(buffer: $0) }
    try writeAtomicData(hiddenData, to: binPath)
    var meta: [String: Any] = [
        "kind": "decode_hidden_boundaries",
        "dtype": "float32",
        "dim": dim,
        "stage_names": stageNames,
        "n_steps": decodeSteps.count,
        "decode_steps": decodeSteps,
        "absolute_positions": absolutePositions,
        "emitted_token_ids": emittedTokenIds,
        "generated_ids_prefixes": generatedIdsPrefixes,
        "prompt_ids": promptIds,
        "hidden_bin": binPath,
    ]
    if let promptText {
        meta["prompt_text"] = promptText
    }
    try writeAtomicJSON(meta, to: metaPath)
    if announce {
        print("  wrote hidden boundaries: \(binPath)")
        print("  wrote hidden meta      : \(metaPath)")
    }
}

func resolvePath(_ rawPath: String, relativeTo metaPath: String) -> String {
    if rawPath.hasPrefix("/") { return rawPath }
    let fileManager = FileManager.default
    let cwd = fileManager.currentDirectoryPath
    let cwdCandidate = URL(fileURLWithPath: cwd).appendingPathComponent(rawPath).path
    if fileManager.fileExists(atPath: cwdCandidate) {
        return cwdCandidate
    }
    let metaDir = URL(fileURLWithPath: metaPath).deletingLastPathComponent()
    var cursor = metaDir
    while true {
        let candidate = cursor.appendingPathComponent(rawPath).path
        if fileManager.fileExists(atPath: candidate) {
            return candidate
        }
        let parent = cursor.deletingLastPathComponent()
        if parent.path == cursor.path {
            break
        }
        cursor = parent
    }
    return metaDir.appendingPathComponent(rawPath).path
}

final class GemmaBPETokenizer {
    let idToToken: [String]
    let tokenToId: [String: Int]
    let mergeRank: [String: Int]
    let mergeCount: Int
    let bosId: Int
    let eosId: Int
    let unkId: Int
    let byteFallback: Bool
    let specialTokens: [(String, Int)]
    let specialTokenIds: Set<Int>

    init(jsonPath: String, bosId: Int, eosId: Int) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: jsonPath))
        let root = try JSONDecoder().decode(GemmaTokenizerRoot.self, from: data)
        precondition(root.model.type == "BPE", "unsupported Gemma tokenizer type: \(root.model.type)")

        self.tokenToId = root.model.vocab
        let maxTokenId = root.model.vocab.values.max() ?? -1
        var idToToken = [String](repeating: "", count: maxTokenId + 1)
        for (token, tokenId) in root.model.vocab {
            idToToken[tokenId] = token
        }
        self.idToToken = idToToken

        var mergeRank = [String: Int](minimumCapacity: root.model.merges.count)
        for (rank, merge) in root.model.merges.enumerated() {
            mergeRank[merge.pairKey] = rank
        }
        self.mergeRank = mergeRank
        self.mergeCount = root.model.merges.count

        self.bosId = bosId
        self.eosId = eosId
        self.unkId = root.model.vocab[root.model.unkToken] ?? 3
        self.byteFallback = root.model.byteFallback

        let specials = root.addedTokens
            .filter { $0.special ?? false }
            .map { ($0.content, $0.id) }
            .sorted { lhs, rhs in lhs.0.count > rhs.0.count }
        self.specialTokens = specials
        self.specialTokenIds = Set(specials.map { $0.1 })
    }

    func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        var ids = [Int]()
        if addSpecialTokens {
            ids.append(bosId)
        }
        ids.append(contentsOf: encodeBody(text))
        return ids
    }

    func decode(_ ids: [Int], skipSpecialTokens: Bool = true) -> String {
        var output = ""
        var pendingBytes = [UInt8]()

        func flushPendingBytes() {
            guard !pendingBytes.isEmpty else { return }
            output += String(decoding: pendingBytes, as: UTF8.self)
            pendingBytes.removeAll(keepingCapacity: true)
        }

        for id in ids {
            guard id >= 0 && id < idToToken.count else { continue }
            if skipSpecialTokens && specialTokenIds.contains(id) {
                flushPendingBytes()
                continue
            }
            let token = idToToken[id]
            if let byte = Self.parseByteFallback(token) {
                pendingBytes.append(byte)
            } else {
                flushPendingBytes()
                output += token
            }
        }
        flushPendingBytes()
        return output.replacingOccurrences(of: "▁", with: " ")
    }

    private func encodeBody(_ text: String) -> [Int] {
        let normalized = text.replacingOccurrences(of: " ", with: "▁")
        if normalized.isEmpty { return [] }

        var ids = [Int]()
        var cursor = normalized.startIndex
        var runStart = cursor
        while cursor < normalized.endIndex {
            var matched: (String, Int)? = nil
            for candidate in specialTokens {
                if normalized[cursor...].hasPrefix(candidate.0) {
                    matched = candidate
                    break
                }
            }
            if let special = matched {
                if runStart < cursor {
                    ids.append(contentsOf: bpeEncode(String(normalized[runStart..<cursor])))
                }
                ids.append(special.1)
                cursor = normalized.index(cursor, offsetBy: special.0.count)
                runStart = cursor
            } else {
                cursor = normalized.index(after: cursor)
            }
        }
        if runStart < normalized.endIndex {
            ids.append(contentsOf: bpeEncode(String(normalized[runStart..<normalized.endIndex])))
        }
        return ids
    }

    private func bpeEncode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        var pieces = initialPieces(text)
        while pieces.count >= 2 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0..<(pieces.count - 1) {
                let pair = "\(pieces[index]) \(pieces[index + 1])"
                if let rank = mergeRank[pair], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            let merged = pieces[bestIndex] + pieces[bestIndex + 1]
            pieces.remove(at: bestIndex + 1)
            pieces[bestIndex] = merged
        }

        var ids = [Int]()
        for piece in pieces {
            appendTokenIds(for: piece, into: &ids)
        }
        return ids
    }

    private func initialPieces(_ text: String) -> [String] {
        var pieces = [String]()
        for scalar in text.unicodeScalars {
            let piece = String(scalar)
            if tokenToId[piece] != nil {
                pieces.append(piece)
            } else if byteFallback {
                for byte in piece.utf8 {
                    pieces.append(String(format: "<0x%02X>", Int(byte)))
                }
            } else {
                pieces.append(idToToken[unkId])
            }
        }
        return pieces
    }

    private func appendTokenIds(for piece: String, into ids: inout [Int]) {
        if let tokenId = tokenToId[piece] {
            ids.append(tokenId)
            return
        }
        if byteFallback {
            for byte in piece.utf8 {
                let byteToken = String(format: "<0x%02X>", Int(byte))
                if let byteId = tokenToId[byteToken] {
                    ids.append(byteId)
                } else {
                    ids.append(unkId)
                }
            }
        } else {
            ids.append(unkId)
        }
    }

    private static func parseByteFallback(_ token: String) -> UInt8? {
        guard token.hasPrefix("<0x"), token.hasSuffix(">") else { return nil }
        let hex = token.dropFirst(3).dropLast()
        return UInt8(hex, radix: 16)
    }
}

final class FP16VectorFile {
    let rawData: Data
    let data: UnsafeBufferPointer<UInt16>
    let count: Int

    init(path: String, count: Int) {
        let url = URL(fileURLWithPath: path)
        self.rawData = try! Data(contentsOf: url)
        self.count = count
        self.data = rawData.withUnsafeBytes { ptr in
            UnsafeBufferPointer(
                start: ptr.baseAddress!.assumingMemoryBound(to: UInt16.self),
                count: ptr.count / 2)
        }
        precondition(self.data.count == count,
                     "FP16 vector mismatch: \(self.data.count) vs \(count)")
    }

    func float32(at index: Int) -> Float {
        Float(Float16(bitPattern: data[index]))
    }

    func scaledValues(dividingBy denominator: Float) -> [Float] {
        var result = [Float](repeating: 0, count: count)
        for index in 0..<count {
            result[index] = float32(at: index) / denominator
        }
        return result
    }
}

final class TokenEmbeddings {
    let rawData: Data
    let data: UnsafeBufferPointer<UInt16>
    let vocab: Int
    let dim: Int
    /// Pre-converted FP32 embedding matrix for Accelerate BLAS (row-major: vocab × dim).
    let embedF32: UnsafeMutablePointer<Float>

    init(path: String, vocab: Int, dim: Int) {
        let url = URL(fileURLWithPath: path)
        self.rawData = try! Data(contentsOf: url)
        self.vocab = vocab
        self.dim = dim
        self.data = rawData.withUnsafeBytes { ptr in
            UnsafeBufferPointer(
                start: ptr.baseAddress!.assumingMemoryBound(to: UInt16.self),
                count: ptr.count / 2)
        }
        precondition(data.count == vocab * dim,
                     "Embedding mismatch: \(data.count) vs \(vocab * dim)")
        // One-time FP16→FP32 conversion at load. ~2.8 GB for 262144×2816 but avoids
        // per-element cast on every token's LM head matmul.
        self.embedF32 = .allocate(capacity: vocab * dim)
        for i in 0..<(vocab * dim) {
            embedF32[i] = Float(Float16(bitPattern: data[i]))
        }
    }

    deinit {
        embedF32.deallocate()
    }

    func lookup(_ tokenId: Int) -> [Float] {
        precondition(tokenId >= 0 && tokenId < vocab, "token out of range: \(tokenId)")
        let start = tokenId * dim
        var result = [Float](repeating: 0, count: dim)
        for i in 0..<dim {
            result[i] = Float(Float16(bitPattern: data[start + i]))
        }
        return result
    }

    func writeScaledRow(_ tokenId: Int, scale: Float, into destination: UnsafeMutablePointer<Float16>) {
        precondition(tokenId >= 0 && tokenId < vocab, "token out of range: \(tokenId)")
        let start = tokenId * dim
        for index in 0..<dim {
            let value = Float(Float16(bitPattern: data[start + index])) * scale
            destination[index] = Float16(value)
        }
    }

    /// Vectorized LM head: logits = embedF32 @ projectedHidden, then tanh·softcap.
    /// Uses cblas_sgemv (BLAS matrix-vector multiply) instead of scalar double-loop.
    func fullLogits(projectedHidden: [Float], softcap: Float) -> [Float] {
        precondition(projectedHidden.count == dim, "projected hidden dim mismatch")
        var logits = [Float](repeating: 0, count: vocab)
        // embedF32 is row-major [vocab, dim]. logits = embed × projectedHidden.
        // vDSP_mmul: C[M×N] = A[M×K] × B[K×N], here M=vocab, K=dim, N=1.
        projectedHidden.withUnsafeBufferPointer { xBuf in
            logits.withUnsafeMutableBufferPointer { yBuf in
                vDSP_mmul(embedF32, 1,
                          xBuf.baseAddress!, 1,
                          yBuf.baseAddress!, 1,
                          vDSP_Length(vocab), vDSP_Length(1), vDSP_Length(dim))
            }
        }
        if softcap > 0 {
            // Vectorized tanh·softcap: vvtanhf + vDSP_vsmul
            var n = Int32(vocab)
            logits.withUnsafeMutableBufferPointer { buf in
                vvtanhf(buf.baseAddress!, buf.baseAddress!, &n)
                var sc = softcap
                vDSP_vsmul(buf.baseAddress!, 1, &sc, buf.baseAddress!, 1, vDSP_Length(vocab))
            }
        }
        return logits
    }

    /// Vectorized argmax via vDSP.
    func argmax(_ logits: [Float]) -> (tokenId: Int, logit: Float) {
        precondition(logits.count == vocab, "logits vocab mismatch")
        var bestLogit: Float = 0
        var bestIdx: vDSP_Length = 0
        logits.withUnsafeBufferPointer { buf in
            vDSP_maxvi(buf.baseAddress!, 1, &bestLogit, &bestIdx, vDSP_Length(vocab))
        }
        return (Int(bestIdx), bestLogit)
    }
}

func makeRoPE(theta: Float, dim: Int, pos: Int,
              cosPtr: UnsafeMutablePointer<Float16>,
              sinPtr: UnsafeMutablePointer<Float16>) {
    let half = dim / 2
    let posF = Float(pos)
    for index in 0..<half {
        let freq = posF / powf(theta, Float(index) / Float(half))
        let cosVal = Float16(cosf(freq))
        let sinVal = Float16(sinf(freq))
        cosPtr[index] = cosVal
        cosPtr[index + half] = cosVal
        sinPtr[index] = sinVal
        sinPtr[index + half] = sinVal
    }
}

func finalNormProjectedHidden(hidden: MLMultiArray,
                              gamma: FP16VectorFile,
                              eps: Float,
                              softcap: Float) -> [Float] {
    let hiddenPtr = hidden.dataPointer.assumingMemoryBound(to: Float16.self)
    let shape = hidden.shape.map { Int(truncating: $0) }
    let strides = hidden.strides.map { Int(truncating: $0) }
    let D = gamma.count
    var hidden32 = [Float](repeating: 0, count: D)
    // Copy FP16→FP32 (stride-aware — ANE output strides may be non-unit)
    for index in 0..<D {
        hidden32[index] = Float(hiddenPtr[flatOffset(for: index,
                                                     shape: shape,
                                                     strides: strides)])
    }
    // vDSP mean-of-squares → RMS
    var meanSq: Float = 0
    vDSP_measqv(hidden32, 1, &meanSq, vDSP_Length(D))
    let rms = sqrtf(meanSq + eps)
    let denom = rms * (softcap > 0 ? softcap : 1)
    // gamma / denom → scale vector (reuse gamma.data directly)
    var gammaOverRms = [Float](repeating: 0, count: D)
    for index in 0..<D {
        gammaOverRms[index] = gamma.float32(at: index) / denom
    }
    // projected = hidden32 .* gammaOverRms  (vDSP element-wise multiply)
    var projected = [Float](repeating: 0, count: D)
    vDSP_vmul(hidden32, 1, gammaOverRms, 1, &projected, 1, vDSP_Length(D))
    return projected
}

func copyFlatFloat16(_ source: MLMultiArray,
                     into target: UnsafeMutablePointer<Float16>,
                     count: Int) {
    precondition(source.count >= count, "source hidden smaller than d_model")
    let sourcePtr = source.dataPointer.assumingMemoryBound(to: Float16.self)
    let shape = source.shape.map { Int(truncating: $0) }
    let strides = source.strides.map { Int(truncating: $0) }
    // Fast path: if innermost stride is 1, use memcpy instead of per-element copy
    if !strides.isEmpty && strides[strides.count - 1] == 1 {
        // For rank-2 [1,D] or rank-3 [1,1,D] with unit inner stride, the first
        // `count` elements are contiguous starting at offset 0.
        memcpy(target, sourcePtr, count * MemoryLayout<Float16>.size)
    } else {
        for index in 0..<count {
            target[index] = sourcePtr[flatOffset(for: index,
                                                 shape: shape,
                                                 strides: strides)]
        }
    }
}

func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    var metaPath = "python/moe/out/gemma_swift_head_meta.json"
    var checkToken = 2
    var promptText: String? = nil
    var promptIds = defaultPromptIds
    var expectedPromptIds: [Int]? = nil
    var expectedGeneratedIds: [Int]? = nil
    var dumpPromptLogitsPrefix: String? = nil
    var dumpDecodeLogitsPrefix: String? = nil
    var traceDecode = false
    var traceLayers = false
    var nNew = 8
    var inspectOnly = false
    var tokenizeOnly = false
    var forcePerExpert = false
    var forceExpertGroups = false
    var expertGroupsMetaPath: String? = nil

    var i = 0
    while i < args.count {
        switch args[i] {
        case "--meta":
            metaPath = args[i + 1]
            i += 2
        case "--check-token":
            checkToken = Int(args[i + 1])!
            i += 2
        case "--prompt":
            promptText = args[i + 1]
            i += 2
        case "--prompt-ids":
            promptIds = parsePromptIds(args[i + 1])
            i += 2
        case "--expect-prompt-ids":
            expectedPromptIds = parsePromptIds(args[i + 1])
            i += 2
        case "--expect-generated-ids":
            expectedGeneratedIds = parsePromptIds(args[i + 1])
            i += 2
        case "--dump-prompt-logits-prefix":
            dumpPromptLogitsPrefix = args[i + 1]
            i += 2
        case "--dump-decode-logits-prefix":
            dumpDecodeLogitsPrefix = args[i + 1]
            i += 2
        case "--trace-decode":
            traceDecode = true
            i += 1
        case "--trace-layers":
            traceLayers = true
            i += 1
        case "--use-per-expert", "--per-expert":
            forcePerExpert = true
            i += 1
        case "--expert-groups":
            forceExpertGroups = true
            i += 1
        case "--expert-groups-meta":
            expertGroupsMetaPath = args[i + 1]
            forceExpertGroups = true
            i += 2
        case "--n-new":
            nNew = Int(args[i + 1])!
            i += 2
        case "--inspect-only":
            inspectOnly = true
            i += 1
        case "--tokenize-only":
            tokenizeOnly = true
            i += 1
        default:
            i += 1
        }
    }

    if let dumpPromptLogitsPrefix, let dumpDecodeLogitsPrefix,
       dumpPromptLogitsPrefix == dumpDecodeLogitsPrefix {
        throw NSError(domain: "gemma_ane", code: 4,
                      userInfo: [NSLocalizedDescriptionKey:
                        "dump prefixes must differ when writing both prompt and decode logits"])
    }

    let meta = try loadMeta(metaPath)
    let embedPath = resolvePath(meta.embedBin, relativeTo: metaPath)
    let gammaPath = resolvePath(meta.finalNormGammaBin, relativeTo: metaPath)
    let tokenizerPath = resolvePath(meta.tokenizerJson, relativeTo: metaPath)

    print("Loading Gemma runtime scaffold...")
    print("  meta: \(metaPath)")
    print("  vocab=\(meta.vocabSize) d_model=\(meta.dModel) softcap=\(meta.softcap)")
    print("  bos=\(meta.bosTokenId) eos=\(meta.eosTokenId) max_ctx=\(meta.maxCtx)")
    print("  tokenizer: \(tokenizerPath)")

    let embd = TokenEmbeddings(path: embedPath, vocab: meta.vocabSize, dim: meta.dModel)
    let gamma = FP16VectorFile(path: gammaPath, count: meta.dModel)
    let embedScale = sqrt(Float(meta.dModel))
    var tokenizer: GemmaBPETokenizer? = nil
    if promptText != nil || tokenizeOnly {
        tokenizer = try GemmaBPETokenizer(jsonPath: tokenizerPath,
                                          bosId: meta.bosTokenId,
                                          eosId: meta.eosTokenId)
        print("  tokenizer merges: \(tokenizer!.mergeCount)")
    }
    if let promptText {
        promptIds = tokenizer!.encode(promptText)
        print("  prompt text : \(String(reflecting: promptText))")
        print("  prompt ids (\(promptIds.count)): \(promptIds)")
        print("  prompt str  : \(String(reflecting: tokenizer!.decode(promptIds)))")
    }
    if let expectedPromptIds {
        let promptIdsOk = idsMatch(promptIds, expected: expectedPromptIds)
        print("  expect prompt ids: \(expectedPromptIds) -> \(promptIdsOk ? "PASS" : "FAIL")")
        if !promptIdsOk {
            throw NSError(domain: "gemma_ane", code: 2,
                          userInfo: [NSLocalizedDescriptionKey:
                            "prompt ids mismatch: actual=\(promptIds) expected=\(expectedPromptIds)"])
        }
    }

    print("  embed bin: \(embedPath)")
    print("  gamma bin: \(gammaPath)")
    print("  tied embeddings: \(meta.tieWordEmbeddings)")
    print("  layers: \(meta.layers.count)")
    for layer in meta.layers {
        let attnOk = FileManager.default.fileExists(atPath: resolvePath(layer.attn, relativeTo: metaPath))
        var partialStatus = ""
        for (pi, pp) in layer.ffnPartials.enumerated() {
            let ok = FileManager.default.fileExists(atPath: resolvePath(pp, relativeTo: metaPath))
            partialStatus += " p\(pi)=\(ok ? "ok" : "MISSING")"
        }
        let lastOk = FileManager.default.fileExists(atPath: resolvePath(layer.ffnLast, relativeTo: metaPath))
        print("    L\(layer.layer): attn=\(attnOk ? "ok" : "MISSING")\(partialStatus) last=\(lastOk ? "ok" : "MISSING")")
    }

    let tokVec = embd.lookup(checkToken)
    let preview = tokVec.prefix(8).map { String(format: "%.4f", Double($0)) }.joined(separator: ", ")
    print("  check token id: \(checkToken)")
    print("  embedding[0:8]: [\(preview)]")
    print("  gamma[0]=\(gamma.float32(at: 0)) gamma[last]=\(gamma.float32(at: meta.dModel - 1))")

    if tokenizeOnly {
        precondition(promptText != nil, "--tokenize-only requires --prompt")
        print("  tokenize-only complete")
        return
    }

    if inspectOnly {
        print("  inspect-only complete")
        return
    }

    if #unavailable(macOS 15.0) {
        throw NSError(domain: "gemma_ane", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "stateful Gemma runtime requires macOS 15.0+"])
    }

    let usePerExpert = forcePerExpert && !forceExpertGroups && (meta.perExpertLayers ?? []).count > 0
    let useExpertGroups = forceExpertGroups

    // Load expert-groups meta if requested
    var egMeta: ExpertGroupsMeta? = nil
    if useExpertGroups {
        let egMetaPath = expertGroupsMetaPath ?? "python/moe/out/expert_groups/expert_groups_meta.json"
        let egResolved = resolvePath(egMetaPath, relativeTo: metaPath)
        let egData = try Data(contentsOf: URL(fileURLWithPath: egResolved))
        egMeta = try JSONDecoder().decode(ExpertGroupsMeta.self, from: egData)
        print("  expert-groups meta: \(egMeta!.nGroups) groups × \(egMeta!.groupSize) experts, \(egMeta!.layers.count) layers")
    }

    let nLayers = useExpertGroups ? egMeta!.layers.count :
                  (usePerExpert ? meta.perExpertLayers!.count : meta.layers.count)
    precondition(nLayers > 0, "no layers configured")
    precondition(!promptIds.isEmpty, "prompt is empty")
    precondition(promptIds.count + nNew <= meta.maxCtx,
                 "prompt length \(promptIds.count) + n_new \(nNew) exceeds max_ctx \(meta.maxCtx)")

        let lmHeadShardSpecs = (useExpertGroups ? egMeta!.lmHeadShards : meta.lmHeadShards) ?? []
        guard !lmHeadShardSpecs.isEmpty else {
                throw NSError(domain: "gemma_ane", code: 3,
                                            userInfo: [NSLocalizedDescriptionKey:
                                                "pure ANE runtime requires lm_head_shards; refusing CPU LM head fallback"])
        }

    let mlConfig = MLModelConfiguration()
    mlConfig.computeUnits = .all  // ANE-preferred; GPU fallback for stateful attn ops
    var attnModels = [MLModel]()
    var attnStates = [MLState]()
    var ffnPartialModels = [[MLModel]]()  // [layer][partial_idx] — all except last (per-expert path only)
    var ffnLastModels = [MLModel]()          // [layer] — last partial + combiner (per-expert path only)
    var ffnPartialPaths = [[String]]()   // [layer][partial_idx] — lazy-loaded paths
    var ffnLastPaths = [String]()            // [layer] — lazy-loaded paths

    // Per-expert dispatch structures
    var expertPaths = [[String]]()     // [layer][expert_id] — paths for lazy loading
    var expertCache = [String: MLModel]()  // path → loaded model, LRU cache
    let expertCacheMax = 256           // max cached expert models (~1 full pass: 8×30=240)
    var combineModels = [MLModel]()    // [layer]
    var layerRouters = [LayerRouter]()  // [layer]

    // Expert-group dispatch structures (O3: sliding window — ANE limit ~120 execution plans)
    var egGroupPaths = [[String]]()       // [layer][group_idx] — resolved paths for all 30 layers
    var egCombinePaths = [String]()       // [layer] — resolved paths
    var egGroupWindow = [[MLModel?]]()    // [layer][group_idx] — nil if not in window
    var egCombineWindow = [MLModel?]()    // [layer] — nil if not in window
    var egWindowStart = 0                  // first layer with loaded group+combine
    var egWindowSize = 0                   // number of layers in window
    let egANESlotBudget = 118              // safe limit (120 minus headroom)
    var egRouters = [LayerRouter]()       // [layer]
    let egGroupSize = egMeta?.groupSize ?? 16  // experts per group shard
    let egNGroups = egMeta?.nGroups ?? 8       // groups per layer

    if usePerExpert {
        let peLayers = meta.perExpertLayers!
        print("  loading \(nLayers) layers (attn + combine + router; experts lazy-loaded)...")
        for (idx, layer) in peLayers.enumerated() {
            // Load attention
            let attnPath = resolvePath(layer.attn, relativeTo: metaPath)
            let attnModel = try MLModel(contentsOf: URL(fileURLWithPath: attnPath),
                                        configuration: mlConfig)
            attnModels.append(attnModel)
            attnStates.append(attnModel.makeState())

            // Store expert paths for lazy loading (NOT pre-loaded)
            var paths = [String]()
            for expertPath in layer.experts {
                paths.append(resolvePath(expertPath, relativeTo: metaPath))
            }
            expertPaths.append(paths)

            // Load combine model
            let combinePath = resolvePath(layer.combine, relativeTo: metaPath)
            combineModels.append(try MLModel(contentsOf: URL(fileURLWithPath: combinePath),
                                            configuration: mlConfig))

            // Load router weights
            let projPath = resolvePath(layer.routerProjBin, relativeTo: metaPath)
            let scalePath = resolvePath(layer.routerPerExpScaleBin, relativeTo: metaPath)
            layerRouters.append(try LayerRouter(
                projBinPath: projPath, scaleBinPath: scalePath,
                nExperts: layer.nExperts, topK: layer.topK, dModel: meta.dModel))

            if (idx + 1) % 5 == 0 || idx == nLayers - 1 {
                print("    loaded \(idx + 1)/\(nLayers) layers")
            }
        }
    } else if useExpertGroups {
        let egLayers = egMeta!.layers
        let egMetaDir = expertGroupsMetaPath ?? "python/moe/out/expert_groups/expert_groups_meta.json"
        let egMetaResolved = resolvePath(egMetaDir, relativeTo: metaPath)

        // Phase 1: Load all 30 attn shards + routers, resolve all paths
        print("  phase 1: loading \(nLayers) attn shards + routers...")
        for (idx, layer) in egLayers.enumerated() {
            let attnPath = resolvePath(layer.attn, relativeTo: metaPath)
            let attnModel = try MLModel(contentsOf: URL(fileURLWithPath: attnPath),
                                        configuration: mlConfig)
            attnModels.append(attnModel)
            attnStates.append(attnModel.makeState())

            // Resolve and store all group+combine paths for later loading
            var gPaths = [String]()
            for gs in layer.groups {
                gPaths.append(resolvePath(gs.modelc, relativeTo: egMetaResolved))
            }
            egGroupPaths.append(gPaths)
            egCombinePaths.append(resolvePath(layer.combine.mlmodelc, relativeTo: egMetaResolved))

            // Load router (CPU-side, no ANE slot)
            let projPath = resolvePath(layer.router.projFp32, relativeTo: egMetaResolved)
            let scalePath = resolvePath(layer.router.perExpertScaleFp32, relativeTo: egMetaResolved)
            egRouters.append(try LayerRouter(
                projBinPath: projPath, scaleBinPath: scalePath,
                nExperts: layer.router.nExperts, topK: egMeta!.topK, dModel: egMeta!.dModel))

            if (idx + 1) % 5 == 0 || idx == nLayers - 1 {
                print("    attn+router: \(idx + 1)/\(nLayers)")
            }
        }

        // Phase 2: Warmup — load all group+combine in batches to prime E5 ANE cache
        let attnSlots = nLayers  // 30 attn models permanently loaded
        let slotsAvail = egANESlotBudget - attnSlots  // ~88 slots for group+combine
        let shardsPerLayer = egNGroups + 1  // 8 groups + 1 combine = 9
        let warmBatch = slotsAvail / shardsPerLayer  // layers per warmup batch
        let nBatches = (nLayers + warmBatch - 1) / warmBatch
        print("  phase 2: ANE warmup — \(nBatches) batches of \(warmBatch) layers (priming E5 cache)...")
        let warmT0 = CFAbsoluteTimeGetCurrent()
        for batch in 0..<nBatches {
            let batchStart = batch * warmBatch
            let batchEnd = min(batchStart + warmBatch, nLayers)
            var warmModels = [MLModel]()
            for li in batchStart..<batchEnd {
                for gPath in egGroupPaths[li] {
                    warmModels.append(try MLModel(contentsOf: URL(fileURLWithPath: gPath),
                                                  configuration: mlConfig))
                }
                warmModels.append(try MLModel(contentsOf: URL(fileURLWithPath: egCombinePaths[li]),
                                              configuration: mlConfig))
            }
            let batchLayers = batchEnd - batchStart
            print("    batch \(batch+1)/\(nBatches): warmed L\(batchStart)..\(batchEnd-1) (\(batchLayers * shardsPerLayer) shards)")
            warmModels.removeAll()  // free ANE slots
        }
        let warmT1 = CFAbsoluteTimeGetCurrent()
        print("  warmup done in \(String(format: "%.1f", warmT1 - warmT0))s")

        // Phase 3: Initialize empty window arrays — on-demand loading handles the rest
        for li in 0..<nLayers {
            egGroupWindow.append([MLModel?](repeating: nil, count: egNGroups))
            egCombineWindow.append(nil)
        }
        print("  phase 3: window arrays ready (on-demand loading, \(nLayers) layers)")
    } else {
        let nFFNShards = meta.layers[0].ffnPartials.count + 1  // partials + last
        let totalShards = nLayers * (1 + nFFNShards)  // attn + ffn shards
        print("  eager-loading \(nLayers) attn + \(nLayers * nFFNShards) FFN shards (\(totalShards) total)...")
        for (idx, layer) in meta.layers.enumerated() {
            let attnPath = resolvePath(layer.attn, relativeTo: metaPath)
            let attnModel = try MLModel(contentsOf: URL(fileURLWithPath: attnPath),
                                        configuration: mlConfig)
            attnModels.append(attnModel)
            attnStates.append(attnModel.makeState())

            // Eager-load FFN partial models (O1: eliminates per-token MLModel init)
            var partialModels = [MLModel]()
            for partialPath in layer.ffnPartials {
                let resolved = resolvePath(partialPath, relativeTo: metaPath)
                partialModels.append(try MLModel(contentsOf: URL(fileURLWithPath: resolved),
                                                 configuration: mlConfig))
            }
            ffnPartialModels.append(partialModels)

            let lastPath = resolvePath(layer.ffnLast, relativeTo: metaPath)
            ffnLastModels.append(try MLModel(contentsOf: URL(fileURLWithPath: lastPath),
                                             configuration: mlConfig))

            if (idx + 1) % 5 == 0 || idx == nLayers - 1 {
                let loaded = (idx + 1) * (nFFNShards + 1)
                print("    loaded \(idx + 1)/\(nLayers) layers (\(loaded) shards)")
            }
        }
    }

    // Load ANE LM head shards. This is mandatory for pure-ANE heavy compute.
    var lmHeadModels = [MLModel]()
    var lmHeadSpecs = [LMHeadShardSpec]()
    let useANEHead = true
    if useANEHead {
        let shards = lmHeadShardSpecs
        print("  loading \(shards.count) LM head shards...")
        for spec in shards {
            let path = resolvePath(spec.mlmodelc, relativeTo: metaPath)
            lmHeadModels.append(try MLModel(contentsOf: URL(fileURLWithPath: path),
                                            configuration: mlConfig))
            lmHeadSpecs.append(spec)
        }
        print("    loaded \(shards.count) LM head shards")
    }

    let xArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.dModel)], dataType: .float16)
    let xPtr = xArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let priorMoeArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.dModel)], dataType: .float16)
    let priorMoePtr = priorMoeArr.dataPointer.assumingMemoryBound(to: Float16.self)
    var moeAccumF32 = [Float](repeating: 0, count: meta.dModel)  // Float32 accumulator for partial_moe
    let cosSArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.slidingDHead)], dataType: .float16)
    let sinSArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.slidingDHead)], dataType: .float16)
    let cosSPtr = cosSArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let sinSPtr = sinSArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let cosGArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.globalRotDim)], dataType: .float16)
    let sinGArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.globalRotDim)], dataType: .float16)
    let cosGPtr = cosGArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let sinGPtr = sinGArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let attnMaskArr = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: meta.maxCtx)], dataType: .float16)
    let kvWriteMaskArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.maxCtx), 1], dataType: .float16)
    let attnMaskPtr = attnMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let kvWriteMaskPtr = kvWriteMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
    for index in 0..<meta.maxCtx {
        attnMaskPtr[index] = Float16(-10000.0)
        kvWriteMaskPtr[index] = 0
    }

    // Pre-allocated feature providers — reused every shard call (90×/token → 0 alloc).
    // The backing MLMultiArrays are the same shared buffers mutated in-place each step.
    let attnProvider = try MLDictionaryFeatureProvider(dictionary: [
        "x": MLFeatureValue(multiArray: xArr),
        "cos_s": MLFeatureValue(multiArray: cosSArr),
        "sin_s": MLFeatureValue(multiArray: sinSArr),
        "cos_g": MLFeatureValue(multiArray: cosGArr),
        "sin_g": MLFeatureValue(multiArray: sinGArr),
        "attn_mask": MLFeatureValue(multiArray: attnMaskArr),
        "kv_write_mask": MLFeatureValue(multiArray: kvWriteMaskArr),
    ])
    let p0Provider = try MLDictionaryFeatureProvider(dictionary: [
        "x": MLFeatureValue(multiArray: xArr),
    ])
    let p1Provider = try MLDictionaryFeatureProvider(dictionary: [
        "x": MLFeatureValue(multiArray: xArr),
        "prior_partial_moe": MLFeatureValue(multiArray: priorMoeArr),
    ])

    // Per-expert dispatch buffers (rank-4: 1,D,1,1 for Conv2d ANE models)
    var expertInputArr: MLMultiArray? = nil
    var expertInputPtr: UnsafeMutablePointer<Float16>? = nil
    var expertInputStride1: Int = 1  // channel stride for expert input
    var expertProvider: MLDictionaryFeatureProvider? = nil
    // Combine shard buffers: x (1,D,1,1) + moe_sum (1,D,1,1)
    var combineXArr: MLMultiArray? = nil
    var combineXPtr: UnsafeMutablePointer<Float16>? = nil
    var combineXStride1: Int = 1
    var combineMoeArr: MLMultiArray? = nil
    var combineMoePtr: UnsafeMutablePointer<Float>? = nil
    var combineMoeStride1: Int = 1
    var combineProvider: MLDictionaryFeatureProvider? = nil
    // Accumulator for weighted expert outputs
    var moeAccum: [Float]? = nil

    // Expert-group dispatch buffers
    var egGroupInputArr: MLMultiArray? = nil    // x: (1,D,1,1) fp16
    var egGroupInputPtr: UnsafeMutablePointer<Float16>? = nil
    var egGroupInputStride1: Int = 1
    var egWeightsArr: MLMultiArray? = nil       // expert_weights: (1,G_SIZE,1,1) fp16
    var egWeightsPtr: UnsafeMutablePointer<Float16>? = nil
    var egWeightsStride1: Int = 1
    var egCombineXArr: MLMultiArray? = nil      // combine x: (1,D,1,1) fp16
    var egCombineXPtr: UnsafeMutablePointer<Float16>? = nil
    var egCombineXStride1: Int = 1
    var egCombineMoeArr: MLMultiArray? = nil    // combine moe_sum: (1,D,1,1) fp16
    var egCombineMoePtr: UnsafeMutablePointer<Float16>? = nil
    var egCombineMoeStride1: Int = 1
    var egCombineProvider: MLDictionaryFeatureProvider? = nil
    var egGroupProvider: MLDictionaryFeatureProvider? = nil  // reusable provider for group shard calls

    if usePerExpert {
        expertInputArr = try MLMultiArray(shape: [1, NSNumber(value: meta.dModel), 1, 1], dataType: .float16)
        expertInputPtr = expertInputArr!.dataPointer.assumingMemoryBound(to: Float16.self)
        expertInputStride1 = Int(truncating: expertInputArr!.strides[1])
        expertProvider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: expertInputArr!),
        ])
        combineXArr = try MLMultiArray(shape: [1, NSNumber(value: meta.dModel), 1, 1], dataType: .float16)
        combineXPtr = combineXArr!.dataPointer.assumingMemoryBound(to: Float16.self)
        combineXStride1 = Int(truncating: combineXArr!.strides[1])
        combineMoeArr = try MLMultiArray(shape: [1, NSNumber(value: meta.dModel), 1, 1], dataType: .float32)
        combineMoePtr = combineMoeArr!.dataPointer.assumingMemoryBound(to: Float.self)
        combineMoeStride1 = Int(truncating: combineMoeArr!.strides[1])
        combineProvider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: combineXArr!),
            "moe_sum": MLFeatureValue(multiArray: combineMoeArr!),
        ])
        moeAccum = [Float](repeating: 0, count: meta.dModel)
    }

    if useExpertGroups {
        let dModel = egMeta!.dModel
        // Group shard input: x (1,D,1,1)
        egGroupInputArr = try MLMultiArray(shape: [1, NSNumber(value: dModel), 1, 1], dataType: .float16)
        egGroupInputPtr = egGroupInputArr!.dataPointer.assumingMemoryBound(to: Float16.self)
        egGroupInputStride1 = Int(truncating: egGroupInputArr!.strides[1])
        // Group shard input: expert_weights (1,G_SIZE,1,1)
        egWeightsArr = try MLMultiArray(shape: [1, NSNumber(value: egGroupSize), 1, 1], dataType: .float16)
        egWeightsPtr = egWeightsArr!.dataPointer.assumingMemoryBound(to: Float16.self)
        egWeightsStride1 = Int(truncating: egWeightsArr!.strides[1])
        // Combine shard inputs
        egCombineXArr = try MLMultiArray(shape: [1, NSNumber(value: dModel), 1, 1], dataType: .float16)
        egCombineXPtr = egCombineXArr!.dataPointer.assumingMemoryBound(to: Float16.self)
        egCombineXStride1 = Int(truncating: egCombineXArr!.strides[1])
        egCombineMoeArr = try MLMultiArray(shape: [1, NSNumber(value: dModel), 1, 1], dataType: .float16)
        egCombineMoePtr = egCombineMoeArr!.dataPointer.assumingMemoryBound(to: Float16.self)
        egCombineMoeStride1 = Int(truncating: egCombineMoeArr!.strides[1])
        egCombineProvider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: egCombineXArr!),
            "moe_sum": MLFeatureValue(multiArray: egCombineMoeArr!),
        ])
        egGroupProvider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: egGroupInputArr!),
            "expert_weights": MLFeatureValue(multiArray: egWeightsArr!),
        ])
        moeAccum = [Float](repeating: 0, count: dModel)
    }

    // ANE LM head: pre-allocate input array shaped (1, d_model, 1, 1)
    var headInputArr: MLMultiArray? = nil
    var headInputPtr: UnsafeMutablePointer<Float16>? = nil
    var headInputStride1: Int = 1
    var headProvider: MLDictionaryFeatureProvider? = nil
    if useANEHead {
        headInputArr = try MLMultiArray(shape: [1, NSNumber(value: meta.dModel), 1, 1], dataType: .float16)
        headInputPtr = headInputArr!.dataPointer.assumingMemoryBound(to: Float16.self)
        headInputStride1 = Int(truncating: headInputArr!.strides[1])
        headProvider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden": MLFeatureValue(multiArray: headInputArr!),
        ])
    }

    // Concurrent head dispatch queue (8 independent shards → ANE can pipeline)
    let headQueue = DispatchQueue(label: "gemma.head", attributes: .concurrent)

    // Per-expert timing accumulators (seconds)
    var peTimingRoute: Double = 0
    var peTimingLoad: Double = 0
    var peTimingDispatch: Double = 0
    var peTimingCombine: Double = 0
    var peTimingAttn: Double = 0

    func step(tokenId: Int, pos: Int, needsFullLogits: Bool = false) throws -> (logits: [Float], nextId: Int, nextLogit: Float) {
        embd.writeScaledRow(tokenId, scale: embedScale, into: xPtr)
        makeRoPE(theta: Float(meta.slidingRopeTheta), dim: meta.slidingDHead, pos: pos,
                 cosPtr: cosSPtr, sinPtr: sinSPtr)
        makeRoPE(theta: Float(meta.globalRopeTheta), dim: meta.globalRotDim, pos: pos,
                 cosPtr: cosGPtr, sinPtr: sinGPtr)
        attnMaskPtr[pos] = 0
        if pos > 0 { kvWriteMaskPtr[pos - 1] = 0 }
        kvWriteMaskPtr[pos] = 1

        for layerIdx in 0..<nLayers {
            let tAttn0 = CFAbsoluteTimeGetCurrent()
            let attnResult = try attnModels[layerIdx].prediction(from: attnProvider,
                                                                  using: attnStates[layerIdx])
            let attnHidden = attnResult.featureValue(for: "hidden")!.multiArrayValue!
            copyFlatFloat16(attnHidden, into: xPtr, count: meta.dModel)
            peTimingAttn += CFAbsoluteTimeGetCurrent() - tAttn0

            if traceLayers {
                var sumSq: Float = 0
                for d in 0..<meta.dModel { sumSq += Float(xPtr[d]) * Float(xPtr[d]) }
                let norm = sqrtf(sumSq)
                print("    L\(layerIdx) attn  x[0:4]=[\(xPtr[0]),\(xPtr[1]),\(xPtr[2]),\(xPtr[3])] L2=\(norm)")
            }

            if usePerExpert {
                // Per-expert MoE dispatch: CPU route → concurrent ANE dispatch → weighted accumulate → combine
                let tRoute0 = CFAbsoluteTimeGetCurrent()
                let (topIdx, topW) = layerRouters[layerIdx].route(xPtr, dModel: meta.dModel,
                                                                   rmsEps: Float(meta.rmsNormEps))
                // RMSNorm x before experts: experts have ln2 scale fused but not 1/rms
                var sumSq: Float = 0
                for d in 0..<meta.dModel {
                    let v = Float(xPtr[d])
                    sumSq += v * v
                }
                let invRms = 1.0 / (sumSq / Float(meta.dModel) + Float(meta.rmsNormEps)).squareRoot()

                // Compute normedX (Float32 buffer)
                var normedX = [Float](repeating: 0, count: meta.dModel)
                for d in 0..<meta.dModel {
                    normedX[d] = Float(xPtr[d]) * invRms
                }

                // Zero the MoE accumulator
                for d in 0..<meta.dModel { moeAccum![d] = 0 }

                // --- O3: Concurrent expert dispatch via DispatchGroup ---
                let K = topIdx.count
                let dModel = meta.dModel

                // Pre-load all needed expert models (cache-friendly)
                let tLoad0 = CFAbsoluteTimeGetCurrent()
                peTimingRoute += tLoad0 - tRoute0
                var expertModels = [MLModel]()
                for k in 0..<K {
                    let path = expertPaths[layerIdx][topIdx[k]]
                    if let cached = expertCache[path] {
                        expertModels.append(cached)
                    } else {
                        let model = try MLModel(contentsOf: URL(fileURLWithPath: path),
                                                configuration: mlConfig)
                        expertCache[path] = model
                        if expertCache.count > expertCacheMax {
                            if let evictKey = expertCache.keys.first(where: { $0 != path }) {
                                expertCache.removeValue(forKey: evictKey)
                            }
                        }
                        expertModels.append(model)
                    }
                }

                // Each expert needs its own input buffer (concurrent writes)
                // and its own feature provider (CoreML is not thread-safe per provider).
                let tDispatch0 = CFAbsoluteTimeGetCurrent()
                peTimingLoad += tDispatch0 - tLoad0
                let expertQueue = DispatchQueue(label: "expert-dispatch", attributes: .concurrent)
                let group = DispatchGroup()
                // Per-expert results: (stride-aware output pointer, stride, weight)
                var expertResults = [(UnsafeMutablePointer<Float16>, Int, Float)?](repeating: nil, count: K)
                var expertResultArrays = [MLMultiArray?](repeating: nil, count: K)
                var expertError: Error? = nil

                for k in 0..<K {
                    group.enter()
                    let model = expertModels[k]
                    let w = topW[k]
                    expertQueue.async {
                        do {
                            // Each concurrent call needs its own input buffer
                            let inputArr = try MLMultiArray(shape: [1, NSNumber(value: dModel), 1, 1], dataType: .float16)
                            let inputPtr = inputArr.dataPointer.assumingMemoryBound(to: Float16.self)
                            let inStride1 = Int(truncating: inputArr.strides[1])
                            for d in 0..<dModel {
                                inputPtr[d * inStride1] = Float16(normedX[d])
                            }
                            let provider = try MLDictionaryFeatureProvider(dictionary: [
                                "x": MLFeatureValue(multiArray: inputArr),
                            ])
                            let result = try model.prediction(from: provider)
                            let outArr = result.featureValue(for: "expert_out")!.multiArrayValue!
                            let outPtr = outArr.dataPointer.assumingMemoryBound(to: Float16.self)
                            let outStride = Int(truncating: outArr.strides[1])
                            expertResults[k] = (outPtr, outStride, w)
                            expertResultArrays[k] = outArr  // keep alive
                        } catch {
                            expertError = error
                        }
                        group.leave()
                    }
                }
                group.wait()
                if let err = expertError { throw err }

                // Accumulate results (sequential, after all experts complete)
                for k in 0..<K {
                    if let (outPtr, stride1, w) = expertResults[k] {
                        for d in 0..<dModel {
                            moeAccum![d] += Float(outPtr[d * stride1]) * w
                        }
                    }
                }

                // Write accumulated MoE output into combine input buffer (rank-4: 1,D,1,1)
                let tCombine0 = CFAbsoluteTimeGetCurrent()
                peTimingDispatch += tCombine0 - tDispatch0
                for d in 0..<meta.dModel { combineMoePtr![d * combineMoeStride1] = moeAccum![d] }
                // Copy x into combine x buffer (stride-aware)
                for d in 0..<meta.dModel { combineXPtr![d * combineXStride1] = xPtr[d] }

                // Run combine shard: dense MLP + norms + residual + layer_scalar
                let combineResult = try combineModels[layerIdx].prediction(from: combineProvider!)
                let combineHidden = combineResult.featureValue(for: "hidden")!.multiArrayValue!
                let combinePtr = combineHidden.dataPointer.assumingMemoryBound(to: Float16.self)
                let combineStride1 = Int(truncating: combineHidden.strides[1])
                for d in 0..<meta.dModel {
                    xPtr[d] = combinePtr[d * combineStride1]
                }
                peTimingCombine += CFAbsoluteTimeGetCurrent() - tCombine0
            } else if useExpertGroups {
                // Expert-GROUP dispatch: CPU route → determine touched groups → run group shards → accumulate → combine
                let tRoute0 = CFAbsoluteTimeGetCurrent()
                let dModel = egMeta!.dModel
                let (topIdx, topW) = egRouters[layerIdx].route(xPtr, dModel: dModel,
                                                                rmsEps: Float(egMeta!.rmsNormEps))
                // RMSNorm x before experts: experts have ln2 scale fused but not 1/rms
                var sumSq: Float = 0
                for d in 0..<dModel {
                    let v = Float(xPtr[d])
                    sumSq += v * v
                }
                let invRms = 1.0 / (sumSq / Float(dModel) + Float(egMeta!.rmsNormEps)).squareRoot()

                // Write normed x into group input buffer (stride-aware, rank-4)
                for d in 0..<dModel {
                    egGroupInputPtr![d * egGroupInputStride1] = Float16(Float(xPtr[d]) * invRms)
                }

                // Zero the MoE accumulator (Float32)
                for d in 0..<dModel { moeAccum![d] = 0 }

                let tDispatch0 = CFAbsoluteTimeGetCurrent()
                peTimingRoute += tDispatch0 - tRoute0

                // Determine which groups are touched and build per-group weight vectors
                // Group g covers experts [g*egGroupSize .. (g+1)*egGroupSize)
                for g in 0..<egNGroups {
                    let expertStart = g * egGroupSize

                    // Build expert_weights for this group: set active expert weights, 0 for inactive
                    var hasActive = false
                    for local in 0..<egGroupSize {
                        egWeightsPtr![local * egWeightsStride1] = 0
                    }
                    for k in 0..<topIdx.count {
                        let globalExpert = topIdx[k]
                        if globalExpert >= expertStart && globalExpert < expertStart + egGroupSize {
                            let localIdx = globalExpert - expertStart
                            egWeightsPtr![localIdx * egWeightsStride1] = Float16(topW[k])
                            hasActive = true
                        }
                    }

                    if !hasActive { continue }  // skip groups with no active experts

                    // Run the group shard (reuse pre-allocated provider — arrays mutated in-place)
                    // Sliding window: ensure this layer's models are loaded
                    if egGroupWindow[layerIdx][g] == nil {
                        egGroupWindow[layerIdx][g] = try MLModel(contentsOf: URL(fileURLWithPath: egGroupPaths[layerIdx][g]),
                                                                  configuration: mlConfig)
                    }
                    let groupResult = try egGroupWindow[layerIdx][g]!.prediction(from: egGroupProvider!)
                    let groupOut = groupResult.featureValue(for: "group_out")!.multiArrayValue!
                    let outPtr = groupOut.dataPointer.assumingMemoryBound(to: Float16.self)
                    let outStride1 = Int(truncating: groupOut.strides[1])

                    // Accumulate into Float32 buffer
                    for d in 0..<dModel {
                        moeAccum![d] += Float(outPtr[d * outStride1])
                    }
                }

                // Write accumulated MoE output into combine input (FP16 for combine shard)
                let tCombine0 = CFAbsoluteTimeGetCurrent()
                peTimingDispatch += tCombine0 - tDispatch0
                for d in 0..<dModel { egCombineMoePtr![d * egCombineMoeStride1] = Float16(moeAccum![d]) }
                for d in 0..<dModel { egCombineXPtr![d * egCombineXStride1] = xPtr[d] }

                // Sliding window: ensure combine model is loaded
                if egCombineWindow[layerIdx] == nil {
                    egCombineWindow[layerIdx] = try MLModel(contentsOf: URL(fileURLWithPath: egCombinePaths[layerIdx]),
                                                             configuration: mlConfig)
                }
                let combineResult = try egCombineWindow[layerIdx]!.prediction(from: egCombineProvider!)
                let combineHidden = combineResult.featureValue(for: "hidden")!.multiArrayValue!
                let combinePtr = combineHidden.dataPointer.assumingMemoryBound(to: Float16.self)
                let combineStride1 = Int(truncating: combineHidden.strides[1])
                for d in 0..<dModel {
                    xPtr[d] = combinePtr[d * combineStride1]
                }
                peTimingCombine += CFAbsoluteTimeGetCurrent() - tCombine0

                // Evict this layer's group+combine models immediately to stay under ANE slot limit.
                // Warm E5 cache means next layer's on-demand loads are ~0.013s each.
                for g in 0..<egNGroups { egGroupWindow[layerIdx][g] = nil }
                egCombineWindow[layerIdx] = nil
            } else {
                // Run FFN partial shards (pre-loaded, O1 optimization).
                // Accumulate partial_moe outputs in Float32, then convert to Float16 for last shard.
                for d in 0..<meta.dModel { moeAccumF32[d] = 0 }  // zero Float32 accumulator
                for partialModel in ffnPartialModels[layerIdx] {
                    let pResult = try partialModel.prediction(from: p0Provider)
                    let partialMoe = pResult.featureValue(for: "partial_moe")!.multiArrayValue!
                    if traceLayers && layerIdx == 0 {
                        print("      partial shape=\(partialMoe.shape) strides=\(partialMoe.strides)")
                    }
                    let pPtr = partialMoe.dataPointer.assumingMemoryBound(to: Float16.self)
                    for d in 0..<meta.dModel {
                        moeAccumF32[d] += Float(pPtr[d])
                    }
                }
                // Convert accumulated Float32 → Float16 for the last shard input
                for d in 0..<meta.dModel { priorMoePtr[d] = Float16(moeAccumF32[d]) }

                let lastResult = try ffnLastModels[layerIdx].prediction(from: p1Provider)
                let layerHidden = lastResult.featureValue(for: "hidden")!.multiArrayValue!
                copyFlatFloat16(layerHidden, into: xPtr, count: meta.dModel)

                if traceLayers {
                    // Also print partial_moe stats and FFN output strides
                    var sumSq: Float = 0
                    for d in 0..<meta.dModel { sumSq += Float(xPtr[d]) * Float(xPtr[d]) }
                    let norm = sqrtf(sumSq)
                    var moeSumSq: Float = 0
                    for d in 0..<meta.dModel { moeSumSq += Float(priorMoePtr[d]) * Float(priorMoePtr[d]) }
                    let moeNorm = sqrtf(moeSumSq)
                    print("    L\(layerIdx) ffn   x[0:4]=[\(xPtr[0]),\(xPtr[1]),\(xPtr[2]),\(xPtr[3])] L2=\(norm) moeL2=\(moeNorm)")
                }
            }
        }

        if useANEHead {
            // Stride-aware copy into head input (rank-4: 1,D,1,1)
            for d in 0..<meta.dModel { headInputPtr![d * headInputStride1] = xPtr[d] }

            if needsFullLogits {
                // Full logit path: sequential dispatch, collect all 262K logits
                var logits = [Float](repeating: 0, count: meta.vocabSize)
                let softcapF = Float(meta.softcap)
                for s in 0..<lmHeadModels.count {
                    let shardResult = try lmHeadModels[s].prediction(from: headProvider!)
                    let partialArr = shardResult.featureValue(for: "logits")!.multiArrayValue!
                    let partialPtr = partialArr.dataPointer.assumingMemoryBound(to: Float16.self)
                    let vocabStart = lmHeadSpecs[s].vocabStart
                    let chunkSize = lmHeadSpecs[s].vocabEnd - vocabStart
                    let stride1 = Int(truncating: partialArr.strides[1])
                    for i in 0..<chunkSize {
                        let raw = Float(partialPtr[i * stride1])
                        logits[vocabStart + i] = tanh(raw / softcapF) * softcapF
                    }
                }
                let result = embd.argmax(logits)
                return (logits: logits, nextId: result.tokenId, nextLogit: result.logit)
            }

            // Greedy fast path: concurrent dispatch, per-shard max, skip tanh (monotonic)
            let nShards = lmHeadModels.count
            var shardBestRaw = [Float](repeating: -Float.infinity, count: nShards)
            var shardBestGlobalIdx = [Int](repeating: 0, count: nShards)
            var shardError: Error? = nil

            let group = DispatchGroup()
            for s in 0..<nShards {
                group.enter()
                headQueue.async {
                    do {
                        let shardResult = try lmHeadModels[s].prediction(from: headProvider!)
                        let partialArr = shardResult.featureValue(for: "logits")!.multiArrayValue!
                        let partialPtr = partialArr.dataPointer.assumingMemoryBound(to: Float16.self)
                        let vocabStart = lmHeadSpecs[s].vocabStart
                        let chunkSize = lmHeadSpecs[s].vocabEnd - vocabStart
                        let stride1 = Int(truncating: partialArr.strides[1])
                        // Find local max (no tanh — it's monotonic)
                        var bestRaw: Float = -Float.infinity
                        var bestLocal = 0
                        for i in 0..<chunkSize {
                            let raw = Float(partialPtr[i * stride1])
                            if raw > bestRaw { bestRaw = raw; bestLocal = i }
                        }
                        shardBestRaw[s] = bestRaw
                        shardBestGlobalIdx[s] = vocabStart + bestLocal
                    } catch {
                        shardError = error
                    }
                    group.leave()
                }
            }
            group.wait()
            if let err = shardError { throw err }

            // Global argmax across 8 shard winners
            var globalBestRaw: Float = -Float.infinity
            var globalBestIdx = 0
            for s in 0..<nShards {
                if shardBestRaw[s] > globalBestRaw {
                    globalBestRaw = shardBestRaw[s]
                    globalBestIdx = shardBestGlobalIdx[s]
                }
            }
            let softcapF = Float(meta.softcap)
            let nextLogit = tanh(globalBestRaw / softcapF) * softcapF
            return (logits: [], nextId: globalBestIdx, nextLogit: nextLogit)
        } else {
            let projectedHidden = finalNormProjectedHidden(hidden: xArr,
                                                           gamma: gamma,
                                                           eps: Float(meta.rmsNormEps),
                                                           softcap: Float(meta.softcap))
            let logits = embd.fullLogits(projectedHidden: projectedHidden, softcap: Float(meta.softcap))
            let result = embd.argmax(logits)
            return (logits: logits, nextId: result.tokenId, nextLogit: result.logit)
        }
    }

    if promptText == nil {
        print("  prompt ids (\(promptIds.count)): \(promptIds)")
    }
    print("  greedy decode: \(nNew) new token ids")
    var promptLogitsFlat: [Float]? = nil
    if dumpPromptLogitsPrefix != nil {
        promptLogitsFlat = []
        promptLogitsFlat!.reserveCapacity(promptIds.count * meta.vocabSize)
    }
    var decodeLogitsFlat: [Float]? = nil
    if dumpDecodeLogitsPrefix != nil {
        decodeLogitsFlat = []
        decodeLogitsFlat!.reserveCapacity(max(0, nNew) * meta.vocabSize)
    }
    var last = (logits: [Float](), nextId: 0, nextLogit: Float(0))
    let promptStart = CFAbsoluteTimeGetCurrent()
    let dumpActive = promptLogitsFlat != nil || decodeLogitsFlat != nil
    for (pos, tokenId) in promptIds.enumerated() {
        last = try step(tokenId: tokenId, pos: pos, needsFullLogits: dumpActive)
        if traceDecode {
            let nextLogitText = String(format: "%.4f", Double(last.nextLogit))
            var tokenText = ""
            if let tokenizer {
                tokenText = " text=\(String(reflecting: tokenizer.decode([tokenId])))"
            }
            print("  prime[\(pos)] token=\(tokenId) next=\(last.nextId) logit=\(nextLogitText)\(tokenText)")
        }
        promptLogitsFlat?.append(contentsOf: last.logits)
    }
    let promptElapsed = CFAbsoluteTimeGetCurrent() - promptStart
    let promptTokPerSec = Double(promptIds.count) / promptElapsed
    print(String(format: "  prompt: %d tok in %.3f s  (%.2f tok/s, TTFT=%.0f ms)",
                 promptIds.count, promptElapsed, promptTokPerSec, promptElapsed * 1000))
    if let dumpPromptLogitsPrefix, let promptLogitsFlat {
        try writePromptLogitsDump(prefix: dumpPromptLogitsPrefix,
                                  promptIds: promptIds,
                                  promptText: promptText,
                                  vocabSize: meta.vocabSize,
                                  logitsFlat: promptLogitsFlat)
    }

    var generated = [Int]()
    var pos = promptIds.count
    let decodeStart = CFAbsoluteTimeGetCurrent()
    for stepIndex in 0..<nNew {
        let emittedTokenId = last.nextId
        generated.append(emittedTokenId)
        if traceDecode {
            let nextLogitText = String(format: "%.4f", Double(last.nextLogit))
            var tokenText = ""
            if let tokenizer {
                tokenText = " text=\(String(reflecting: tokenizer.decode([emittedTokenId])))"
            }
            print("  decode[\(stepIndex)] -> id=\(emittedTokenId) logit=\(nextLogitText)\(tokenText)")
        }
        let emittedEos = emittedTokenId == meta.eosTokenId
        if emittedEos && dumpDecodeLogitsPrefix == nil { break }
        last = try step(tokenId: emittedTokenId, pos: pos, needsFullLogits: dumpActive)
        pos += 1
        decodeLogitsFlat?.append(contentsOf: last.logits)
        if emittedEos { break }
    }
    let decodeElapsed = CFAbsoluteTimeGetCurrent() - decodeStart
    let decodeTokPerSec = generated.count > 0 ? Double(generated.count) / decodeElapsed : 0
    print(String(format: "  decode: %d tok in %.3f s  (%.2f tok/s)",
                 generated.count, decodeElapsed, decodeTokPerSec))
    if let dumpDecodeLogitsPrefix, let decodeLogitsFlat {
        try writeDecodeLogitsDump(prefix: dumpDecodeLogitsPrefix,
                                  promptIds: promptIds,
                                  promptText: promptText,
                                  generatedIds: generated,
                                  nNewRequested: nNew,
                                  vocabSize: meta.vocabSize,
                                  logitsFlat: decodeLogitsFlat,
                                  announce: true)
    }

    let lastLogitText = String(format: "%.4f", Double(last.nextLogit))
    print("  generated ids : \(generated)")
    if let expectedGeneratedIds {
        let generatedIdsOk = idsMatch(generated, expected: expectedGeneratedIds)
        print("  expect generated ids: \(expectedGeneratedIds) -> \(generatedIdsOk ? "PASS" : "FAIL")")
        if !generatedIdsOk {
            throw NSError(domain: "gemma_ane", code: 3,
                          userInfo: [NSLocalizedDescriptionKey:
                            "generated ids mismatch: actual=\(generated) expected=\(expectedGeneratedIds)"])
        }
    }
    if let tokenizer {
        print("  completion str: \(String(reflecting: tokenizer.decode(generated)))")
        print("  full str      : \(String(reflecting: tokenizer.decode(promptIds + generated)))")
    }
    print("  last logit    : \(lastLogitText)")
    if usePerExpert {
        let total = peTimingAttn + peTimingRoute + peTimingLoad + peTimingDispatch + peTimingCombine
        print(String(format: "  pe-timing: attn=%.2fs route=%.3fs load=%.2fs dispatch=%.2fs combine=%.2fs total=%.2fs",
                     peTimingAttn, peTimingRoute, peTimingLoad, peTimingDispatch, peTimingCombine, total))
    }
    print("  runtime path ready: text and prompt-id entrypoints are both wired")
}

do {
    try main()
} catch {
    fputs("FATAL: \(error)\n", stderr)
    exit(1)
}