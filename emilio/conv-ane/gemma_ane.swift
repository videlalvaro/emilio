// Gemma ANE runtime.
//
// This is the first real Swift-side prompt-id runtime for T4.2. It loads the
// exported Swift head artifacts plus the 3 CoreML shards, primes shard-local KV
// state with prompt token IDs, then greedily decodes more token IDs.
//
// Build:
//   swiftc -O -framework CoreML -o gemma_ane gemma_ane.swift
//
// Usage:
//   ./gemma_ane --meta python/moe/out/gemma_swift_head_meta.json \
//       --prompt-ids 2,818,5279,529,7001,563 --n-new 8

import CoreML
import Foundation

struct GemmaShardSpec: Decodable {
    let start: Int
    let end: Int
    let path: String
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
    let shards: [GemmaShardSpec]

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
        case shards
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

    func fullLogits(projectedHidden: [Float], softcap: Float) -> [Float] {
        precondition(projectedHidden.count == dim, "projected hidden dim mismatch")
        var logits = [Float](repeating: 0, count: vocab)
        for tokenId in 0..<vocab {
            let start = tokenId * dim
            var raw: Float = 0
            for index in 0..<dim {
                raw += Float(Float16(bitPattern: data[start + index])) * projectedHidden[index]
            }
            logits[tokenId] = softcap > 0 ? tanhf(raw) * softcap : raw
        }
        return logits
    }

    func argmax(_ logits: [Float]) -> (tokenId: Int, logit: Float) {
        precondition(logits.count == vocab, "logits vocab mismatch")
        var bestToken = 0
        var bestLogit: Float = -.infinity
        for tokenId in 0..<vocab {
            let logit = logits[tokenId]
            if logit > bestLogit {
                bestLogit = logit
                bestToken = tokenId
            }
        }
        return (bestToken, bestLogit)
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
    var hidden32 = [Float](repeating: 0, count: gamma.count)
    var meanSquare: Float = 0
    for index in 0..<gamma.count {
        let value = Float(hiddenPtr[flatOffset(for: index,
                                               shape: shape,
                                               strides: strides)])
        hidden32[index] = value
        meanSquare += value * value
    }
    meanSquare /= Float(gamma.count)
    let rms = sqrtf(meanSquare + eps)
    let gammaOverRms = gamma.scaledValues(dividingBy: rms * (softcap > 0 ? softcap : 1))
    var projected = [Float](repeating: 0, count: gamma.count)
    for index in 0..<gamma.count {
        projected[index] = hidden32[index] * gammaOverRms[index]
    }
    return projected
}

func copyFlatFloat16(_ source: MLMultiArray,
                     into target: UnsafeMutablePointer<Float16>,
                     count: Int) {
    precondition(source.count >= count, "source hidden smaller than d_model")
    let sourcePtr = source.dataPointer.assumingMemoryBound(to: Float16.self)
    let shape = source.shape.map { Int(truncating: $0) }
    let strides = source.strides.map { Int(truncating: $0) }
    for index in 0..<count {
        target[index] = sourcePtr[flatOffset(for: index,
                                             shape: shape,
                                             strides: strides)]
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
    var dumpHiddenBoundaryPrefix: String? = nil
    var dumpHiddenBoundarySteps = Set<Int>()
    var traceDecode = false
    var nNew = 8
    var inspectOnly = false
    var tokenizeOnly = false

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
        case "--dump-hidden-boundary-prefix":
            dumpHiddenBoundaryPrefix = args[i + 1]
            i += 2
        case "--dump-hidden-boundary-steps":
            dumpHiddenBoundarySteps = parseIntSet(args[i + 1])
            i += 2
        case "--trace-decode":
            traceDecode = true
            i += 1
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
        if dumpHiddenBoundaryPrefix != nil && dumpHiddenBoundarySteps.isEmpty {
                throw NSError(domain: "gemma_ane", code: 5,
                                            userInfo: [NSLocalizedDescriptionKey:
                                                "--dump-hidden-boundary-prefix requires --dump-hidden-boundary-steps"])
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
    print("  shards:")
    for shard in meta.shards {
        let shardPath = resolvePath(shard.path, relativeTo: metaPath)
        let exists = FileManager.default.fileExists(atPath: shardPath)
        print("    [\(shard.start),\(shard.end)) \(shardPath) exists=\(exists)")
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

    let sortedShards = meta.shards.sorted { lhs, rhs in lhs.start < rhs.start }
    precondition(!sortedShards.isEmpty, "no shards configured")
    precondition(sortedShards.first!.start == 0, "first shard must start at 0")
    precondition(!promptIds.isEmpty, "prompt is empty")
    precondition(promptIds.count + nNew <= meta.maxCtx,
                 "prompt length \(promptIds.count) + n_new \(nNew) exceeds max_ctx \(meta.maxCtx)")
    for index in 1..<sortedShards.count {
        precondition(sortedShards[index - 1].end == sortedShards[index].start,
                     "shards must be contiguous")
    }

    let mlConfig = MLModelConfiguration()
    mlConfig.computeUnits = .cpuAndNeuralEngine
    var shardModels = [MLModel]()
    var shardStates = [MLState]()
    var shardTapOutputs = [[String]]()
    var hiddenBoundaryStageNames = [String]()
    for shard in sortedShards {
        let shardPath = resolvePath(shard.path, relativeTo: metaPath)
        let model = try MLModel(contentsOf: URL(fileURLWithPath: shardPath), configuration: mlConfig)
        let finalStageName = "hidden_l\(shard.end)"
        let tapOutputs = model.modelDescription.outputDescriptionsByName.keys
            .compactMap { outputName -> String? in
                guard outputName != "hidden",
                      let stage = hiddenStageKey(outputName),
                      let boundary = Optional(stage.boundary),
                      boundary > shard.start,
                      boundary < shard.end,
                      outputName != finalStageName else {
                    return nil
                }
                return outputName
            }
            .sorted { lhs, rhs in
                let lhsKey = hiddenStageKey(lhs)!
                let rhsKey = hiddenStageKey(rhs)!
                if lhsKey.boundary != rhsKey.boundary {
                    return lhsKey.boundary < rhsKey.boundary
                }
                return lhsKey.suffix < rhsKey.suffix
            }
        shardModels.append(model)
        shardStates.append(model.makeState())
        shardTapOutputs.append(tapOutputs)
        hiddenBoundaryStageNames.append(contentsOf: tapOutputs)
        hiddenBoundaryStageNames.append(finalStageName)
    }
    hiddenBoundaryStageNames.append("projected_hidden")

    let xArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.dModel)], dataType: .float16)
    let xPtr = xArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let cosSArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.slidingDHead)], dataType: .float16)
    let sinSArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.slidingDHead)], dataType: .float16)
    let cosGArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.globalRotDim)], dataType: .float16)
    let sinGArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.globalRotDim)], dataType: .float16)
    let cosSPtr = cosSArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let sinSPtr = sinSArr.dataPointer.assumingMemoryBound(to: Float16.self)
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

    func step(tokenId: Int,
              pos: Int,
              captureHiddenBoundaries: Bool = false) throws -> (logits: [Float], nextId: Int, nextLogit: Float, hiddenSnapshot: HiddenBoundarySnapshot?) {
        embd.writeScaledRow(tokenId, scale: embedScale, into: xPtr)
        makeRoPE(theta: Float(meta.slidingRopeTheta), dim: meta.slidingDHead, pos: pos,
                 cosPtr: cosSPtr, sinPtr: sinSPtr)
        makeRoPE(theta: Float(meta.globalRopeTheta), dim: meta.globalRotDim, pos: pos,
                 cosPtr: cosGPtr, sinPtr: sinGPtr)
        attnMaskPtr[pos] = 0
        if pos > 0 { kvWriteMaskPtr[pos - 1] = 0 }
        kvWriteMaskPtr[pos] = 1

        var hiddenValue = MLFeatureValue(multiArray: xArr)
        var shardHiddenSnapshots = [[Float]]()
        if captureHiddenBoundaries {
            shardHiddenSnapshots.reserveCapacity(hiddenBoundaryStageNames.count - 1)
        }
        for (index, pair) in zip(shardModels, shardStates).enumerated() {
            let (model, state) = pair
            let features: [String: MLFeatureValue] = [
                "x": hiddenValue,
                "cos_s": MLFeatureValue(multiArray: cosSArr),
                "sin_s": MLFeatureValue(multiArray: sinSArr),
                "cos_g": MLFeatureValue(multiArray: cosGArr),
                "sin_g": MLFeatureValue(multiArray: sinGArr),
                "attn_mask": MLFeatureValue(multiArray: attnMaskArr),
                "kv_write_mask": MLFeatureValue(multiArray: kvWriteMaskArr),
            ]
            let provider = try MLDictionaryFeatureProvider(dictionary: features)
            let result = try model.prediction(from: provider, using: state)
            let shardHidden = result.featureValue(for: "hidden")!.multiArrayValue!
            if captureHiddenBoundaries {
                for tapName in shardTapOutputs[index] {
                    guard let tapHidden = result.featureValue(for: tapName)?.multiArrayValue else {
                        throw NSError(domain: "gemma_ane", code: 6,
                                      userInfo: [NSLocalizedDescriptionKey:
                                        "missing tapped hidden output \(tapName) from shard \(index)"])
                    }
                    shardHiddenSnapshots.append(float32Array(from: tapHidden, count: meta.dModel))
                }
                shardHiddenSnapshots.append(float32Array(from: shardHidden, count: meta.dModel))
            }
            if index + 1 < shardModels.count {
                copyFlatFloat16(shardHidden, into: xPtr, count: meta.dModel)
                hiddenValue = MLFeatureValue(multiArray: xArr)
            } else {
                hiddenValue = result.featureValue(for: "hidden")!
            }
        }

        let hidden = hiddenValue.multiArrayValue!
        let projectedHidden = finalNormProjectedHidden(hidden: hidden,
                                                       gamma: gamma,
                                                       eps: Float(meta.rmsNormEps),
                                                       softcap: Float(meta.softcap))
        let logits = embd.fullLogits(projectedHidden: projectedHidden, softcap: Float(meta.softcap))
        let result = embd.argmax(logits)
        let hiddenSnapshot = captureHiddenBoundaries
            ? HiddenBoundarySnapshot(shardHidden: shardHiddenSnapshots, projectedHidden: projectedHidden)
            : nil
        return (logits: logits, nextId: result.tokenId, nextLogit: result.logit, hiddenSnapshot: hiddenSnapshot)
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
    var hiddenBoundaryDecodeSteps = [Int]()
    var hiddenBoundaryAbsolutePositions = [Int]()
    var hiddenBoundaryEmittedTokenIds = [Int]()
    var hiddenBoundaryGeneratedPrefixes = [[Int]]()
    var hiddenBoundaryFlat = [Float]()
    if dumpHiddenBoundaryPrefix != nil {
        hiddenBoundaryFlat.reserveCapacity(max(0, dumpHiddenBoundarySteps.count)
                                           * hiddenBoundaryStageNames.count
                                           * meta.dModel)
    }
    var last = (logits: [Float](), nextId: 0, nextLogit: Float(0), hiddenSnapshot: Optional<HiddenBoundarySnapshot>.none)
    for (pos, tokenId) in promptIds.enumerated() {
        last = try step(tokenId: tokenId, pos: pos)
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
    if let dumpPromptLogitsPrefix, let promptLogitsFlat {
        try writePromptLogitsDump(prefix: dumpPromptLogitsPrefix,
                                  promptIds: promptIds,
                                  promptText: promptText,
                                  vocabSize: meta.vocabSize,
                                  logitsFlat: promptLogitsFlat)
    }

    var generated = [Int]()
    var pos = promptIds.count
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
        let captureHiddenBoundaries = dumpHiddenBoundaryPrefix != nil
            && dumpHiddenBoundarySteps.contains(stepIndex)
        last = try step(tokenId: emittedTokenId, pos: pos, captureHiddenBoundaries: captureHiddenBoundaries)
        let absolutePos = pos
        pos += 1
        decodeLogitsFlat?.append(contentsOf: last.logits)
        if let dumpDecodeLogitsPrefix, let decodeLogitsFlat {
            try writeDecodeLogitsDump(prefix: dumpDecodeLogitsPrefix,
                                      promptIds: promptIds,
                                      promptText: promptText,
                                      generatedIds: generated,
                                      nNewRequested: nNew,
                                      vocabSize: meta.vocabSize,
                                      logitsFlat: decodeLogitsFlat,
                                      announce: false)
        }
        if let dumpHiddenBoundaryPrefix,
           captureHiddenBoundaries,
           let hiddenSnapshot = last.hiddenSnapshot {
            hiddenBoundaryDecodeSteps.append(stepIndex)
            hiddenBoundaryAbsolutePositions.append(absolutePos)
            hiddenBoundaryEmittedTokenIds.append(emittedTokenId)
            hiddenBoundaryGeneratedPrefixes.append(generated)
            for stageHidden in hiddenSnapshot.shardHidden {
                hiddenBoundaryFlat.append(contentsOf: stageHidden)
            }
            hiddenBoundaryFlat.append(contentsOf: hiddenSnapshot.projectedHidden)
            try writeHiddenBoundaryDump(prefix: dumpHiddenBoundaryPrefix,
                                        promptIds: promptIds,
                                        promptText: promptText,
                                        dim: meta.dModel,
                                        stageNames: hiddenBoundaryStageNames,
                                        decodeSteps: hiddenBoundaryDecodeSteps,
                                        absolutePositions: hiddenBoundaryAbsolutePositions,
                                        emittedTokenIds: hiddenBoundaryEmittedTokenIds,
                                        generatedIdsPrefixes: hiddenBoundaryGeneratedPrefixes,
                                        hiddenFlat: hiddenBoundaryFlat,
                                        announce: false)
        }
        if emittedEos { break }
    }
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
    if let dumpHiddenBoundaryPrefix, !hiddenBoundaryDecodeSteps.isEmpty {
        try writeHiddenBoundaryDump(prefix: dumpHiddenBoundaryPrefix,
                                    promptIds: promptIds,
                                    promptText: promptText,
                                    dim: meta.dModel,
                                    stageNames: hiddenBoundaryStageNames,
                                    decodeSteps: hiddenBoundaryDecodeSteps,
                                    absolutePositions: hiddenBoundaryAbsolutePositions,
                                    emittedTokenIds: hiddenBoundaryEmittedTokenIds,
                                    generatedIdsPrefixes: hiddenBoundaryGeneratedPrefixes,
                                    hiddenFlat: hiddenBoundaryFlat,
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
    print("  runtime path ready: text and prompt-id entrypoints are both wired")
}

do {
    try main()
} catch {
    fputs("FATAL: \(error)\n", stderr)
    exit(1)
}