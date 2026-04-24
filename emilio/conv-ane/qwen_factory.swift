// QwenFactory: Multi-shard Qwen2.5 inference on Apple Neural Engine.
//
// Factory-level runtime that loads N sharded CoreML models from a master
// meta JSON, chains them through ANE, and does final norm + LM head on host.
//
// Follows the proven Gemma multi-shard pattern:
//   - Each shard: stateful KV cache, inputs (x, rope, masks), outputs hidden
//   - Host: embedding lookup → chain shards → final norm → LM head → sample
//   - Backward compatible: single-shard meta works as a monolithic model
//
// Build:
//   swiftc -O -framework CoreML -framework Accelerate \
//          -o qwen_factory qwen_factory.swift
//
// Usage:
//   ./qwen_factory --meta path/to/meta.json [--prompt "text"] [--max-tokens N]

import CoreML
import Foundation
import Accelerate

// MARK: - Meta JSON

struct QwenShardSpec: Decodable {
    let start: Int
    let end: Int
    let path: String
}

struct QwenFactoryMeta: Decodable {
    let artifactsVersion: Int
    let modelFamily: String
    let dModel: Int
    let nHeads: Int
    let nKvHeads: Int
    let dHead: Int
    let dFf: Int
    let vocabSize: Int
    let nLayers: Int
    let maxSeqLen: Int
    let rmsNormEps: Double
    let ropeFreqBase: Double
    let eosTokenId: Int
    let bosTokenId: Int
    let tieWordEmbeddings: Bool
    let quantBits: Int
    let embedBin: String
    let finalNormBin: String
    let lmHeadBin: String
    let tokenizerJson: String
    let ropeCos: [[Double]]
    let ropeSin: [[Double]]
    let shards: [QwenShardSpec]

    enum CodingKeys: String, CodingKey {
        case artifactsVersion = "artifacts_version"
        case modelFamily = "model_family"
        case dModel = "d_model"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case dHead = "d_head"
        case dFf = "d_ff"
        case vocabSize = "vocab_size"
        case nLayers = "n_layers"
        case maxSeqLen = "max_seq_len"
        case rmsNormEps = "rms_norm_eps"
        case ropeFreqBase = "rope_freq_base"
        case eosTokenId = "eos_token_id"
        case bosTokenId = "bos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantBits = "quant_bits"
        case embedBin = "embed_bin"
        case finalNormBin = "final_norm_bin"
        case lmHeadBin = "lm_head_bin"
        case tokenizerJson = "tokenizer_json"
        case ropeCos = "rope_cos"
        case ropeSin = "rope_sin"
        case shards
    }
}

// MARK: - BPE Tokenizer (same as qwen_ane.swift, factored out)

class BPETokenizer {
    let vocab: [String]
    let tokenToId: [String: Int]
    let merges: [(String, String)]
    let mergeRank: [String: Int]
    let eosId: Int
    let bosId: Int
    let specialTokens: [(String, Int)]

    static let byteToUnicode: [UInt8: Character] = {
        var map = [UInt8: Character]()
        var n = 256
        for b: UInt8 in 0...255 {
            if (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b) {
                map[b] = Character(Unicode.Scalar(UInt32(b))!)
            } else {
                map[b] = Character(Unicode.Scalar(UInt32(n))!)
                n += 1
            }
        }
        return map
    }()

    static let unicodeToByte: [Character: UInt8] = {
        var map = [Character: UInt8]()
        for (k, v) in byteToUnicode { map[v] = k }
        return map
    }()

    init(jsonPath: String) {
        let data = try! Data(contentsOf: URL(fileURLWithPath: jsonPath))
        let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]

        let tokens = json["tokens"] as! [String]
        self.vocab = tokens
        var t2i = [String: Int]()
        for (i, tok) in tokens.enumerated() { t2i[tok] = i }
        self.tokenToId = t2i

        let mergeStrings = json["merges"] as! [String]
        var mergeList = [(String, String)]()
        var rankMap = [String: Int]()
        for (i, m) in mergeStrings.enumerated() {
            if let spaceIdx = m.firstIndex(of: " ") {
                let left = String(m[m.startIndex..<spaceIdx])
                let right = String(m[m.index(after: spaceIdx)...])
                mergeList.append((left, right))
                rankMap[m] = i
            }
        }
        self.merges = mergeList
        self.mergeRank = rankMap
        self.eosId = (json["eos_token_id"] as? Int) ?? 151645
        self.bosId = (json["bos_token_id"] as? Int) ?? 151643

        var specials = [(String, Int)]()
        for (i, tok) in tokens.enumerated() where i >= 151643 {
            specials.append((tok, i))
        }
        specials.sort { $0.0.count > $1.0.count }
        self.specialTokens = specials
    }

    func encode(_ text: String) -> [Int] {
        var ids = [Int]()
        var cursor = text.startIndex
        var runStart = cursor
        while cursor < text.endIndex {
            var matched: (String, Int)? = nil
            for cand in specialTokens {
                if text[cursor...].hasPrefix(cand.0) { matched = cand; break }
            }
            if let m = matched {
                if runStart < cursor {
                    ids.append(contentsOf: bpeEncode(String(text[runStart..<cursor])))
                }
                ids.append(m.1)
                cursor = text.index(cursor, offsetBy: m.0.count)
                runStart = cursor
            } else {
                cursor = text.index(after: cursor)
            }
        }
        if runStart < text.endIndex {
            ids.append(contentsOf: bpeEncode(String(text[runStart..<text.endIndex])))
        }
        return ids
    }

    private func bpeEncode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        let utf8Bytes = Array(text.utf8)
        let gptChars = utf8Bytes.map { BPETokenizer.byteToUnicode[$0]! }
        var word = gptChars.map { String($0) }

        while word.count >= 2 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(word.count - 1) {
                let pair = "\(word[i]) \(word[i+1])"
                if let rank = mergeRank[pair], rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if bestIdx == -1 { break }
            let merged = word[bestIdx] + word[bestIdx + 1]
            word.remove(at: bestIdx + 1)
            word[bestIdx] = merged
        }

        var ids = [Int]()
        for token in word {
            if let id = tokenToId[token] {
                ids.append(id)
            } else {
                for ch in token {
                    if let id = tokenToId[String(ch)] { ids.append(id) }
                }
            }
        }
        return ids
    }

    func decode(_ ids: [Int]) -> String {
        var bytes = [UInt8]()
        for id in ids {
            guard id >= 0, id < vocab.count else { continue }
            if id >= 151643 { continue }
            let token = vocab[id]
            for ch in token {
                if let b = BPETokenizer.unicodeToByte[ch] { bytes.append(b) }
            }
        }
        return String(bytes: bytes, encoding: .utf8)
            ?? String(bytes.map { Character(Unicode.Scalar($0)) })
    }

    func encodeChatML(system: String = "You are a helpful assistant.",
                      user: String) -> [Int] {
        let prompt = "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
        return encode(prompt)
    }
}

// MARK: - Float16 Binary Files

class FP16BinaryFile {
    let data: Data
    let count: Int

    init(path: String, expectedCount: Int? = nil) {
        self.data = try! Data(contentsOf: URL(fileURLWithPath: path))
        self.count = data.count / 2
        if let expected = expectedCount {
            precondition(count == expected,
                         "FP16 file \(path): got \(count) elements, expected \(expected)")
        }
    }

    /// Look up a row (e.g., token embedding). Returns Float32 array.
    func row(_ index: Int, dim: Int) -> [Float] {
        precondition(index * dim + dim <= count, "row index out of bounds")
        var result = [Float](repeating: 0, count: dim)
        data.withUnsafeBytes { ptr in
            let fp16 = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
            let start = index * dim
            for i in 0..<dim {
                result[i] = Float(fp16[start + i])
            }
        }
        return result
    }

    /// Write a row as Float16 into a pre-allocated buffer.
    func writeRow(_ index: Int, dim: Int, into ptr: UnsafeMutablePointer<Float16>) {
        data.withUnsafeBytes { rawPtr in
            let fp16 = rawPtr.baseAddress!.assumingMemoryBound(to: Float16.self)
            memcpy(ptr, fp16 + index * dim, dim * MemoryLayout<Float16>.size)
        }
    }

    /// Full matmul: logits = hidden @ embed^T (for LM head with tied embeddings).
    /// hidden: [Float] of length d_model
    /// Returns: [Float] of length vocab_size
    func matmulTranspose(hidden: [Float], rows: Int, cols: Int) -> [Float] {
        precondition(hidden.count == cols, "hidden dim mismatch")
        precondition(rows * cols <= count, "weight matrix too small")
        var result = [Float](repeating: 0, count: rows)
        data.withUnsafeBytes { ptr in
            let fp16 = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
            // result[r] = sum_c hidden[c] * weight[r * cols + c]
            for r in 0..<rows {
                var dot: Float = 0
                let base = r * cols
                for c in 0..<cols {
                    dot += hidden[c] * Float(fp16[base + c])
                }
                result[r] = dot
            }
        }
        return result
    }
}

// MARK: - Host-side Final Norm + LM Head

/// RMS norm on host: out[i] = (x[i] / rms) * gamma[i]
func rmsNorm(_ x: [Float], gamma: FP16BinaryFile, eps: Float) -> [Float] {
    let d = x.count
    var meanSq: Float = 0
    for v in x { meanSq += v * v }
    meanSq /= Float(d)
    let rms = sqrtf(meanSq + eps)
    let invRms = 1.0 / rms

    var result = [Float](repeating: 0, count: d)
    gamma.data.withUnsafeBytes { ptr in
        let gPtr = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
        for i in 0..<d {
            result[i] = x[i] * invRms * Float(gPtr[i])
        }
    }
    return result
}

/// Extract hidden state from shard output MLMultiArray → [Float]
func extractHidden(_ arr: MLMultiArray, dim: Int) -> [Float] {
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
    let shape = arr.shape.map { Int(truncating: $0) }
    let strides = arr.strides.map { Int(truncating: $0) }
    var result = [Float](repeating: 0, count: dim)
    for i in 0..<dim {
        // Handle arbitrary strides (CoreML may add padding)
        var linearIdx = i
        var offset = 0
        for d in stride(from: shape.count - 1, through: 0, by: -1) {
            let idx = linearIdx % shape[d]
            offset += idx * strides[d]
            linearIdx /= shape[d]
        }
        result[i] = Float(ptr[offset])
    }
    return result
}

/// Write [Float] into MLMultiArray as Float16
func writeHiddenToMLArray(_ hidden: [Float], arr: MLMultiArray) {
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
    for i in 0..<hidden.count {
        ptr[i] = Float16(hidden[i])
    }
}

// MARK: - Sampling

var gTemperature: Float = 0.0
var gTopP: Float = 1.0
var gRepetitionPenalty: Float = 1.0
var gRecentTokens: [Int] = []

func sampleFromLogits(_ logits: [Float]) -> Int {
    if gTemperature <= 0 && gRepetitionPenalty == 1.0 {
        // Greedy argmax
        var maxVal: Float = -.infinity
        var maxIdx = 0
        for (i, v) in logits.enumerated() {
            if v > maxVal { maxVal = v; maxIdx = i }
        }
        return maxIdx
    }

    var scaled = logits
    let invT = gTemperature > 0 ? 1.0 / gTemperature : 1.0
    for i in 0..<scaled.count { scaled[i] *= invT }

    if gRepetitionPenalty != 1.0 {
        for tid in gRecentTokens where tid >= 0 && tid < scaled.count {
            scaled[tid] = scaled[tid] > 0
                ? scaled[tid] / gRepetitionPenalty
                : scaled[tid] * gRepetitionPenalty
        }
    }

    // Softmax
    var maxL: Float = -.infinity
    for v in scaled { if v > maxL { maxL = v } }
    var probs = [Float](repeating: 0, count: scaled.count)
    var sum: Float = 0
    for i in 0..<scaled.count {
        let e = expf(scaled[i] - maxL)
        probs[i] = e
        sum += e
    }
    for i in 0..<scaled.count { probs[i] /= sum }

    // Top-p
    if gTopP < 1.0 && gTopP > 0 {
        var idx = Array(0..<scaled.count)
        idx.sort { probs[$0] > probs[$1] }
        var cum: Float = 0
        var keep = Set<Int>()
        for j in idx {
            cum += probs[j]
            keep.insert(j)
            if cum >= gTopP { break }
        }
        var newSum: Float = 0
        for i in 0..<scaled.count {
            if !keep.contains(i) { probs[i] = 0 } else { newSum += probs[i] }
        }
        if newSum > 0 { for i in 0..<scaled.count { probs[i] /= newSum } }
    }

    // Sample
    let r = Float.random(in: 0..<1)
    var acc: Float = 0
    for i in 0..<scaled.count {
        acc += probs[i]
        if r < acc { return i }
    }
    return scaled.count - 1
}

// MARK: - Path Resolution

func resolvePath(_ relative: String, relativeTo basePath: String) -> String {
    if relative.hasPrefix("/") { return relative }
    let baseDir = (basePath as NSString).deletingLastPathComponent
    return (baseDir as NSString).appendingPathComponent(relative)
}

// MARK: - Main

func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    var metaPath = ""
    var prompt = "The meaning of life is"
    var maxTokens = 64
    var useChatML = true

    var i = 0
    while i < args.count {
        switch args[i] {
        case "--meta":
            metaPath = args[i + 1]; i += 2
        case "--prompt":
            prompt = args[i + 1]; i += 2
        case "--max-tokens":
            maxTokens = Int(args[i + 1])!; i += 2
        case "--raw":
            useChatML = false; i += 1
        case "--temperature":
            gTemperature = Float(args[i + 1])!; i += 2
        case "--top-p":
            gTopP = Float(args[i + 1])!; i += 2
        case "--repetition-penalty":
            gRepetitionPenalty = Float(args[i + 1])!; i += 2
        default:
            i += 1
        }
    }

    guard !metaPath.isEmpty else {
        print("Usage: qwen_factory --meta path/to/meta.json [--prompt \"text\"]")
        return
    }

    // ── Load meta ──
    print("Loading meta: \(metaPath)")
    let metaData = try Data(contentsOf: URL(fileURLWithPath: metaPath))
    let meta = try JSONDecoder().decode(QwenFactoryMeta.self, from: metaData)
    let d = meta.dModel
    let dHalf = meta.dHead / 2
    let maxSeq = meta.maxSeqLen

    print("  \(meta.modelFamily) \(meta.nLayers)L, d=\(d), nh=\(meta.nHeads), nkv=\(meta.nKvHeads)")
    print("  \(meta.shards.count) shard(s), seq_len=\(maxSeq), INT\(meta.quantBits)")

    // ── Validate shard contiguity ──
    let sortedShards = meta.shards.sorted { $0.start < $1.start }
    precondition(!sortedShards.isEmpty, "no shards configured")
    precondition(sortedShards.first!.start == 0, "first shard must start at 0")
    precondition(sortedShards.last!.end == meta.nLayers,
                 "last shard must end at n_layers (\(meta.nLayers)), got \(sortedShards.last!.end)")
    for idx in 1..<sortedShards.count {
        precondition(sortedShards[idx - 1].end == sortedShards[idx].start,
                     "shards must be contiguous: shard \(idx-1) ends at \(sortedShards[idx-1].end) "
                     + "but shard \(idx) starts at \(sortedShards[idx].start)")
    }

    // ── Load shared artifacts ──
    print("Loading embeddings...")
    let embedPath = resolvePath(meta.embedBin, relativeTo: metaPath)
    let embed = FP16BinaryFile(path: embedPath, expectedCount: meta.vocabSize * d)
    print("  \(meta.vocabSize) × \(d) (\(embed.data.count / 1_000_000) MB)")

    print("Loading final norm...")
    let normPath = resolvePath(meta.finalNormBin, relativeTo: metaPath)
    let finalNorm = FP16BinaryFile(path: normPath, expectedCount: d)

    print("Loading LM head weights...")
    let lmHeadPath = resolvePath(meta.lmHeadBin, relativeTo: metaPath)
    let lmHead: FP16BinaryFile
    if meta.tieWordEmbeddings {
        lmHead = embed  // Same weights
        print("  Tied to embeddings ✓")
    } else {
        lmHead = FP16BinaryFile(path: lmHeadPath, expectedCount: meta.vocabSize * d)
        print("  Separate LM head (\(lmHead.data.count / 1_000_000) MB)")
    }

    print("Loading tokenizer...")
    let tokPath = resolvePath(meta.tokenizerJson, relativeTo: metaPath)
    let tokenizer = BPETokenizer(jsonPath: tokPath)
    print("  \(tokenizer.vocab.count) tokens, \(tokenizer.merges.count) merges")

    // ── Load shard models ──
    guard #available(macOS 15.0, *) else {
        print("ERROR: Stateful models require macOS 15.0+")
        return
    }

    let mlConfig = MLModelConfiguration()
    // .all required for stateful models — .cpuAndNeuralEngine produces NaN
    mlConfig.computeUnits = .all

    var shardModels = [MLModel]()
    var shardStates = [MLState]()

    print("\nLoading \(sortedShards.count) shard model(s)...")
    for shard in sortedShards {
        let shardPath = resolvePath(shard.path, relativeTo: metaPath)
        guard FileManager.default.fileExists(atPath: shardPath) else {
            print("ERROR: shard not found: \(shardPath)")
            print("  Run: xcrun coremlcompiler compile <pkg>.mlpackage <outputdir>")
            return
        }
        print("  [\(shard.start),\(shard.end)) \(shard.path)")
        let model = try MLModel(contentsOf: URL(fileURLWithPath: shardPath),
                                configuration: mlConfig)
        shardModels.append(model)
        shardStates.append(model.makeState())
    }
    print("  All shards loaded ✓")

    // ── Pre-allocate input buffers ──
    let xArr = try MLMultiArray(
        shape: [1, NSNumber(value: d), 1, 1], dataType: .float16)
    let xPtr = xArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let cosArr = try MLMultiArray(shape: [1, NSNumber(value: dHalf)], dataType: .float16)
    let cosPtr = cosArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let sinArr = try MLMultiArray(shape: [1, NSNumber(value: dHalf)], dataType: .float16)
    let sinPtr = sinArr.dataPointer.assumingMemoryBound(to: Float16.self)

    let attnMaskArr = try MLMultiArray(
        shape: [1, 1, 1, NSNumber(value: maxSeq)], dataType: .float16)
    let attnMaskPtr = attnMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
    for j in 0..<maxSeq { attnMaskPtr[j] = Float16(-10000.0) }

    let kvWriteMaskArr = try MLMultiArray(
        shape: [1, 1, NSNumber(value: maxSeq), 1], dataType: .float16)
    let kvWriteMaskPtr = kvWriteMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
    for j in 0..<maxSeq { kvWriteMaskPtr[j] = 0 }

    var cacheSeqLen = 0

    // Timing accumulators
    var tPack: Double = 0
    var tPredict: Double = 0
    var tHead: Double = 0
    var nForward: Int = 0

    // ── Forward one token ──
    func forwardOne(tokenId: Int, pos: Int) throws -> Int {
        let t0 = CFAbsoluteTimeGetCurrent()
        let seq = cacheSeqLen

        // Embedding lookup → x buffer
        embed.writeRow(tokenId, dim: d, into: xPtr)

        // RoPE from meta
        let ropePos = min(pos, meta.ropeCos.count - 1)
        for j in 0..<dHalf {
            cosPtr[j] = Float16(meta.ropeCos[ropePos][j])
            sinPtr[j] = Float16(meta.ropeSin[ropePos][j])
        }

        // Attention mask: unmask position seq
        attnMaskPtr[seq] = 0
        // KV write mask: clear previous, set current
        if seq > 0 { kvWriteMaskPtr[seq - 1] = 0 }
        kvWriteMaskPtr[seq] = Float16(1.0)

        let t1 = CFAbsoluteTimeGetCurrent()

        // ── Chain through shards ──
        var hiddenValue = MLFeatureValue(multiArray: xArr)

        for (shardIdx, pair) in zip(shardModels, shardStates).enumerated() {
            let (model, state) = pair
            let features: [String: MLFeatureValue] = [
                "x": hiddenValue,
                "rope_cos": MLFeatureValue(multiArray: cosArr),
                "rope_sin": MLFeatureValue(multiArray: sinArr),
                "attn_mask": MLFeatureValue(multiArray: attnMaskArr),
                "kv_write_mask": MLFeatureValue(multiArray: kvWriteMaskArr),
            ]
            let provider = try MLDictionaryFeatureProvider(dictionary: features)
            let result = try model.prediction(from: provider, using: state)

            // Get shard output: "hidden" for shards, "logits" for monolithic
            let outputName = result.featureNames.contains("hidden") ? "hidden" : "logits"
            let output = result.featureValue(for: outputName)!.multiArrayValue!

            if shardIdx + 1 < shardModels.count {
                // Copy hidden into x buffer for next shard
                let srcPtr = output.dataPointer.assumingMemoryBound(to: Float16.self)
                memcpy(xPtr, srcPtr, d * MemoryLayout<Float16>.size)
                hiddenValue = MLFeatureValue(multiArray: xArr)
            } else {
                hiddenValue = MLFeatureValue(multiArray: output)
            }
        }

        let t2 = CFAbsoluteTimeGetCurrent()
        cacheSeqLen += 1

        // ── Host-side final norm + LM head ──
        let hiddenArr = hiddenValue.multiArrayValue!
        let outputName = hiddenArr.shape.last!.intValue

        // If monolithic model returned logits directly, just argmax
        if outputName == meta.vocabSize {
            let logitsPtr = hiddenArr.dataPointer.assumingMemoryBound(to: Float16.self)
            var logits = [Float](repeating: 0, count: meta.vocabSize)
            for i in 0..<meta.vocabSize { logits[i] = Float(logitsPtr[i]) }
            let t3 = CFAbsoluteTimeGetCurrent()
            tPack += t1 - t0; tPredict += t2 - t1; tHead += t3 - t2
            nForward += 1
            return sampleFromLogits(logits)
        }

        // Shard mode: hidden → RMS norm → LM head projection
        let hidden = extractHidden(hiddenArr, dim: d)
        let normed = rmsNorm(hidden, gamma: finalNorm, eps: Float(meta.rmsNormEps))
        let logits = lmHead.matmulTranspose(hidden: normed, rows: meta.vocabSize, cols: d)

        let t3 = CFAbsoluteTimeGetCurrent()
        tPack += t1 - t0
        tPredict += t2 - t1
        tHead += t3 - t2
        nForward += 1
        return sampleFromLogits(logits)
    }

    // ── Tokenize ──
    let tokens: [Int]
    if useChatML {
        tokens = tokenizer.encodeChatML(user: prompt)
        print("\nChatML prompt: \(tokens.count) tokens")
    } else {
        tokens = tokenizer.encode(prompt)
        print("\nRaw prompt: \(tokens.count) tokens")
    }

    precondition(tokens.count + maxTokens <= maxSeq,
                 "prompt (\(tokens.count)) + max_tokens (\(maxTokens)) exceeds seq_len (\(maxSeq))")

    // ── Generation loop ──
    print("Generating (max \(maxTokens) tokens, \(sortedShards.count) shard(s))...")
    print(String(repeating: "─", count: 56))
    fflush(stdout)

    gRecentTokens = Array(tokens.suffix(256))
    var generated = [Int]()
    let startTime = CFAbsoluteTimeGetCurrent()

    // Prefill
    var lastToken = 0
    for (pos, tok) in tokens.enumerated() {
        lastToken = try forwardOne(tokenId: tok, pos: pos)
    }
    generated.append(lastToken)
    gRecentTokens.append(lastToken)
    if gRecentTokens.count > 256 { gRecentTokens.removeFirst(gRecentTokens.count - 256) }
    let firstText = tokenizer.decode([lastToken])
    print(firstText, terminator: "")
    fflush(stdout)

    let prefillTime = CFAbsoluteTimeGetCurrent() - startTime

    // Decode
    let decodeStart = CFAbsoluteTimeGetCurrent()
    for step in 0..<(maxTokens - 1) {
        let pos = tokens.count + step
        if pos >= maxSeq - 1 { break }

        let next = try forwardOne(tokenId: generated.last!, pos: pos)
        generated.append(next)
        gRecentTokens.append(next)
        if gRecentTokens.count > 256 { gRecentTokens.removeFirst(gRecentTokens.count - 256) }

        let text = tokenizer.decode([next])
        print(text, terminator: "")
        fflush(stdout)

        if next == meta.eosTokenId || next == 151645 || next == 151643 { break }
    }
    let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
    let totalTime = CFAbsoluteTimeGetCurrent() - startTime

    print("\n")
    let fullText = tokenizer.decode(generated)
    let decodeTokPerSec = decodeTime > 0 ? Double(generated.count - 1) / decodeTime : 0

    print("""
    ╔════════════════════════════════════════════════════════╗
    ║  QwenFactory · Multi-Shard ANE Inference               ║
    ╚════════════════════════════════════════════════════════╝

      Model:       \(meta.modelFamily) (\(meta.nLayers)L, d=\(d))
      Shards:      \(sortedShards.count) (layers: \(sortedShards.map { "[\($0.start),\($0.end))" }.joined(separator: " ")))
      Compute:     ANE (stateful KV, \(maxSeq) positions)
      Quant:       INT\(meta.quantBits)
      LM Head:     Host-side (\(meta.tieWordEmbeddings ? "tied to embed" : "separate") \(meta.vocabSize)×\(d))

      Prefill:     \(tokens.count) tokens → \(String(format: "%.1f", prefillTime * 1000)) ms (\(String(format: "%.0f", Double(tokens.count) / prefillTime)) tok/s)
      Decode:      \(generated.count) tokens → \(String(format: "%.1f", decodeTime * 1000)) ms (\(String(format: "%.1f", decodeTokPerSec)) tok/s)
      Total:       \(String(format: "%.2f", totalTime)) s

      ── Hot Path Breakdown (\(nForward) forward calls) ──
      Pack inputs:    \(String(format: "%.1f", tPack * 1000)) ms (\(String(format: "%.1f", tPack / (tPack + tPredict + tHead) * 100))%)
      ANE predict:    \(String(format: "%.1f", tPredict * 1000)) ms (\(String(format: "%.1f", tPredict / (tPack + tPredict + tHead) * 100))%)
      Host LM head:   \(String(format: "%.1f", tHead * 1000)) ms (\(String(format: "%.1f", tHead / (tPack + tPredict + tHead) * 100))%)
      Avg per token:  pack=\(String(format: "%.2f", tPack / Double(nForward) * 1000))ms predict=\(String(format: "%.2f", tPredict / Double(nForward) * 1000))ms head=\(String(format: "%.2f", tHead / Double(nForward) * 1000))ms

      Output:      \(fullText.prefix(80))...
    """)
}

try main()
