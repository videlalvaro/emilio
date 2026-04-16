// QwenANE: Conv-only Qwen2.5-0.5B inference on Apple Neural Engine.
//
// All matmuls are Conv2d(1×1). KV cache is CoreML StateType (on-device).
// Float16 throughout. Proper BPE tokenizer from GGUF vocab+merges.
//
// Build:
//   swiftc -O -framework CoreML -framework Accelerate \
//          -o qwen_ane qwen_ane.swift
//
// Usage:
//   ./qwen_ane [--layers N] [--prompt "text"] [--max-tokens N]

import CoreML
import Foundation

// MARK: - Config

struct QwenConfig {
    let dModel: Int
    let nHeads: Int
    let nKvHeads: Int
    let dHead: Int
    let dFf: Int
    let vocabSize: Int
    let nLayers: Int
    let maxSeqLen: Int
    let rmsNormEps: Float
    let ropeFreqBase: Float
    let eosTokenId: Int
    let bosTokenId: Int

    var ropeCos: [[Float]] = []
    var ropeSin: [[Float]] = []

    static func fromJSON(_ path: String) -> QwenConfig {
        let data = try! Data(contentsOf: URL(fileURLWithPath: path))
        let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]

        let cosRaw = json["rope_cos"] as! [[Double]]
        let sinRaw = json["rope_sin"] as! [[Double]]

        var cfg = QwenConfig(
            dModel: json["d_model"] as! Int,
            nHeads: json["n_heads"] as! Int,
            nKvHeads: json["n_kv_heads"] as! Int,
            dHead: json["d_head"] as! Int,
            dFf: json["d_ff"] as! Int,
            vocabSize: json["vocab_size"] as! Int,
            nLayers: json["n_layers_exported"] as! Int,
            maxSeqLen: json["max_seq_len"] as! Int,
            rmsNormEps: Float(json["rms_norm_eps"] as! Double),
            ropeFreqBase: Float(json["rope_freq_base"] as! Double),
            eosTokenId: (json["eos_token_id"] as? Int) ?? 151645,
            bosTokenId: (json["bos_token_id"] as? Int) ?? 151643
        )
        cfg.ropeCos = cosRaw.map { $0.map { Float($0) } }
        cfg.ropeSin = sinRaw.map { $0.map { Float($0) } }
        return cfg
    }
}

// MARK: - BPE Tokenizer

class BPETokenizer {
    let vocab: [String]          // token_id → token string
    let tokenToId: [String: Int] // token string → token_id
    let merges: [(String, String)]
    let mergeRank: [String: Int] // "a b" → rank
    let eosId: Int
    let bosId: Int

    // GPT-2 byte-to-unicode mapping
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
        for (k, v) in byteToUnicode {
            map[v] = k
        }
        return map
    }()

    init(jsonPath: String) {
        let data = try! Data(contentsOf: URL(fileURLWithPath: jsonPath))
        let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]

        let tokens = json["tokens"] as! [String]
        self.vocab = tokens
        var t2i = [String: Int]()
        for (i, tok) in tokens.enumerated() {
            t2i[tok] = i
        }
        self.tokenToId = t2i

        let mergeStrings = json["merges"] as! [String]
        var mergeList = [(String, String)]()
        var rankMap = [String: Int]()
        for (i, m) in mergeStrings.enumerated() {
            // Each merge is "left right"
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
    }

    /// Encode text → token IDs using GPT-2 BPE
    func encode(_ text: String) -> [Int] {
        // Step 1: Convert UTF-8 bytes to GPT-2 unicode characters
        let utf8Bytes = Array(text.utf8)
        let gptChars = utf8Bytes.map { BPETokenizer.byteToUnicode[$0]! }
        var word = gptChars.map { String($0) }

        // Step 2: BPE merge loop
        while word.count >= 2 {
            // Find the pair with the lowest merge rank
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

            // Merge the pair
            let merged = word[bestIdx] + word[bestIdx + 1]
            word.remove(at: bestIdx + 1)
            word[bestIdx] = merged
        }

        // Step 3: Look up token IDs
        var ids = [Int]()
        for token in word {
            if let id = tokenToId[token] {
                ids.append(id)
            } else {
                // Fallback: encode each character separately
                for ch in token {
                    if let id = tokenToId[String(ch)] {
                        ids.append(id)
                    }
                }
            }
        }
        return ids
    }

    /// Decode token IDs → text
    func decode(_ ids: [Int]) -> String {
        var bytes = [UInt8]()
        for id in ids {
            guard id >= 0, id < vocab.count else { continue }
            let token = vocab[id]
            for ch in token {
                if let b = BPETokenizer.unicodeToByte[ch] {
                    bytes.append(b)
                }
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? String(bytes.map { Character(Unicode.Scalar($0)) })
    }

    /// Format as ChatML
    func encodeChatML(system: String = "You are a helpful assistant.",
                      user: String) -> [Int] {
        let prompt = "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
        return encode(prompt)
    }
}

// MARK: - Float16 Embeddings

class TokenEmbeddings {
    let data: UnsafeBufferPointer<UInt16>  // fp16 raw storage
    let rawData: Data
    let vocab: Int
    let dim: Int

    init(path: String, vocab: Int, dim: Int) {
        let url = URL(fileURLWithPath: path)
        self.rawData = try! Data(contentsOf: url)
        self.vocab = vocab
        self.dim = dim
        self.data = rawData.withUnsafeBytes { ptr in
            UnsafeBufferPointer(start: ptr.baseAddress!.assumingMemoryBound(to: UInt16.self),
                                count: ptr.count / 2)
        }
        precondition(data.count == vocab * dim,
                     "Embedding mismatch: \(data.count) vs \(vocab * dim)")
    }

    /// Look up token embedding, return as Float32 array
    func lookup(_ tokenId: Int) -> [Float] {
        let start = tokenId * dim
        var result = [Float](repeating: 0, count: dim)
        for i in 0..<dim {
            result[i] = float16ToFloat32(data[start + i])
        }
        return result
    }

    private func float16ToFloat32(_ h: UInt16) -> Float {
        let f16 = Float16(bitPattern: h)
        return Float(f16)
    }
}

// MARK: - Argmax

// Dragon Book: strength reduction — raw Float16 pointer instead of NSNumber subscript
func argmax(_ arr: MLMultiArray) -> Int {
    let count = arr.count
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
    var maxVal: Float16 = -.infinity
    var maxIdx = 0
    for i in 0..<count {
        if ptr[i] > maxVal {
            maxVal = ptr[i]
            maxIdx = i
        }
    }
    return maxIdx
}

// MARK: - Main

func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    var nLayers = 24
    var prompt = "The meaning of life is"
    var maxTokens = 64
    var useChatML = true
    var useFixed = false
    var useStateful = false

    var i = 0
    while i < args.count {
        switch args[i] {
        case "--layers":
            nLayers = Int(args[i + 1])!; i += 2
        case "--prompt":
            prompt = args[i + 1]; i += 2
        case "--max-tokens":
            maxTokens = Int(args[i + 1])!; i += 2
        case "--raw":
            useChatML = false; i += 1
        case "--fixed":
            useFixed = true; i += 1
        case "--stateful":
            useStateful = true; i += 1
        default:
            i += 1
        }
    }

    let prefix: String
    if useStateful {
        prefix = "QwenANE_\(nLayers)L_stateful"
    } else if useFixed {
        prefix = "QwenANE_\(nLayers)L_fixed"
    } else {
        prefix = "QwenANE_\(nLayers)L"
    }

    // ── Load config ──
    print("Loading \(prefix)...")
    let cfg = QwenConfig.fromJSON("\(prefix)_meta.json")
    print("  \(cfg.nLayers)L, d=\(cfg.dModel), nh=\(cfg.nHeads), nkv=\(cfg.nKvHeads)")

    // ── Load tokenizer ──
    print("Loading BPE tokenizer...")
    let tokenizer = BPETokenizer(jsonPath: "\(prefix)_tokenizer.json")
    print("  \(tokenizer.vocab.count) tokens, \(tokenizer.merges.count) merges")

    // ── Load embeddings (fp16) ──
    print("Loading embeddings (fp16)...")
    let embd = TokenEmbeddings(
        path: "\(prefix)_embd.bin", vocab: cfg.vocabSize, dim: cfg.dModel)

    // ── Load CoreML model ──
    print("Loading CoreML model...")
    let modelURL = URL(fileURLWithPath: "\(prefix).mlmodelc")
    guard FileManager.default.fileExists(atPath: "\(prefix).mlmodelc") else {
        print("ERROR: \(prefix).mlmodelc not found. Run:")
        print("  xcrun coremlcompiler compile \(prefix).mlpackage .")
        return
    }

    let mlConfig = MLModelConfiguration()
    mlConfig.computeUnits = .cpuAndNeuralEngine
    let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)

    // ═══════════════════════════════════════════════════════════════════
    // KV Cache — book-driven optimizations:
    //
    // 1. Iverson (A Programming Language): Store data in the "right shape"
    //    from the start. The representation should match the consumption
    //    pattern so the reshape is a no-op.
    //    → Store KV already in (nkv, seq, dh) layout. No transpose needed.
    //
    // 2. Stepanov (Elements of Programming): Pre-allocate to max_seq_len
    //    with a write cursor. The buffer is a regular type with a
    //    well-defined orbit (append advances the cursor, never reallocates).
    //    → Fixed-size Float16 arrays, cursor tracks fill level.
    //
    // 3. Dragon Book (Compilers: Aho et al.): Strength reduction — replace
    //    expensive operations with cheaper equivalents.
    //    → Raw Float16 pointer memcpy instead of NSNumber subscript access.
    //    NSNumber dispatch costs ~200ns/element (ObjC message send +
    //    boxing). Raw pointer write costs ~1ns/element. 200× speedup
    //    on the copy path.
    //
    // 4. Brodie (Thinking Forth): Factor so operation boundaries align
    //    with data layout. The "word" should express the intent directly.
    //    → Keep MLMultiArrays alive between steps. Don't reallocate 48
    //    arrays per token. Just update the strides/shape in place.
    // ═══════════════════════════════════════════════════════════════════

    let maxSeq = cfg.maxSeqLen

    // Pre-allocated KV cache as raw pointers — Stepanov: allocate once
    // Layout: (nKvHeads, maxSeq, dHead) per layer — Iverson: match consumption shape
    // Dragon Book: raw pointer eliminates Array bounds-checking + CoW overhead
    let kvElems = cfg.nKvHeads * maxSeq * cfg.dHead
    let kCachePtr = (0..<cfg.nLayers).map { _ -> UnsafeMutablePointer<Float16> in
        let p = UnsafeMutablePointer<Float16>.allocate(capacity: kvElems)
        p.initialize(repeating: 0, count: kvElems)
        return p
    }
    let vCachePtr = (0..<cfg.nLayers).map { _ -> UnsafeMutablePointer<Float16> in
        let p = UnsafeMutablePointer<Float16>.allocate(capacity: kvElems)
        p.initialize(repeating: 0, count: kvElems)
        return p
    }
    var cacheSeqLen = 0

    print("  Model loaded ✓")
    fflush(stdout)

    // ── Tokenize ──
    let tokens: [Int]
    if useChatML {
        tokens = tokenizer.encodeChatML(user: prompt)
        print("\nChatML prompt: \(tokens.count) tokens")
    } else {
        tokens = tokenizer.encode(prompt)
        print("\nRaw prompt: \(tokens.count) tokens")
    }

    let dHalf = cfg.dHead / 2

    // Timing accumulators for hot-path breakdown
    var tPack: Double = 0    // Host: build MLMultiArrays + copy KV
    var tPredict: Double = 0 // ANE: model.prediction()
    var tExtract: Double = 0 // Host: extract outputs + append KV
    var nForward: Int = 0

    // ═══════════════════════════════════════════════════════════════════
    // Pre-allocated input buffers — Stepanov: regular types with
    // fixed orbit. Allocate once, overwrite each forward call.
    // Eliminates 51 MLMultiArray mallocs + 48 memcpys per token.
    // ═══════════════════════════════════════════════════════════════════
    let xArr = try MLMultiArray(
        shape: [1, NSNumber(value: cfg.dModel), 1, 1], dataType: .float16)
    let xPtr = xArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let cosArr = try MLMultiArray(shape: [1, NSNumber(value: dHalf)], dataType: .float16)
    let cosPtr = cosArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let sinArr = try MLMultiArray(shape: [1, NSNumber(value: dHalf)], dataType: .float16)
    let sinPtr = sinArr.dataPointer.assumingMemoryBound(to: Float16.self)

    // Pre-build key strings (avoid String interpolation in hot path)
    let kCacheKeys = (0..<cfg.nLayers).map { "k_cache_\($0)" }
    let vCacheKeys = (0..<cfg.nLayers).map { "v_cache_\($0)" }
    let newKKeys = (0..<cfg.nLayers).map { "new_k_\($0)" }
    let newVKeys = (0..<cfg.nLayers).map { "new_v_\($0)" }

    // ═══════════════════════════════════════════════════════════════════
    // Fixed-shape mode: pre-allocated masks + full-size KV views
    // Eliminates ALL dynamic shapes → ANE never replans execution graph.
    // attn_mask:     (1, 1, 1, maxSeq) — 0 valid, -1e4 masked
    // kv_write_mask: (1, 1, maxSeq, 1) — 1.0 at write position, 0 elsewhere
    // ═══════════════════════════════════════════════════════════════════
    let attnMaskArr: MLMultiArray?
    let attnMaskPtr: UnsafeMutablePointer<Float16>?
    let kvWriteMaskArr: MLMultiArray?
    let kvWriteMaskPtr: UnsafeMutablePointer<Float16>?
    // Fixed-mode KV views: always (1, nkv, maxSeq, dh) — truly zero-copy
    let fixedKViews: [MLMultiArray]?
    let fixedVViews: [MLMultiArray]?

    if useFixed || useStateful {
        let am = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: maxSeq)], dataType: .float16)
        let amp = am.dataPointer.assumingMemoryBound(to: Float16.self)
        // Initialize all masked
        for j in 0..<maxSeq { amp[j] = Float16(-10000.0) }
        attnMaskArr = am
        attnMaskPtr = amp

        let wm = try MLMultiArray(
            shape: [1, 1, NSNumber(value: maxSeq), 1], dataType: .float16)
        let wmp = wm.dataPointer.assumingMemoryBound(to: Float16.self)
        for j in 0..<maxSeq { wmp[j] = 0 }
        kvWriteMaskArr = wm
        kvWriteMaskPtr = wmp

        if useFixed {
            // Pre-build KV cache views — these never change shape
            var kViews = [MLMultiArray]()
            var vViews = [MLMultiArray]()
            for layer in 0..<cfg.nLayers {
                let kv = try MLMultiArray(
                    dataPointer: UnsafeMutableRawPointer(kCachePtr[layer]),
                    shape: [1, NSNumber(value: cfg.nKvHeads), NSNumber(value: maxSeq), NSNumber(value: cfg.dHead)],
                    dataType: .float16,
                    strides: [
                        NSNumber(value: cfg.nKvHeads * maxSeq * cfg.dHead),
                        NSNumber(value: maxSeq * cfg.dHead),
                        NSNumber(value: cfg.dHead),
                        NSNumber(value: 1)
                    ],
                    deallocator: nil
                )
                let vv = try MLMultiArray(
                    dataPointer: UnsafeMutableRawPointer(vCachePtr[layer]),
                    shape: [1, NSNumber(value: cfg.nKvHeads), NSNumber(value: maxSeq), NSNumber(value: cfg.dHead)],
                    dataType: .float16,
                    strides: [
                        NSNumber(value: cfg.nKvHeads * maxSeq * cfg.dHead),
                        NSNumber(value: maxSeq * cfg.dHead),
                        NSNumber(value: cfg.dHead),
                        NSNumber(value: 1)
                    ],
                    deallocator: nil
                )
                kViews.append(kv)
                vViews.append(vv)
            }
            fixedKViews = kViews
            fixedVViews = vViews
            print("  Fixed-shape mode: masks + full KV views pre-allocated ✓")
        } else {
            fixedKViews = nil
            fixedVViews = nil
        }
    } else {
        attnMaskArr = nil
        attnMaskPtr = nil
        kvWriteMaskArr = nil
        kvWriteMaskPtr = nil
        fixedKViews = nil
        fixedVViews = nil
    }

    // ═══════════════════════════════════════════════════════════════════
    // Stateful mode: MLState keeps KV cache on-device (ANE).
    // Zero host↔ANE KV data transfer. Only x, rope, masks cross the bus.
    // ═══════════════════════════════════════════════════════════════════
    var mlState: MLState? = nil
    if useStateful {
        if #available(macOS 15.0, *) {
            mlState = model.makeState()
            print("  Stateful mode: MLState created (KV on-device) ✓")
        } else {
            print("ERROR: Stateful mode requires macOS 15.0+")
            return
        }
    }

    // ── Forward one token through ANE (stateful — zero KV copy) ──
    func forwardOneStateful(tokenId: Int, pos: Int) throws -> Int {
        let t0 = CFAbsoluteTimeGetCurrent()
        let embedding = embd.lookup(tokenId)
        let seq = cacheSeqLen

        // x: overwrite pre-allocated buffer
        for j in 0..<cfg.dModel { xPtr[j] = Float16(embedding[j]) }

        // RoPE
        let ropePos = min(pos, cfg.ropeCos.count - 1)
        for j in 0..<dHalf {
            cosPtr[j] = Float16(cfg.ropeCos[ropePos][j])
            sinPtr[j] = Float16(cfg.ropeSin[ropePos][j])
        }

        // attn_mask: unmask position seq
        attnMaskPtr![seq] = 0
        // kv_write_mask: clear previous, set current
        if seq > 0 { kvWriteMaskPtr![seq - 1] = 0 }
        kvWriteMaskPtr![seq] = Float16(1.0)

        // Only 5 inputs — NO KV cache arrays at all!
        let inputDict: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: xArr),
            "rope_cos": MLFeatureValue(multiArray: cosArr),
            "rope_sin": MLFeatureValue(multiArray: sinArr),
            "attn_mask": MLFeatureValue(multiArray: attnMaskArr!),
            "kv_write_mask": MLFeatureValue(multiArray: kvWriteMaskArr!),
        ]

        let t1 = CFAbsoluteTimeGetCurrent()
        let provider = try MLDictionaryFeatureProvider(dictionary: inputDict)
        if #available(macOS 15.0, *) {
            let result = try model.prediction(from: provider, using: mlState!)
            let t2 = CFAbsoluteTimeGetCurrent()

            // No KV extraction needed — state is updated internally on-device
            cacheSeqLen += 1

            let logits = result.featureValue(for: "logits")!.multiArrayValue!
            let t3 = CFAbsoluteTimeGetCurrent()
            tPack += t1 - t0
            tPredict += t2 - t1
            tExtract += t3 - t2
            nForward += 1
            return argmax(logits)
        }
        fatalError("Stateful requires macOS 15+")
    }

    // ── Forward one token through ANE (fixed-shape) ──
    func forwardOneFixed(tokenId: Int, pos: Int) throws -> Int {
        let t0 = CFAbsoluteTimeGetCurrent()
        let embedding = embd.lookup(tokenId)
        let seq = cacheSeqLen

        // x: overwrite pre-allocated buffer
        for j in 0..<cfg.dModel { xPtr[j] = Float16(embedding[j]) }

        // RoPE
        let ropePos = min(pos, cfg.ropeCos.count - 1)
        for j in 0..<dHalf {
            cosPtr[j] = Float16(cfg.ropeCos[ropePos][j])
            sinPtr[j] = Float16(cfg.ropeSin[ropePos][j])
        }

        // attn_mask: unmask position seq (0=valid for positions 0..seq)
        // On first call (seq=0), position 0 is unmasked.
        // On subsequent calls, we unmask one more position.
        attnMaskPtr![seq] = 0  // Unmask the new position

        // kv_write_mask: clear previous, set current
        if seq > 0 { kvWriteMaskPtr![seq - 1] = 0 }
        kvWriteMaskPtr![seq] = Float16(1.0)

        var inputDict: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: xArr),
            "rope_cos": MLFeatureValue(multiArray: cosArr),
            "rope_sin": MLFeatureValue(multiArray: sinArr),
            "attn_mask": MLFeatureValue(multiArray: attnMaskArr!),
            "kv_write_mask": MLFeatureValue(multiArray: kvWriteMaskArr!),
        ]

        // KV cache: fixed-size views — shapes NEVER change
        for layer in 0..<cfg.nLayers {
            inputDict[kCacheKeys[layer]] = MLFeatureValue(multiArray: fixedKViews![layer])
            inputDict[vCacheKeys[layer]] = MLFeatureValue(multiArray: fixedVViews![layer])
        }

        let t1 = CFAbsoluteTimeGetCurrent()
        let provider = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let result = try model.prediction(from: provider)
        let t2 = CFAbsoluteTimeGetCurrent()

        // Write new KV entries into pre-allocated buffer at position `seq`
        for layer in 0..<cfg.nLayers {
            let newK = result.featureValue(for: newKKeys[layer])!.multiArrayValue!
            let newV = result.featureValue(for: newVKeys[layer])!.multiArrayValue!
            let srcK = newK.dataPointer.assumingMemoryBound(to: Float16.self)
            let srcV = newV.dataPointer.assumingMemoryBound(to: Float16.self)
            for h in 0..<cfg.nKvHeads {
                let dstOffset = h * maxSeq * cfg.dHead + seq * cfg.dHead
                let srcOffset = h * cfg.dHead
                memcpy(kCachePtr[layer] + dstOffset, srcK + srcOffset,
                       cfg.dHead * MemoryLayout<Float16>.size)
                memcpy(vCachePtr[layer] + dstOffset, srcV + srcOffset,
                       cfg.dHead * MemoryLayout<Float16>.size)
            }
        }
        cacheSeqLen += 1

        let logits = result.featureValue(for: "logits")!.multiArrayValue!
        let t3 = CFAbsoluteTimeGetCurrent()
        tPack += t1 - t0
        tPredict += t2 - t1
        tExtract += t3 - t2
        nForward += 1
        return argmax(logits)
    }

    // ── Forward one token through ANE (flexible-shape) ──
    func forwardOneFlex(tokenId: Int, pos: Int) throws -> Int {
        let t0 = CFAbsoluteTimeGetCurrent()
        let embedding = embd.lookup(tokenId)

        // x: (1, d, 1, 1) — overwrite pre-allocated buffer
        for j in 0..<cfg.dModel {
            xPtr[j] = Float16(embedding[j])
        }

        // RoPE — overwrite pre-allocated buffers
        let ropePos = min(pos, cfg.ropeCos.count - 1)
        for j in 0..<dHalf {
            cosPtr[j] = Float16(cfg.ropeCos[ropePos][j])
            sinPtr[j] = Float16(cfg.ropeSin[ropePos][j])
        }

        var inputDict: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: xArr),
            "rope_cos": MLFeatureValue(multiArray: cosArr),
            "rope_sin": MLFeatureValue(multiArray: sinArr),
        ]

        // KV cache: zero-copy strided views into pre-allocated buffers
        // The cache is (nKvHeads, maxSeq, dHead) contiguous.
        // We expose (1, nKvHeads, seqToUse, dHead) via strides that skip
        // over unused maxSeq positions — no memcpy needed.
        let seq = cacheSeqLen
        let seqToUse = max(seq, 1)
        for layer in 0..<cfg.nLayers {
            let kView = try MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(kCachePtr[layer]),
                shape: [1, NSNumber(value: cfg.nKvHeads), NSNumber(value: seqToUse), NSNumber(value: cfg.dHead)],
                dataType: .float16,
                strides: [
                    NSNumber(value: cfg.nKvHeads * maxSeq * cfg.dHead),
                    NSNumber(value: maxSeq * cfg.dHead),
                    NSNumber(value: cfg.dHead),
                    NSNumber(value: 1)
                ],
                deallocator: nil
            )
            let vView = try MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(vCachePtr[layer]),
                shape: [1, NSNumber(value: cfg.nKvHeads), NSNumber(value: seqToUse), NSNumber(value: cfg.dHead)],
                dataType: .float16,
                strides: [
                    NSNumber(value: cfg.nKvHeads * maxSeq * cfg.dHead),
                    NSNumber(value: maxSeq * cfg.dHead),
                    NSNumber(value: cfg.dHead),
                    NSNumber(value: 1)
                ],
                deallocator: nil
            )
            inputDict[kCacheKeys[layer]] = MLFeatureValue(multiArray: kView)
            inputDict[vCacheKeys[layer]] = MLFeatureValue(multiArray: vView)
        }

        let t1 = CFAbsoluteTimeGetCurrent()
        let provider = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let result = try model.prediction(from: provider)
        let t2 = CFAbsoluteTimeGetCurrent()

        // Append new KV entries — write directly into pre-allocated buffers
        // Stepanov: cursor-based append, no reallocation
        for layer in 0..<cfg.nLayers {
            let newK = result.featureValue(for: newKKeys[layer])!.multiArrayValue!
            let newV = result.featureValue(for: newVKeys[layer])!.multiArrayValue!
            let srcK = newK.dataPointer.assumingMemoryBound(to: Float16.self)
            let srcV = newV.dataPointer.assumingMemoryBound(to: Float16.self)

            // new_k/new_v: (1, nkv, 1, dh) → write into (nkv, maxSeq, dh) at pos=seq
            for h in 0..<cfg.nKvHeads {
                let dstOffset = h * maxSeq * cfg.dHead + seq * cfg.dHead
                let srcOffset = h * cfg.dHead
                memcpy(kCachePtr[layer] + dstOffset, srcK + srcOffset,
                       cfg.dHead * MemoryLayout<Float16>.size)
                memcpy(vCachePtr[layer] + dstOffset, srcV + srcOffset,
                       cfg.dHead * MemoryLayout<Float16>.size)
            }
        }
        cacheSeqLen += 1

        let logits = result.featureValue(for: "logits")!.multiArrayValue!
        let t3 = CFAbsoluteTimeGetCurrent()
        tPack += t1 - t0
        tPredict += t2 - t1
        tExtract += t3 - t2
        nForward += 1
        return argmax(logits)
    }

    // ── Generation loop ──
    let modeStr = useStateful ? "STATEFUL (KV on-device, zero copy)" :
                  (useFixed ? "FIXED shapes (no ANE replanning)" : "flexible shapes")
    print("\nGenerating (max \(maxTokens) tokens, \(modeStr))...")
    print("─" + String(repeating: "─", count: 55))
    fflush(stdout)

    // Dispatch: stateful > fixed > flex
    let forward: (Int, Int) throws -> Int =
        useStateful ? forwardOneStateful :
        (useFixed ? forwardOneFixed : forwardOneFlex)

    var generated = [Int]()
    let startTime = CFAbsoluteTimeGetCurrent()

    // Prefill
    var lastToken = 0
    for (pos, tok) in tokens.enumerated() {
        lastToken = try forward(tok, pos)
    }
    generated.append(lastToken)
    // Stream first decoded token
    let firstText = tokenizer.decode([lastToken])
    print(firstText, terminator: "")
    fflush(stdout)

    let prefillTime = CFAbsoluteTimeGetCurrent() - startTime

    // Decode
    let decodeStart = CFAbsoluteTimeGetCurrent()
    for step in 0..<(maxTokens - 1) {
        let pos = tokens.count + step
        if pos >= cfg.maxSeqLen - 1 { break }

        let next = try forward(generated.last!, pos)
        generated.append(next)

        // Stream decoded text
        let text = tokenizer.decode([next])
        print(text, terminator: "")
        fflush(stdout)

        // Stop on EOS / EOT
        if next == cfg.eosTokenId || next == 151645 || next == 151643 { break }
    }
    let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
    let totalTime = CFAbsoluteTimeGetCurrent() - startTime

    print("\n")

    let fullText = tokenizer.decode(generated)
    let decodeTokPerSec = decodeTime > 0 ? Double(generated.count - 1) / decodeTime : 0

    print("""
    ╔════════════════════════════════════════════════════════╗
    ║  QwenANE · Conv-Only Inference on Neural Engine        ║
    ╚════════════════════════════════════════════════════════╝

      Model:       Qwen2.5-0.5B (\(cfg.nLayers) layers)
      Primitive:   Conv2d(1×1) — every matmul
      Compute:     Apple Neural Engine (fp16)
      KV Cache:    Pre-alloc fp16, raw-ptr copy (\(cfg.maxSeqLen) positions)
      Tokenizer:   BPE (\(tokenizer.vocab.count) vocab)

      Prefill:     \(tokens.count) tokens → \(String(format: "%.1f", prefillTime * 1000)) ms (\(String(format: "%.0f", Double(tokens.count) / prefillTime)) tok/s)
      Decode:      \(generated.count) tokens → \(String(format: "%.1f", decodeTime * 1000)) ms (\(String(format: "%.1f", decodeTokPerSec)) tok/s)
      Total:       \(String(format: "%.2f", totalTime)) s

      ── Hot Path Breakdown (\(nForward) forward calls) ──
      Pack inputs:    \(String(format: "%.1f", tPack * 1000)) ms (\(String(format: "%.1f", tPack / (tPack + tPredict + tExtract) * 100))%)
      ANE predict:    \(String(format: "%.1f", tPredict * 1000)) ms (\(String(format: "%.1f", tPredict / (tPack + tPredict + tExtract) * 100))%)
      Extract output: \(String(format: "%.1f", tExtract * 1000)) ms (\(String(format: "%.1f", tExtract / (tPack + tPredict + tExtract) * 100))%)
      Avg per token:  pack=\(String(format: "%.2f", tPack / Double(nForward) * 1000))ms predict=\(String(format: "%.2f", tPredict / Double(nForward) * 1000))ms extract=\(String(format: "%.2f", tExtract / Double(nForward) * 1000))ms

      Output:      \(fullText.prefix(80))...
    """)
}

try main()
