// GOL-ANE Qwen: Conv-only transformer inference on Apple Neural Engine.
//
// Every linear projection is a Conv2d(1×1). The ANE's convolution engine
// computes all matmuls. The host manages the KV cache and autoregressive loop.
//
// Build: swiftc -O -framework CoreML -framework Accelerate -o qwen_conv main_qwen.swift
// Usage: ./qwen_conv [--layers N] [--prompt "text"]

import CoreML
import Foundation
import Accelerate

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

    static func fromJSON(_ path: String) -> (QwenConfig, [[Float]], [[Float]]) {
        let data = try! Data(contentsOf: URL(fileURLWithPath: path))
        let json = try! JSONSerialization.jsonObject(with: data) as! [String: Any]

        let config = QwenConfig(
            dModel: json["d_model"] as! Int,
            nHeads: json["n_heads"] as! Int,
            nKvHeads: json["n_kv_heads"] as! Int,
            dHead: json["d_head"] as! Int,
            dFf: json["d_ff"] as! Int,
            vocabSize: json["vocab_size"] as! Int,
            nLayers: json["n_layers_exported"] as! Int,
            maxSeqLen: json["max_seq_len"] as! Int,
            rmsNormEps: Float(json["rms_norm_eps"] as! Double),
            ropeFreqBase: Float(json["rope_freq_base"] as! Double)
        )

        // RoPE tables: (max_seq_len, d_half)
        let cosRaw = json["rope_cos"] as! [[Double]]
        let sinRaw = json["rope_sin"] as! [[Double]]
        let ropeCos = cosRaw.map { $0.map { Float($0) } }
        let ropeSin = sinRaw.map { $0.map { Float($0) } }

        return (config, ropeCos, ropeSin)
    }
}

// MARK: - Embeddings

class TokenEmbeddings {
    let data: [Float]
    let vocab: Int
    let dim: Int

    init(path: String, vocab: Int, dim: Int) {
        let url = URL(fileURLWithPath: path)
        let raw = try! Data(contentsOf: url)
        self.data = raw.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
        self.vocab = vocab
        self.dim = dim
        assert(data.count == vocab * dim, "Embedding size mismatch: \(data.count) vs \(vocab * dim)")
    }

    func lookup(_ tokenId: Int) -> [Float] {
        let start = tokenId * dim
        return Array(data[start..<start + dim])
    }
}

// MARK: - KV Cache

class KVCache {
    let nLayers: Int
    let nKvHeads: Int
    let dHead: Int

    // Per-layer caches: (nKvHeads, seq_len, dHead) stored flat
    var kCaches: [[Float]]
    var vCaches: [[Float]]
    var seqLen: Int

    init(nLayers: Int, nKvHeads: Int, dHead: Int) {
        self.nLayers = nLayers
        self.nKvHeads = nKvHeads
        self.dHead = dHead
        self.kCaches = Array(repeating: [], count: nLayers)
        self.vCaches = Array(repeating: [], count: nLayers)
        self.seqLen = 0
    }

    func appendKV(layer: Int, newK: MLMultiArray, newV: MLMultiArray) {
        // newK/newV: (1, nKvHeads, 1, dHead)
        let count = nKvHeads * dHead
        // Handle both Float32 and Float16
        for i in 0..<count {
            kCaches[layer].append(newK[i].floatValue)
            vCaches[layer].append(newV[i].floatValue)
        }
    }

    func cacheArray(layer: Int, isKey: Bool) throws -> MLMultiArray {
        let cache = isKey ? kCaches[layer] : vCaches[layer]
        let seq = seqLen
        if seq == 0 {
            // Return a dummy (1, nKvHeads, 1, dHead) with zeros for first step
            let arr = try MLMultiArray(
                shape: [1, NSNumber(value: nKvHeads), 1, NSNumber(value: dHead)],
                dataType: .float32)
            let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<(nKvHeads * dHead) { ptr[i] = 0 }
            return arr
        }
        let arr = try MLMultiArray(
            shape: [1, NSNumber(value: nKvHeads), NSNumber(value: seq), NSNumber(value: dHead)],
            dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        // Data in cache is stored as seq consecutive (nKvHeads * dHead) blocks
        // Need to transpose to (nKvHeads, seq, dHead)
        for s in 0..<seq {
            for h in 0..<nKvHeads {
                for d in 0..<dHead {
                    let srcIdx = s * nKvHeads * dHead + h * dHead + d
                    let dstIdx = h * seq * dHead + s * dHead + d
                    ptr[dstIdx] = cache[srcIdx]
                }
            }
        }
        return arr
    }

    func step() {
        seqLen += 1
    }
}

// MARK: - Argmax

func argmax(_ arr: MLMultiArray) -> Int {
    let count = arr.count
    var maxVal: Float = -Float.infinity
    var maxIdx: Int = 0

    // Handle both Float32 and Float16 MLMultiArrays
    if arr.dataType == .float32 {
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            if ptr[i] > maxVal {
                maxVal = ptr[i]
                maxIdx = i
            }
        }
    } else {
        // Float16 or other — use subscript access (slower but safe)
        for i in 0..<count {
            let val = arr[i].floatValue
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }
    }
    return maxIdx
}

// MARK: - Main

func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    var nLayers = 1
    var prompt = "The meaning of life is"

    // Parse args
    var i = 0
    while i < args.count {
        if args[i] == "--layers", i + 1 < args.count {
            nLayers = Int(args[i + 1])!
            i += 2
        } else if args[i] == "--prompt", i + 1 < args.count {
            prompt = args[i + 1]
            i += 2
        } else {
            i += 1
        }
    }

    let modelName = "QwenConvOnly_\(nLayers)L"

    // Load config and RoPE tables
    print("Loading config from \(modelName)_meta.json ...")
    let (cfg, ropeCos, ropeSin) = QwenConfig.fromJSON("\(modelName)_meta.json")
    print("  d_model=\(cfg.dModel), n_heads=\(cfg.nHeads), n_kv_heads=\(cfg.nKvHeads)")
    print("  layers=\(cfg.nLayers), vocab=\(cfg.vocabSize)")
    print("  RoPE tables: \(ropeCos.count) positions × \(ropeCos[0].count) dims")

    // Load embeddings
    print("Loading embeddings from \(modelName)_embd.bin ...")
    let embd = TokenEmbeddings(
        path: "\(modelName)_embd.bin",
        vocab: cfg.vocabSize,
        dim: cfg.dModel)
    print("  \(embd.vocab) × \(embd.dim)")
    fflush(stdout)

    // Load CoreML model
    print("Loading CoreML model...")
    let modelURL = URL(fileURLWithPath: "\(modelName).mlmodelc")
    guard FileManager.default.fileExists(atPath: "\(modelName).mlmodelc") else {
        print("Error: \(modelName).mlmodelc not found. Run:")
        print("  xcrun coremlcompiler compile \(modelName).mlpackage .")
        return
    }

    let aneConfig = MLModelConfiguration()
    aneConfig.computeUnits = .cpuAndNeuralEngine
    let model = try MLModel(contentsOf: modelURL, configuration: aneConfig)
    print("  Loaded ✓")
    fflush(stdout)

    // Simple tokenizer: just use byte-level encoding as a demo
    // In production, you'd use the BPE tokenizer from the GGUF
    // For now, encode each character as its ASCII value (capped at vocab)
    print("\nPrompt: \"\(prompt)\"")

    // For the 1-layer POC, we use simple character-level tokens
    // A real implementation would use the BPE tokenizer
    let tokens = Array(prompt.utf8).map { Int($0) }
    print("Tokens: \(tokens)")

    // Initialize KV cache
    let kvCache = KVCache(nLayers: cfg.nLayers, nKvHeads: cfg.nKvHeads, dHead: cfg.dHead)
    let dHalf = cfg.dHead / 2

    // Forward pass helper
    func forwardOne(tokenId: Int, pos: Int) throws -> Int {
        // 1. Embedding lookup (host-side, not a conv)
        let embedding = embd.lookup(tokenId)

        // 2. Pack as (1, d, 1, 1) conv input
        let xArr = try MLMultiArray(
            shape: [1, NSNumber(value: cfg.dModel), 1, 1],
            dataType: .float32)
        let xPtr = xArr.dataPointer.assumingMemoryBound(to: Float.self)
        for j in 0..<cfg.dModel { xPtr[j] = embedding[j] }

        // 3. RoPE cos/sin for this position
        let cosArr = try MLMultiArray(shape: [1, NSNumber(value: dHalf)], dataType: .float32)
        let sinArr = try MLMultiArray(shape: [1, NSNumber(value: dHalf)], dataType: .float32)
        let cosPtr = cosArr.dataPointer.assumingMemoryBound(to: Float.self)
        let sinPtr = sinArr.dataPointer.assumingMemoryBound(to: Float.self)
        let ropePos = min(pos, ropeCos.count - 1)
        for j in 0..<dHalf {
            cosPtr[j] = ropeCos[ropePos][j]
            sinPtr[j] = ropeSin[ropePos][j]
        }

        // 4. Build input dictionary
        var inputDict: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: xArr),
            "rope_cos": MLFeatureValue(multiArray: cosArr),
            "rope_sin": MLFeatureValue(multiArray: sinArr),
        ]

        // 5. KV caches
        for layer in 0..<cfg.nLayers {
            let kArr = try kvCache.cacheArray(layer: layer, isKey: true)
            let vArr = try kvCache.cacheArray(layer: layer, isKey: false)
            inputDict["k_cache_\(layer)"] = MLFeatureValue(multiArray: kArr)
            inputDict["v_cache_\(layer)"] = MLFeatureValue(multiArray: vArr)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: inputDict)

        // 6. Run model on ANE
        let result = try model.prediction(from: provider)

        // 7. Extract logits and new KV entries
        let logits = result.featureValue(for: "logits")!.multiArrayValue!
        let nextToken = argmax(logits)

        // 8. Append new KV to cache
        for layer in 0..<cfg.nLayers {
            let newK = result.featureValue(for: "new_k_\(layer)")!.multiArrayValue!
            let newV = result.featureValue(for: "new_v_\(layer)")!.multiArrayValue!
            kvCache.appendKV(layer: layer, newK: newK, newV: newV)
        }
        kvCache.step()

        return nextToken
    }

    // Generation loop
    let maxNew = 32
    print("\nGenerating (\(cfg.nLayers) layer(s), \(maxNew) tokens max)...")
    print("─────────────────────────────────────")

    var generated: [Int] = []
    let startTime = CFAbsoluteTimeGetCurrent()

    // Prefill: process each prompt token
    var lastLogitToken = 0
    for (pos, tokenId) in tokens.enumerated() {
        lastLogitToken = try forwardOne(tokenId: tokenId, pos: pos)
        if pos < tokens.count - 1 {
            // Still prefilling, don't output
        }
    }
    generated.append(lastLogitToken)

    let prefillTime = CFAbsoluteTimeGetCurrent() - startTime
    print("Prefill: \(tokens.count) tokens in \(String(format: "%.1f", prefillTime * 1000)) ms")

    // Decode
    let decodeStart = CFAbsoluteTimeGetCurrent()
    for step in 0..<(maxNew - 1) {
        let pos = tokens.count + step
        let next = try forwardOne(tokenId: generated.last!, pos: pos)
        generated.append(next)

        // Simple stopping: EOS tokens
        if next == 151643 || next == 151645 { break }
    }
    let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart

    let totalTime = CFAbsoluteTimeGetCurrent() - startTime

    // Output raw token IDs (without a proper tokenizer, we can't decode to text)
    // But we CAN show bytes for tokens < 256
    var outputStr = ""
    for t in generated {
        if t < 128, let scalar = Unicode.Scalar(t) {
            outputStr += String(scalar)
        } else {
            outputStr += "[<\(t)>]"
        }
    }

    print("\nGenerated tokens: \(generated)")
    print("Decoded (approx): \(outputStr)")

    print("""

    ╔════════════════════════════════════════════════════════╗
    ║  GOL-ANE Qwen  ·  Conv-Only Inference                 ║
    ╚════════════════════════════════════════════════════════╝

      Model:           Qwen2.5-0.5B (\(cfg.nLayers) layer\(cfg.nLayers > 1 ? "s" : ""))
      Primitive:       Conv2d(1×1)  — all matmuls
      Compute:         Apple Neural Engine

      Prefill:         \(tokens.count) tokens → \(String(format: "%.1f", prefillTime * 1000)) ms
      Decode:          \(generated.count) tokens → \(String(format: "%.1f", decodeTime * 1000)) ms
      Total:           \(String(format: "%.1f", totalTime * 1000)) ms
      Throughput:      \(String(format: "%.1f", Double(generated.count) / decodeTime)) tok/sec (decode)
    """)
}

try main()
