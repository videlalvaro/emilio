// pf_e2e.swift — Pure-Swift end-to-end forward pass for the OpenAI PII model.
//
// Pipeline (per sentence, B=1, T=128):
//   ids -> embed (CPU)
//   for L in 0..<8:
//     x_attn, normed_x, router_probs =                    [ANE — fused pack]
//         PF_fused_L{L}_T128.predict(x_in, pad_add)
//     topk_idx, topk_w = argpartition(router_probs)       [CPU — index pick only]
//     for each distinct expert e in topk_idx:
//       y = PF_expert_L{L}_{e}_B64.predict(normed_x)     [ANE, concurrent]
//     scatter w*K*y back into delta
//     x = x_attn + delta
//   logits = PF_tail_T128.predict(x)                      [CoreML tail pack]
//     (fused final RMSNorm + unembed [640→33])
//
// ----- CPU residency justification (April 2026) -----
// ALL arithmetic compute (attn, MLP-norm, gate, softmax, SwiGLU experts, final
// norm, unembed) runs via CoreML packs. The remaining CPU ops fall into three
// categories — none can move to ANE with current public APIs:
//
// A. CoreML API boundary (pack/unpack/residual-add):
//    MLModel.prediction() is a full host↔ANE round-trip. No public API keeps
//    tensors resident on-device between calls. Apple's private
//    _ANEChainingRequest supports loopback symbols + shared memory pools, but
//    it is SPI. Until CoreML exposes chaining, every predict() boundary
//    requires CPU-side format conversion (transpose + fp32↔fp16).
//
// B. Dynamic indexing (embed, top-K, expert-input gather, scatter):
//    ANE is a fixed-dataflow SIMD engine with no gather, scatter, sort, or
//    argmax primitives. Embedding is a 244 MB table (>96 MB ANE cliff) with
//    runtime-dependent token-ID indexing. Top-K requires comparison-based
//    sorting. Expert input gather and scatter both index by routing decisions
//    that change per sentence. The only ANE-viable alternative is dense MoE
//    (all 128 experts), but 128×2.4 MB = 307 MB per layer > 96 MB cliff.
//
// C. Economics (residual add: 82K FLOPs < CoreML dispatch overhead of ~0.1 ms).
// -----------------------------------------------------------------
// Validate each sentence's logits vs golden.
//
// Build:
//   swiftc -O -framework CoreML -framework Accelerate \
//          -o pf_e2e pf_e2e.swift
// Run:
//   ./pf_e2e
import CoreML
import Accelerate
import Foundation

// ----- Constants (mirror manifest) -----
let DIR        = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let REPO       = URL(fileURLWithPath: DIR).deletingLastPathComponent().deletingLastPathComponent().path
let SDIR       = "\(DIR)/PF_swift"
let N_LAYERS   = 8
let D_MODEL    = 640
let N_EXPERTS  = 128
let TOPK       = 4
let T_SEQ      = 128
let VOCAB      = 200064
let N_LABELS   = 33
let RMS_EPS    = Float(1e-5)
let B_PACK     = 64                    // expert pack batch size
let N_SENT     = 8

// ----- Helpers -----
func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

// Round fp32 to bf16 (truncate-to-nearest) — matches Python opf bf16 forward.
@inline(__always)
func bf16round(_ x: Float) -> Float {
    let bits = x.bitPattern &+ 0x8000
    return Float(bitPattern: bits & 0xFFFF_0000)
}
// Vectorized: rebind [Float] -> [UInt32] and let -O auto-vectorize the simple bit op.
@inline(__always)
func bf16RoundInPlace(_ a: inout [Float]) {
    let n = a.count
    a.withUnsafeMutableBufferPointer { buf in
        buf.baseAddress!.withMemoryRebound(to: UInt32.self, capacity: n) { p in
            for i in 0..<n {
                p[i] = (p[i] &+ 0x0000_8000) & 0xFFFF_0000
            }
        }
    }
}
func ms(_ a: UInt64, _ b: UInt64) -> Double { Double(b - a) / 1e6 }
func loadOnANE(_ p: String) throws -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: URL(fileURLWithPath: p), configuration: cfg)
}
func readBinF16(_ name: String, _ count: Int) -> [Float16] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: "\(SDIR)/\(name).bin"))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self).prefix(count)) }
}
func readBinI32(_ name: String, _ count: Int) -> [Int32] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: "\(SDIR)/\(name).bin"))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self).prefix(count)) }
}
func readBinF32(_ name: String, _ count: Int) -> [Float] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: "\(SDIR)/\(name).bin"))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
}

// fp16 -> Float (uses Accelerate)
func f16ToF32(_ src: [Float16]) -> [Float] {
    var dst = [Float](repeating: 0, count: src.count)
    src.withUnsafeBufferPointer { sb in
        dst.withUnsafeMutableBufferPointer { db in
            var srcImg = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: sb.baseAddress!),
                                       height: 1, width: vImagePixelCount(src.count),
                                       rowBytes: src.count * 2)
            var dstImg = vImage_Buffer(data: db.baseAddress!,
                                       height: 1, width: vImagePixelCount(src.count),
                                       rowBytes: src.count * 4)
            vImageConvert_Planar16FtoPlanarF(&srcImg, &dstImg, 0)
        }
    }
    return dst
}

// RMSNorm in-place on x[T*D] using f32 scale (vDSP-accelerated).
func rmsNorm(_ x: inout [Float], _ scale: [Float], T: Int, D: Int, eps: Float) {
    let invD = Float(D)
    x.withUnsafeMutableBufferPointer { xb in
        scale.withUnsafeBufferPointer { sb in
            let xp = xb.baseAddress!
            let sp = sb.baseAddress!
            for t in 0..<T {
                let row = xp + t * D
                var ss: Float = 0
                vDSP_svesq(row, 1, &ss, vDSP_Length(D))
                var inv = 1.0 / sqrtf(ss / invD + eps)
                vDSP_vsmul(row, 1, &inv, row, 1, vDSP_Length(D))
                vDSP_vmul(row, 1, sp, 1, row, 1, vDSP_Length(D))
            }
        }
    }
}

// out[T,K] gate logits = x[T,D] @ W[K,D]^T + b[K]
func linear(_ x: [Float], W: [Float], b: [Float], T: Int, D: Int, K: Int) -> [Float] {
    var out = [Float](repeating: 0, count: T * K)
    // y = x * W^T + bias ; use cblas_sgemm in row-major.
    // GEMM: M=T, N=K, K=D
    out.withUnsafeMutableBufferPointer { op in
        x.withUnsafeBufferPointer { xp in
            W.withUnsafeBufferPointer { wp in
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            Int32(T), Int32(K), Int32(D),
                            1.0, xp.baseAddress, Int32(D),
                                 wp.baseAddress, Int32(D),
                            0.0, op.baseAddress, Int32(K))
            }
        }
    }
    // add bias
    for t in 0..<T {
        let off = t * K
        for k in 0..<K { out[off + k] += b[k] }
    }
    return out
}

// Top-K (descending) per row of logits[T,Kall].
struct TopK { let idx: [Int]; let w: [Float] }   // idx[T*K], w[T*K] (already softmax/TOPK)
func topkSoftmax(_ logits: [Float], T: Int, Kall: Int, K: Int) -> TopK {
    var idx = [Int](repeating: 0, count: T * K)
    var wts = [Float](repeating: 0, count: T * K)
    for t in 0..<T {
        let off = t * Kall
        // Find top-K via partial sort.
        var pairs: [(Float, Int)] = []
        pairs.reserveCapacity(Kall)
        for k in 0..<Kall { pairs.append((logits[off + k], k)) }
        pairs.sort { $0.0 > $1.0 }
        // softmax over the K winners
        var maxv: Float = -.infinity
        for k in 0..<K { if pairs[k].0 > maxv { maxv = pairs[k].0 } }
        var sum: Float = 0
        var es = [Float](repeating: 0, count: K)
        for k in 0..<K { es[k] = expf(pairs[k].0 - maxv); sum += es[k] }
        let invSumK = 1.0 / (sum * Float(K))
        for k in 0..<K {
            idx[t*K + k] = pairs[k].1
            wts[t*K + k] = es[k] * invSumK   // softmax / K (matches stored convention)
        }
    }
    return TopK(idx: idx, w: wts)
}

// Top-K from full softmax probabilities (fused pack outputs router_probs = softmax over 128)
// Re-normalize top-K probs: w_i = prob_i / sum(top4) / K  (identical to softmax-over-top-K / K)
func topkFromProbs(_ probs: [Float], T: Int, Kall: Int, K: Int) -> TopK {
    var idx = [Int](repeating: 0, count: T * K)
    var wts = [Float](repeating: 0, count: T * K)
    for t in 0..<T {
        let off = t * Kall
        var pairs: [(Float, Int)] = []
        pairs.reserveCapacity(Kall)
        for k in 0..<Kall { pairs.append((probs[off + k], k)) }
        pairs.sort { $0.0 > $1.0 }
        var sum: Float = 0
        for k in 0..<K { sum += pairs[k].0 }
        let invSumK = 1.0 / (sum * Float(K))
        for k in 0..<K {
            idx[t*K + k] = pairs[k].1
            wts[t*K + k] = pairs[k].0 * invSumK
        }
    }
    return TopK(idx: idx, w: wts)
}

// ===========================================================================
// Load all weights & artifacts
// ===========================================================================
print("[load] reading manifest tensors ...")
let embed       = readBinF16("embedding",        VOCAB * D_MODEL)
// final_norm_scale and unembedding are now baked into the tail pack.

// Router weights (mlpScale, gateW, gateB) are now baked into fused packs — no CPU loading.
let inputIds   = readBinI32("input_ids",      N_SENT * T_SEQ)
let attnMask   = readBinI32("attention_mask", N_SENT * T_SEQ)
let goldenF32  = readBinF32("golden_logits",  N_SENT * T_SEQ * N_LABELS)
print("  loaded weights for \(N_LAYERS) layers, vocab=\(VOCAB)")

print("[load] loading fused .mlmodelc + tail .mlmodelc + 8x128 expert .mlmodelc ...")
let lt0 = nowNs()
var fusedPacks: [MLModel] = []
for L in 0..<N_LAYERS {
    fusedPacks.append(try loadOnANE("\(DIR)/PF_fused_L\(L)_T128.mlmodelc"))
}
let tailPack = try loadOnANE("\(DIR)/PF_tail_T128.mlmodelc")

// Parallel expert loading: 1024 mlmodelc are I/O + driver-registration bound.
// Concurrent loading with 10 lanes cuts ~20s sequential → ~2–3s.
let totalExperts = N_LAYERS * N_EXPERTS
let expertFlat = UnsafeMutablePointer<MLModel?>.allocate(capacity: totalExperts)
expertFlat.initialize(repeating: nil, count: totalExperts)
let loadGroup = DispatchGroup()
let loadQ = DispatchQueue(label: "expert-load", attributes: .concurrent)
let loadSema = DispatchSemaphore(value: 10)  // cap concurrent file handles
for L in 0..<N_LAYERS {
    for e in 0..<N_EXPERTS {
        loadGroup.enter()
        loadQ.async {
            loadSema.wait()
            defer { loadSema.signal(); loadGroup.leave() }
            do {
                let m = try loadOnANE("\(DIR)/PF_expert_L\(L)_\(e)_B64_fp16.mlmodelc")
                expertFlat[L * N_EXPERTS + e] = m
            } catch {
                fatalError("[load] expert L\(L) e\(e): \(error)")
            }
        }
    }
}
loadGroup.wait()
var experts: [[MLModel]] = []
for L in 0..<N_LAYERS {
    var arr: [MLModel] = []
    for e in 0..<N_EXPERTS {
        arr.append(expertFlat[L * N_EXPERTS + e]!)
    }
    experts.append(arr)
}
expertFlat.deallocate()
print(String(format: "[load] all artifacts in %.1fs", ms(lt0, nowNs())/1000))

// ===========================================================================
// Forward pass for one sentence: input_ids[T] + mask[T] -> logits[T*N_LABELS]
// ===========================================================================
let q = DispatchQueue.global(qos: .userInitiated)

func dumpF32(_ a: [Float], _ name: String) {
    let dir = "\(REPO)/python/privacy/out/swift_dump"
    let path = "\(dir)/\(name).bin"
    try? FileManager.default.createDirectory(atPath: dir,
                                             withIntermediateDirectories: true)
    a.withUnsafeBufferPointer { p in
        try? Data(bytes: p.baseAddress!, count: p.count * 4)
            .write(to: URL(fileURLWithPath: path))
    }
}
func dumpI32(_ a: [Int32], _ name: String) {
    let dir = "\(REPO)/python/privacy/out/swift_dump"
    let path = "\(dir)/\(name).bin"
    a.withUnsafeBufferPointer { p in
        try? Data(bytes: p.baseAddress!, count: p.count * 4)
            .write(to: URL(fileURLWithPath: path))
    }
}

func forwardSentence(ids: ArraySlice<Int32>, mask: ArraySlice<Int32>,
                     dumpStages: Bool = false,
                     goldenTopK: Bool = false) throws -> [Float] {
    let idsArr = Array(ids); let maskArr = Array(mask)
    // ---- Embed (CPU) -> x[T, D] fp32 ----
    // CPU reason B: 244 MB table > 96 MB ANE cliff; gather-by-token-ID has no
    // ANE primitive (would need 200K-class one-hot matmul = 16B MADs).
    var x = [Float](repeating: 0, count: T_SEQ * D_MODEL)
    embed.withUnsafeBufferPointer { ep in
        x.withUnsafeMutableBufferPointer { xp in
            for t in 0..<T_SEQ {
                let tok = Int(idsArr[t])
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: ep.baseAddress! + tok * D_MODEL),
                    height: 1, width: vImagePixelCount(D_MODEL),
                    rowBytes: D_MODEL * MemoryLayout<Float16>.stride)
                var dst = vImage_Buffer(
                    data: UnsafeMutableRawPointer(xp.baseAddress! + t * D_MODEL),
                    height: 1, width: vImagePixelCount(D_MODEL),
                    rowBytes: D_MODEL * MemoryLayout<Float>.stride)
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
            }
        }
    }

    // pad_add[T] = 0 if mask=1 else -1e4
    var padAddF16 = [Float16](repeating: 0, count: T_SEQ)
    for t in 0..<T_SEQ { padAddF16[t] = maskArr[t] > 0 ? Float16(0) : Float16(-1e4) }

    // bf16round removed — fp16 has more precision than bf16, ANE computes in fp16,
    // and full pipeline validates (span-F1 100%, cos ≥ 0.99).
    if dumpStages { dumpF32(x, "embed") }

    for L in 0..<N_LAYERS {
        // ---- PACK x → fp16 [1,D,1,T] (CPU reason A: CoreML API boundary) ----
        // Public MLModel.prediction() requires host-side MLMultiArray population.
        let xin = try MLMultiArray(shape: [1, NSNumber(value: D_MODEL), 1,
                                           NSNumber(value: T_SEQ)], dataType: .float16)
        let xinP = xin.dataPointer.bindMemory(to: Float16.self, capacity: xin.count)
        let xS = xin.strides.map { Int(truncating: $0) }
        // Fast path: innermost (T) stride == 1 and D-stride == T_SEQ (no padding).
        if xS[3] == 1 && xS[1] == T_SEQ {
            // 1) Transpose x[T,D] -> xT[D,T] via vDSP, then 2) single-shot fp32->fp16.
            var xT = [Float](repeating: 0, count: D_MODEL * T_SEQ)
            x.withUnsafeBufferPointer { xp in
                xT.withUnsafeMutableBufferPointer { xtp in
                    vDSP_mtrans(xp.baseAddress!, 1, xtp.baseAddress!, 1,
                                vDSP_Length(D_MODEL), vDSP_Length(T_SEQ))
                }
            }
            xT.withUnsafeMutableBufferPointer { xtp in
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(xtp.baseAddress!),
                    height: vImagePixelCount(D_MODEL),
                    width: vImagePixelCount(T_SEQ),
                    rowBytes: T_SEQ * MemoryLayout<Float>.stride)
                var dst = vImage_Buffer(
                    data: UnsafeMutableRawPointer(xinP),
                    height: vImagePixelCount(D_MODEL),
                    width: vImagePixelCount(T_SEQ),
                    rowBytes: T_SEQ * MemoryLayout<Float16>.stride)
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
            }
        } else {
            // Fallback: scalar.
            for d in 0..<D_MODEL {
                for t in 0..<T_SEQ {
                    xinP[d * xS[1] + t * xS[3]] = Float16(x[t * D_MODEL + d])
                }
            }
        }
        let padIn = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: T_SEQ)],
                                     dataType: .float16)
        let padP = padIn.dataPointer.bindMemory(to: Float16.self, capacity: padIn.count)
        let pS = padIn.strides.map { Int(truncating: $0) }
        for t in 0..<T_SEQ { padP[t * pS[3]] = padAddF16[t] }

        let prov = try MLDictionaryFeatureProvider(dictionary: [
            "x_in": xin, "pad_add": padIn,
        ])
        // ---- FUSED PREDICT (attn + MLP-norm + gate + softmax — single ANE call) ----
        let pred = try fusedPacks[L].prediction(from: prov)

        // x_attn [1,D,1,T] fp16: attention output with residual
        let xAttnArr = pred.featureValue(for: "x_attn")!.multiArrayValue!
        let xAttnP = xAttnArr.dataPointer.bindMemory(to: Float16.self, capacity: xAttnArr.count)
        let xAttnS = xAttnArr.strides.map { Int(truncating: $0) }

        // normed_x [1,D,1,T] fp16: MLP-normed x, feeds directly to expert packs
        let normedArr = pred.featureValue(for: "normed_x")!.multiArrayValue!
        let normedP = normedArr.dataPointer.bindMemory(to: Float16.self, capacity: normedArr.count)
        let normedS = normedArr.strides.map { Int(truncating: $0) }

        // router_probs [1,128,1,T] fp16: full softmax over experts
        let routerArr = pred.featureValue(for: "router_probs")!.multiArrayValue!
        let routerP = routerArr.dataPointer.bindMemory(to: Float16.self, capacity: routerArr.count)
        let routerS = routerArr.strides.map { Int(truncating: $0) }

        // Unpack x_attn → x[T,D] fp32 (transpose D <-> T)
        if xAttnS[3] == 1 && xAttnS[1] == T_SEQ {
            var yT = [Float](repeating: 0, count: D_MODEL * T_SEQ)
            yT.withUnsafeMutableBufferPointer { ytp in
                var src = vImage_Buffer(
                    data: UnsafeMutableRawPointer(xAttnP),
                    height: vImagePixelCount(D_MODEL),
                    width: vImagePixelCount(T_SEQ),
                    rowBytes: T_SEQ * MemoryLayout<Float16>.stride)
                var dst = vImage_Buffer(
                    data: UnsafeMutableRawPointer(ytp.baseAddress!),
                    height: vImagePixelCount(D_MODEL),
                    width: vImagePixelCount(T_SEQ),
                    rowBytes: T_SEQ * MemoryLayout<Float>.stride)
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
            }
            yT.withUnsafeBufferPointer { ytp in
                x.withUnsafeMutableBufferPointer { xp in
                    vDSP_mtrans(ytp.baseAddress!, 1, xp.baseAddress!, 1,
                                vDSP_Length(T_SEQ), vDSP_Length(D_MODEL))
                }
            }
        } else {
            for d in 0..<D_MODEL {
                for t in 0..<T_SEQ {
                    x[t * D_MODEL + d] = Float(xAttnP[d * xAttnS[1] + t * xAttnS[3]])
                }
            }
        }
        if let dp = ProcessInfo.processInfo.environment["PF_DUMP_ATTN_L"],
           let dl = Int(dp), dl == L {
            let f16: [Float16] = x.map { Float16($0) }
            let path = "/tmp/_swift_attn_L\(L).bin"
            try? Data(bytes: f16, count: f16.count * 2)
                .write(to: URL(fileURLWithPath: path))
            print("  [dump] wrote \(path)")
        }

        if dumpStages { dumpF32(x, "attn_out_L\(L)") }

        // ---- ROUTER: top-K from ANE softmax probs ----
        // CPU reason B: ANE has no sort/argmax. Softmax done on ANE; only the
        // index selection (sort 128 floats, pick top 4) remains here.
        var routerProbs = [Float](repeating: 0, count: T_SEQ * N_EXPERTS)
        for t in 0..<T_SEQ {
            for k in 0..<N_EXPERTS {
                routerProbs[t * N_EXPERTS + k] = Float(routerP[k * routerS[1] + t * routerS[3]])
            }
        }
        var tk = topkFromProbs(routerProbs, T: T_SEQ, Kall: N_EXPERTS, K: TOPK)
        if goldenTopK {
            let DDIR = "\(REPO)/python/privacy/out/swift_dump"
            let nE = T_SEQ * TOPK
            let didx = try Data(contentsOf: URL(fileURLWithPath: "\(DDIR)/golden_topk_idx_L\(L).bin"))
            let dw   = try Data(contentsOf: URL(fileURLWithPath: "\(DDIR)/golden_topk_w_L\(L).bin"))
            let gIdx: [Int32] = didx.withUnsafeBytes { Array($0.bindMemory(to: Int32.self).prefix(nE)) }
            let gW:   [Float] = dw.withUnsafeBytes  { Array($0.bindMemory(to: Float.self).prefix(nE)) }
            tk = TopK(idx: gIdx.map { Int($0) }, w: gW)
        }

        if dumpStages {
            dumpI32(tk.idx.map { Int32($0) }, "topk_idx_L\(L)")
            dumpF32(tk.w, "topk_w_L\(L)")
        }

        // Per-expert dispatch in chunks of B_PACK rows
        var delta = [Float](repeating: 0, count: T_SEQ * D_MODEL)
        let kScale = Float(TOPK)
        var s = 0
        while s < T_SEQ {
            let e_end = min(s + B_PACK, T_SEQ)
            let n_rows = e_end - s
            // Build per-expert assignments for this chunk
            var byExpert: [Int: [(Int, Float)]] = [:]
            for r in 0..<n_rows {
                for k in 0..<TOPK {
                    let eid = tk.idx[(s + r) * TOPK + k]
                    let wv  = tk.w[(s + r) * TOPK + k]
                    if wv == 0 { continue }
                    byExpert[eid, default: []].append((r, wv))
                }
            }
            // Dispatch
            let group = DispatchGroup()
            let lock = NSLock()
            // Each (rows, fp32 contiguous out buf [rows.count * D_MODEL]).
            var outs: [(rows: [(Int, Float)], buf: UnsafeMutablePointer<Float>)] = []
            for (eid, rows) in byExpert {
                let xin2 = try MLMultiArray(shape: [NSNumber(value: B_PACK),
                                                    NSNumber(value: D_MODEL), 1, 1],
                                            dataType: .float16)
                let p2 = xin2.dataPointer.bindMemory(to: Float16.self, capacity: xin2.count)
                memset(p2, 0, xin2.count * MemoryLayout<Float16>.stride)
                let xS2 = xin2.strides.map { Int(truncating: $0) }
                // Fast path: D-axis is contiguous (stride 1), rows packed at slot*xS2[0].
                // normed_x from fused pack is already fp16 — no f32→f16 conversion!
                // CPU reason B: dynamic token→expert assignment changes per sentence;
                // ANE has no gather primitive. Dense MoE would be 307 MB/layer > 96 MB cliff.
                if xS2[1] == 1 {
                    for (slot, (r, _)) in rows.enumerated() {
                        let t = s + r
                        let dstBase = slot * xS2[0]
                        let srcStride = normedS[1]
                        for d in 0..<D_MODEL {
                            p2[dstBase + d] = normedP[d * srcStride + t * normedS[3]]
                        }
                    }
                } else {
                    for (slot, (r, _)) in rows.enumerated() {
                        let t = s + r
                        for d in 0..<D_MODEL {
                            p2[slot * xS2[0] + d * xS2[1]] = normedP[d * normedS[1] + t * normedS[3]]
                        }
                    }
                }
                let prov2 = try MLDictionaryFeatureProvider(dictionary: ["x_in": xin2])
                let outBuf = UnsafeMutablePointer<Float>.allocate(
                    capacity: rows.count * D_MODEL)
                let m = experts[L][eid]
                let rowCount = rows.count
                group.enter()
                q.async {
                    do {
                        let pr = try m.prediction(from: prov2)
                        let ya = pr.featureValue(for: "y_out")!.multiArrayValue!
                        let yap = ya.dataPointer.bindMemory(to: Float16.self,
                                                            capacity: ya.count)
                        let yaS = ya.strides.map { Int(truncating: $0) }
                        // Strided fp16 -> contiguous fp32. Output shape is
                        // [B,D,1,1] but ANE pads channel: typically yaS[1] != 1
                        // (e.g. 32). Compiler will auto-vectorize this tight loop.
                        let s0 = yaS[0], s1 = yaS[1]
                        for slot in 0..<rowCount {
                            let base = slot * s0
                            let dst = slot * D_MODEL
                            for d in 0..<D_MODEL {
                                outBuf[dst + d] = Float(yap[base + d * s1])
                            }
                        }
                    } catch { fatalError("expert L\(L) e\(eid): \(error)") }
                    group.leave()
                }
                lock.lock(); outs.append((rows, outBuf)); lock.unlock()
            }
            group.wait()
            // Scatter: delta[dst..dst+D] += (wv*K) * outBuf[slot*D..(slot+1)*D]
            // CPU reason B: dynamic-index write-scatter; ANE has no indexed-write.
            delta.withUnsafeMutableBufferPointer { dp in
                for o in outs {
                    for (slot, (r, wv)) in o.rows.enumerated() {
                        let dst = (s + r) * D_MODEL
                        var sc = wv * kScale
                        vDSP_vsma(o.buf + slot * D_MODEL, 1, &sc,
                                  dp.baseAddress! + dst, 1,
                                  dp.baseAddress! + dst, 1,
                                  vDSP_Length(D_MODEL))
                    }
                    o.buf.deallocate()
                }
            }
            s = e_end
        }

        // Residual: x += delta (CPU reason C: 82K FLOPs < CoreML dispatch overhead)
        x.withUnsafeMutableBufferPointer { xp in
            delta.withUnsafeBufferPointer { dp in
                vDSP_vadd(xp.baseAddress!, 1, dp.baseAddress!, 1,
                          xp.baseAddress!, 1, vDSP_Length(T_SEQ * D_MODEL))
            }
        }
        // bf16round removed — fp16 ANE compute already validates correctly.
        if dumpStages {
            dumpF32(delta, "mlp_delta_L\(L)")
            dumpF32(x, "post_layer_L\(L)")
        }
    }

    // ---- Final norm + unembed via tail pack ----
    // Pack x[T,D] fp32 → tailIn [1,D,1,T] fp16 (same pack as layer input)
    let tailIn = try MLMultiArray(shape: [1, NSNumber(value: D_MODEL), 1,
                                          NSNumber(value: T_SEQ)], dataType: .float16)
    let tailP = tailIn.dataPointer.bindMemory(to: Float16.self, capacity: tailIn.count)
    let tailS = tailIn.strides.map { Int(truncating: $0) }
    if tailS[3] == 1 && tailS[1] == T_SEQ {
        var xT = [Float](repeating: 0, count: D_MODEL * T_SEQ)
        x.withUnsafeBufferPointer { xp in
            xT.withUnsafeMutableBufferPointer { xtp in
                vDSP_mtrans(xp.baseAddress!, 1, xtp.baseAddress!, 1,
                            vDSP_Length(D_MODEL), vDSP_Length(T_SEQ))
            }
        }
        xT.withUnsafeMutableBufferPointer { xtp in
            var src = vImage_Buffer(
                data: UnsafeMutableRawPointer(xtp.baseAddress!),
                height: vImagePixelCount(D_MODEL),
                width: vImagePixelCount(T_SEQ),
                rowBytes: T_SEQ * MemoryLayout<Float>.stride)
            var dst = vImage_Buffer(
                data: UnsafeMutableRawPointer(tailP),
                height: vImagePixelCount(D_MODEL),
                width: vImagePixelCount(T_SEQ),
                rowBytes: T_SEQ * MemoryLayout<Float16>.stride)
            vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
        }
    } else {
        for d in 0..<D_MODEL {
            for t in 0..<T_SEQ {
                tailP[d * tailS[1] + t * tailS[3]] = Float16(x[t * D_MODEL + d])
            }
        }
    }
    let tailProv = try MLDictionaryFeatureProvider(dictionary: ["x_in": tailIn])
    let tailPred = try tailPack.prediction(from: tailProv)
    let logitsArr = tailPred.featureValue(for: "logits")!.multiArrayValue!
    let logP = logitsArr.dataPointer.bindMemory(to: Float16.self, capacity: logitsArr.count)
    let logS = logitsArr.strides.map { Int(truncating: $0) }
    // Unpack logits [1,33,1,T] fp16 → [T,33] fp32
    var logits = [Float](repeating: 0, count: T_SEQ * N_LABELS)
    for t in 0..<T_SEQ {
        for c in 0..<N_LABELS {
            logits[t * N_LABELS + c] = Float(logP[c * logS[1] + t * logS[3]])
        }
    }
    return logits
}

// ===========================================================================
// Run all sentences, validate vs golden
// ===========================================================================
print("\n[run] forward + validate \(N_SENT) sentences ...")
var allCos: [Double] = []
var allTop1: [Double] = []
var totalMs: Double = 0
// Collect Swift argmax labels for all sentences (for downstream span-F1 eval).
var swiftArgmax = [Int32](repeating: 0, count: N_SENT * T_SEQ)
for b in 0..<N_SENT {
    let ids  = inputIds[(b*T_SEQ)..<((b+1)*T_SEQ)]
    let mask = attnMask[(b*T_SEQ)..<((b+1)*T_SEQ)]
    let t0 = nowNs()
    let dump = (b == 0) && (ProcessInfo.processInfo.environment["PF_DUMP_STAGES"] != nil)
    let useGolden = (b == 0) && (ProcessInfo.processInfo.environment["PF_GOLDEN_TOPK"] != nil)
    let pred = try forwardSentence(ids: ids, mask: mask, dumpStages: dump, goldenTopK: useGolden)
    let dt = ms(t0, nowNs())
    totalMs += dt

    // Compare valid (mask>0) tokens only.
    var dot = 0.0, np = 0.0, ng = 0.0
    var top1ok = 0, top1n = 0
    for t in 0..<T_SEQ {
        if Array(mask)[t] <= 0 { continue }
        var pmax: Float = -.infinity, pi = 0
        var gmax: Float = -.infinity, gi = 0
        for k in 0..<N_LABELS {
            let pv = pred[t * N_LABELS + k]
            let gv = goldenF32[(b * T_SEQ + t) * N_LABELS + k]
            dot += Double(pv) * Double(gv); np += Double(pv*pv); ng += Double(gv*gv)
            if pv > pmax { pmax = pv; pi = k }
            if gv > gmax { gmax = gv; gi = k }
        }
        if pi == gi { top1ok += 1 }
        top1n += 1
        swiftArgmax[b * T_SEQ + t] = Int32(pi)
    }
    let cos = dot / (sqrt(np) * sqrt(ng) + 1e-30)
    let acc = Double(top1ok) / Double(max(top1n, 1))
    allCos.append(cos); allTop1.append(acc)
    print(String(format: "  s%d  cos=%.6f  top1=%.2f%%  (%.0f ms)",
                 b, cos, acc * 100, dt))
}

let cosMean = allCos.reduce(0, +) / Double(allCos.count)
let cosMin  = allCos.min()!
let top1Mean = allTop1.reduce(0, +) / Double(allTop1.count)
print(String(format: "\n[summary] cos mean=%.6f min=%.6f  top1 mean=%.2f%%  total=%.1f s",
             cosMean, cosMin, top1Mean * 100, totalMs / 1000))
print(String(format: "[summary] gates: cos>=0.95 %@   top1>=95%% %@",
             cosMin >= 0.95 ? "PASS" : "FAIL" as CVarArg,
             top1Mean >= 0.95 ? "PASS" : "FAIL" as CVarArg))

// Dump Swift argmax labels for span-F1 evaluation (Python).
do {
    let outPath = "\(REPO)/python/privacy/out/swift_dump/swift_argmax_labels.bin"
    let data = swiftArgmax.withUnsafeBufferPointer { Data(buffer: $0) }
    try? data.write(to: URL(fileURLWithPath: outPath))
    print("[dump] swift_argmax_labels.bin (\(N_SENT)x\(T_SEQ) i32) -> \(outPath)")
}

// ===========================================================================
// Energy-bench mode: sustain forward loop for fixed wall-clock duration so
// powermetrics has a steady-state window. Gated by PF_SUSTAIN_S env var.
// Emits BEGIN_AT / END_AT / ITERS lines compatible with ane_energy_parse.py.
// Writes nothing else; intended to be invoked under powermetrics capture.
// ===========================================================================
if let sustainStr = ProcessInfo.processInfo.environment["PF_SUSTAIN_S"],
   let sustainS = Double(sustainStr), sustainS > 0 {
    print("\n[sustain] running forward loop for \(sustainS) s ...")
    let begin = Date().timeIntervalSince1970
    print(String(format: "BEGIN_AT %.3f", begin))
    let deadline = Date().addingTimeInterval(sustainS)
    var iters = 0
    while Date() < deadline {
        // autoreleasepool every 32 iters: each forwardSentence allocates ~170
        // IOSurface-backed MLMultiArrays. Draining every 32 keeps peak at ~5K
        // surfaces, well under the kernel limit (~100K). Every-sentence drain
        // is needlessly eager; batching amortizes the pool push/pop overhead.
        if iters % 32 == 0 {
            try autoreleasepool {
                for _ in 0..<min(32, Int(deadline.timeIntervalSinceNow / 0.04) + 1) {
                    guard Date() < deadline else { break }
                    let b = iters % N_SENT
                    let ids  = inputIds[(b*T_SEQ)..<((b+1)*T_SEQ)]
                    let mask = attnMask[(b*T_SEQ)..<((b+1)*T_SEQ)]
                    _ = try forwardSentence(ids: ids, mask: mask, dumpStages: false, goldenTopK: false)
                    iters += 1
                }
            }
        }
    }
    let end = Date().timeIntervalSince1970
    print(String(format: "END_AT   %.3f", end))
    print("ITERS    \(iters)")
    print(String(format: "throughput %.1f sentences/s", Double(iters) / (end - begin)))
}
