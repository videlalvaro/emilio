// pf_chain_per_expert.swift
// End-to-end Swift bench for 8-layer per-expert MoE on ANE.
// Loads 8x128 expert .mlmodelcs, runs gather→dispatch→scatter for each layer.
// Validates final per-layer delta vs golden _pf_chain_L*_delta_b64.bin.
//
// Build: swiftc -O -framework CoreML -o pf_chain_per_expert pf_chain_per_expert.swift
// Run:   ./pf_chain_per_expert
import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let B = 64, K = 4, D_MODEL = 640, N_EXPERTS = 128, N_LAYERS = 8
let WARMUP = 5, ITERS = 30

func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
func ms(_ a: UInt64, _ b: UInt64) -> Double { Double(b - a) / 1e6 }
func median(_ xs: [Double]) -> Double {
    let s = xs.sorted(); let n = s.count
    return n % 2 == 1 ? s[n/2] : 0.5 * (s[n/2 - 1] + s[n/2])
}
func loadOnANE(_ p: String) throws -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: URL(fileURLWithPath: p), configuration: cfg)
}
func makeF16(_ shape: [Int], fill: Float16 = 0) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count)
    for i in 0..<a.count { p[i] = fill }
    return a
}
func readBinF16(_ path: String, _ count: Int) throws -> [Float16] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self).prefix(count)) }
}
func readBinI32(_ path: String, _ count: Int) throws -> [Int32] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self).prefix(count)) }
}

// ---------------------------------------------------------------------------
// Load 8 layers x 128 experts
// ---------------------------------------------------------------------------
print("[load] loading \(N_LAYERS)x\(N_EXPERTS) experts on ANE ...")
let lt0 = nowNs()
var experts: [[MLModel]] = []  // experts[layer][eid]
for L in 0..<N_LAYERS {
    var layer: [MLModel] = []
    for e in 0..<N_EXPERTS {
        let p = "\(DIR)/PF_expert_L\(L)_\(e)_B\(B)_fp16.mlmodelc"
        layer.append(try loadOnANE(p))
    }
    experts.append(layer)
    let dt = ms(lt0, nowNs())
    print(String(format: "  L%d loaded (%.1fs elapsed)", L, dt/1000))
}
let loadMs = ms(lt0, nowNs())
print(String(format: "[load] all %d models loaded in %.1fs", N_LAYERS*N_EXPERTS, loadMs/1000))

// ---------------------------------------------------------------------------
// Load per-layer routing + golden delta
// ---------------------------------------------------------------------------
struct LayerData {
    let norm: [Float16]    // B*D
    let idx:  [Int32]      // B*K
    let w:    [Float16]    // B*K
    let delta:[Float16]    // B*D (golden)
    let distinct: [Int]
    let perExpertRows: [Int: [(Int, Float)]]   // eid -> [(row, weight)]
}

var layers: [LayerData] = []
for L in 0..<N_LAYERS {
    let norm  = try readBinF16("\(DIR)/_pf_chain_L\(L)_norm_b64.bin",  B*D_MODEL)
    let idx   = try readBinI32("\(DIR)/_pf_chain_L\(L)_idx_b64.bin",   B*K)
    let w     = try readBinF16("\(DIR)/_pf_chain_L\(L)_w_b64.bin",     B*K)
    let delta = try readBinF16("\(DIR)/_pf_chain_L\(L)_delta_b64.bin", B*D_MODEL)
    var byExpert: [Int: [(Int, Float)]] = [:]
    for r in 0..<B {
        for k in 0..<K {
            let eid = Int(idx[r*K + k])
            let wv = Float(w[r*K + k])
            if wv == 0 { continue }
            byExpert[eid, default: []].append((r, wv))
        }
    }
    let distinct = Array(byExpert.keys).sorted()
    layers.append(LayerData(norm: norm, idx: idx, w: w, delta: delta,
                            distinct: distinct, perExpertRows: byExpert))
    print("  L\(L): distinct experts=\(distinct.count)")
}

// ---------------------------------------------------------------------------
// One-layer per-expert MoE step (concurrent dispatch). Returns delta (B*D fp32).
// ---------------------------------------------------------------------------
let q = DispatchQueue.global(qos: .userInitiated)

func runLayer(_ L: Int) throws -> [Float] {
    let ld = layers[L]
    let layerExperts = experts[L]

    // Build per-expert padded inputs and dispatch concurrently
    let group = DispatchGroup()
    let lock = NSLock()
    var results: [(eid: Int, rows: [(Int, Float)], outBuf: UnsafeMutablePointer<Float16>)] = []

    for eid in ld.distinct {
        let rows = ld.perExpertRows[eid]!
        let xin = try makeF16([B, D_MODEL, 1, 1], fill: 0)
        let xp = xin.dataPointer.bindMemory(to: Float16.self, capacity: xin.count)
        for (slot, (r, _)) in rows.enumerated() {
            // copy norm[r, :] to slot row
            let src = r * D_MODEL
            let dst = slot * D_MODEL
            for d in 0..<D_MODEL { xp[dst + d] = ld.norm[src + d] }
        }
        let prov = try MLDictionaryFeatureProvider(dictionary: ["x_in": xin])
        let model = layerExperts[eid]
        let outBuf = UnsafeMutablePointer<Float16>.allocate(capacity: rows.count * D_MODEL)

        group.enter()
        q.async {
            do {
                let pred = try model.prediction(from: prov)
                let yArr = pred.featureValue(for: "y_out")!.multiArrayValue!
                let yp = yArr.dataPointer.bindMemory(to: Float16.self, capacity: yArr.count)
                // Output stride is typically [20480, 32, 32, 1] (ANE-padded), so
                // element [slot, d, 0, 0] is at slot*strides[0] + d*strides[1].
                let s0 = Int(truncating: yArr.strides[0])
                let s1 = Int(truncating: yArr.strides[1])
                for slot in 0..<rows.count {
                    let dst = slot * D_MODEL
                    let src = slot * s0
                    for d in 0..<D_MODEL { outBuf[dst + d] = yp[src + d * s1] }
                }
                lock.lock()
                results.append((eid, rows, outBuf))
                lock.unlock()
            } catch {
                fatalError("predict failed for L\(L) e\(eid): \(error)")
            }
            group.leave()
        }
    }
    group.wait()

    // Scatter-sum weighted contributions.
    // Note: stored topk_w = softmax/TOPK (sums to 0.25). Golden mlp_delta uses
    // raw softmax weights (sum=1.0), so multiply by TOPK on accumulation.
    var delta = [Float](repeating: 0, count: B * D_MODEL)
    let kScale = Float(K)
    for r in results {
        for (slot, (row, wv)) in r.rows.enumerated() {
            let src = slot * D_MODEL
            let dst = row * D_MODEL
            let s = wv * kScale
            for d in 0..<D_MODEL {
                delta[dst + d] += s * Float(r.outBuf[src + d])
            }
        }
        r.outBuf.deallocate()
    }
    return delta
}

// ---------------------------------------------------------------------------
// Validate: run once, compare to golden
// ---------------------------------------------------------------------------
print("\n[validate] running each layer once vs golden delta ...")
for L in 0..<N_LAYERS {
    let pred = try runLayer(L)
    let g = layers[L].delta
    var dot: Double = 0; var np: Double = 0; var ng: Double = 0
    for i in 0..<(B * D_MODEL) {
        let p = Double(pred[i]); let gv = Double(g[i])
        dot += p * gv; np += p * p; ng += gv * gv
    }
    let cos = dot / (sqrt(np) * sqrt(ng) + 1e-30)
    let status = cos >= 0.97 ? "PASS" : "FAIL"
    print(String(format: "  L%d cos=%.6f  ||pred||=%.1f  ||gold||=%.1f  [%@]",
                 L, cos, sqrt(np), sqrt(ng), status as CVarArg))
}

// ---------------------------------------------------------------------------
// Bench: full 8-layer chain wall-time (per-expert)
// ---------------------------------------------------------------------------
print("\n[bench] warmup \(WARMUP)× full chain ...")
for _ in 0..<WARMUP { for L in 0..<N_LAYERS { _ = try runLayer(L) } }

print("[bench] timing \(ITERS)× full \(N_LAYERS)-layer chain ...")
var chainTimes: [Double] = []
var perLayer: [[Double]] = Array(repeating: [], count: N_LAYERS)
for _ in 0..<ITERS {
    let t0 = nowNs()
    for L in 0..<N_LAYERS {
        let s = nowNs()
        _ = try runLayer(L)
        perLayer[L].append(ms(s, nowNs()))
    }
    chainTimes.append(ms(t0, nowNs()))
}
let chainMed = median(chainTimes)
print("\n=== Per-expert MoE chain (M4 Max, 8 layers × per-expert) ===")
for L in 0..<N_LAYERS {
    let m = median(perLayer[L])
    print(String(format: "  L%d: %.2f ms (distinct=%d)", L, m, layers[L].distinct.count))
}
print(String(format: "\n  total chain median: %.2f ms", chainMed))
print(String(format: "  vs Python full-chain wall (8.1s): %.0f× faster (MoE-only path)", 8100.0 / chainMed))
print(String(format: "  vs dense-pack chain (~610 ms = 8 × 76 ms): %.1f× faster", 610.0 / chainMed))
