// pf_ane_smoke_per_expert_real.swift — Option A with REAL routing.
// Loads topk_indices/topk_weights from pf_layer0_moe.npz (exported as bins).
// Bench: distinct experts only (skew-aware), seq + concurrent.
import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let DENSE = "\(DIR)/PF_packed_iverson_L0_N4_int8.mlmodelc"
let IDX_BIN = "\(DIR)/_pf_topk_idx_b64.bin"
let W_BIN   = "\(DIR)/_pf_topk_w_b64.bin"

let B = 64, K = 4, D_MODEL = 640, N_EXPERTS = 128
let WARMUP = 10, ITERS = 50

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted(); let n = s.count
    return n % 2 == 1 ? s[n/2] : 0.5 * (s[n/2 - 1] + s[n/2])
}
func loadOnANE(_ p: String) throws -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: URL(fileURLWithPath: p), configuration: cfg)
}
func makeF16(_ shape: [Int], fill: Float16) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count)
    for i in 0..<a.count { p[i] = fill }
    return a
}
func readBin<T>(_ path: String, _ count: Int, _ type: T.Type) throws -> [T] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: T.self).prefix(count)) }
}

// ---- Load real routing ----
let idxFlat: [Int32]   = try readBin(IDX_BIN, B*K, Int32.self)
let _wFlat:  [Float16] = try readBin(W_BIN,   B*K, Float16.self)
var topkIdx = [[Int32]](repeating: [], count: B)
for b in 0..<B { topkIdx[b] = Array(idxFlat[b*K..<(b+1)*K]) }
let distinctExperts = Set(idxFlat)
let counts = idxFlat.reduce(into: [:]) { c, e in c[e, default: 0] += 1 }
let maxHits = counts.values.max() ?? 0
print("[routing] B=\(B) K=\(K)  distinct experts=\(distinctExperts.count)  max hits/expert=\(maxHits)")

// ---- Load all 128 experts (we hold all so any routing works) ----
print("[load] loading 128 experts...")
let lt0 = DispatchTime.now().uptimeNanoseconds
var experts: [MLModel] = []
for e in 0..<N_EXPERTS {
    let p = "\(DIR)/PF_expert_L0_\(e)_B\(B)_fp16.mlmodelc"
    experts.append(try loadOnANE(p))
}
let loadMs = Double(DispatchTime.now().uptimeNanoseconds - lt0) / 1e6
print("[load] done in \(String(format: "%.0f", loadMs)) ms")

let x = try makeF16([B, D_MODEL, 1, 1], fill: Float16(0.01))
let xProv = try MLDictionaryFeatureProvider(dictionary: ["x_in": x])

let distinctList = Array(distinctExperts).sorted()

// Warmup all distinct experts
print("[warmup] firing each of the \(distinctList.count) distinct experts \(WARMUP)×...")
for _ in 0..<WARMUP {
    for e in distinctList { _ = try experts[Int(e)].prediction(from: xProv) }
}

// =========================================================================
// MODE A: distinct experts, sequential
// =========================================================================
var t1: [Double] = []
for _ in 0..<ITERS {
    let s = DispatchTime.now().uptimeNanoseconds
    for e in distinctList { _ = try experts[Int(e)].prediction(from: xProv) }
    t1.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
}
let m1 = median(t1)
print("[distinct_seq]  \(distinctList.count) calls: median=\(String(format: "%.2f", m1)) ms  per_call=\(String(format: "%.3f", m1/Double(distinctList.count)))")

// =========================================================================
// MODE B: distinct experts, concurrent
// =========================================================================
let q = DispatchQueue.global(qos: .userInitiated)
let group = DispatchGroup()
for _ in 0..<WARMUP {
    for e in distinctList {
        group.enter()
        q.async { _ = try? experts[Int(e)].prediction(from: xProv); group.leave() }
    }
    group.wait()
}
var t2: [Double] = []
for _ in 0..<ITERS {
    let s = DispatchTime.now().uptimeNanoseconds
    for e in distinctList {
        group.enter()
        q.async { _ = try? experts[Int(e)].prediction(from: xProv); group.leave() }
    }
    group.wait()
    t2.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
}
let m2 = median(t2)
print("[distinct_conc] \(distinctList.count) calls: median=\(String(format: "%.2f", m2)) ms")

// =========================================================================
// Reference: dense pack with same routing
// =========================================================================
let dense = try loadOnANE(DENSE)
let xd = try makeF16([B, D_MODEL, 1, 1], fill: Float16(0.01))
let gd = try makeF16([B, N_EXPERTS, 1, 1], fill: 0)
let gp = gd.dataPointer.bindMemory(to: Float16.self, capacity: gd.count)
for b in 0..<B { for k in 0..<K { gp[b * N_EXPERTS + Int(topkIdx[b][k])] = Float16(0.25) } }
let dProv = try MLDictionaryFeatureProvider(dictionary: ["x_in": xd, "g_in": gd])
for _ in 0..<WARMUP { _ = try dense.prediction(from: dProv) }
var td: [Double] = []
for _ in 0..<ITERS {
    let s = DispatchTime.now().uptimeNanoseconds
    _ = try dense.prediction(from: dProv)
    td.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
}
let mD = median(td)
print("[dense_pack]    median=\(String(format: "%.2f", mD)) ms")

// =========================================================================
print("")
print("=== Option A REAL ROUTING (M4 Max, B=\(B), \(distinctList.count)/\(N_EXPERTS) experts hit) ===")
print("  dense pack (status quo):     \(String(format: "%.2f", mD)) ms")
print("  per-expert seq:              \(String(format: "%.2f", m1)) ms  (\(String(format: "%.1f", mD/m1))×)")
print("  per-expert conc:             \(String(format: "%.2f", m2)) ms  (\(String(format: "%.1f", mD/m2))×)")
let best = min(m1, m2)
print("  BEST: \(String(format: "%.2f", best)) ms → \(String(format: "%.1f", mD/best))× faster than dense")
print("")
let chainBest = 8.0 * 16.0 * best + 8.0 * 8.0 * 0.6
let chainDense = 8.0 * 16.0 * mD + 8.0 * 8.0 * 0.6
print("Full B=8/T=128 forward projection:")
print("  dense:        \(String(format: "%.0f", chainDense)) ms")
print("  per-expert:   \(String(format: "%.0f", chainBest)) ms (\(String(format: "%.1f", chainDense/chainBest))× vs dense)")
print("  CPU baseline: 8955 ms → per-expert is \(String(format: "%.1f", 8955.0/chainBest))× faster")
