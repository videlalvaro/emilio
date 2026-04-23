// Per-expert single-call probe: time one expert .mlmodelc, simulate top-4
// dispatch via 4 calls to the same expert (worst case: cold weights every
// call), and concurrent dispatch via DispatchGroup.
import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let EXPERT = "\(DIR)/PF_expert_0_B64_fp16.mlmodelc"
let DENSE  = "\(DIR)/PF_packed_iverson_L0_N4_int8.mlmodelc"
let TOPK4  = "\(DIR)/PF_topk4_B64_int8.mlmodelc"

let B = 64, D_MODEL = 640
let WARMUP = 20, ITERS = 200

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

let model = try loadOnANE(EXPERT)
let x = try makeF16([B, D_MODEL, 1, 1], fill: Float16(0.01))
let prov = try MLDictionaryFeatureProvider(dictionary: ["x_in": x])

// ---- 1. Single expert call ----
for _ in 0..<WARMUP { _ = try model.prediction(from: prov) }
var t1: [Double] = []
for _ in 0..<ITERS {
    let t0 = DispatchTime.now().uptimeNanoseconds
    _ = try model.prediction(from: prov)
    t1.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
}
let single = median(t1)
print("[single_expert]   median=\(String(format:"%.3f", single)) ms  min=\(String(format:"%.3f", t1.min()!)) max=\(String(format:"%.3f", t1.max()!))")

// ---- 2. Four sequential calls to same model (proxy for top-4 if cached) ----
for _ in 0..<WARMUP { for _ in 0..<4 { _ = try model.prediction(from: prov) } }
var t4: [Double] = []
for _ in 0..<ITERS {
    let t0 = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<4 { _ = try model.prediction(from: prov) }
    t4.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
}
let seq4 = median(t4)
print("[seq_4_calls]     median=\(String(format:"%.3f", seq4)) ms  per_call=\(String(format:"%.3f", seq4/4)) ms")

// ---- 3. Concurrent 4 calls via DispatchGroup ----
let q = DispatchQueue.global(qos: .userInitiated)
let group = DispatchGroup()
for _ in 0..<WARMUP {
    for _ in 0..<4 {
        group.enter()
        q.async { _ = try? model.prediction(from: prov); group.leave() }
    }
    group.wait()
}
var tc: [Double] = []
for _ in 0..<ITERS {
    let t0 = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<4 {
        group.enter()
        q.async { _ = try? model.prediction(from: prov); group.leave() }
    }
    group.wait()
    tc.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
}
let conc4 = median(tc)
print("[conc_4_calls]    median=\(String(format:"%.3f", conc4)) ms  per_call=\(String(format:"%.3f", conc4/4)) ms")

// ---- Compare ----
let dense = 76.5  // measured earlier
let topk  = 1.6
print("")
print("=== verdict (Option A feasibility) ===")
print("  dense 128-expert pack:           \(String(format:"%.2f", dense)) ms (status quo)")
print("  static top-4 ceiling:            \(String(format:"%.2f", topk)) ms (silicon floor)")
print("  ONE expert call:                 \(String(format:"%.2f", single)) ms")
print("  4 sequential expert calls:       \(String(format:"%.2f", seq4)) ms")
print("  4 concurrent expert calls:       \(String(format:"%.2f", conc4)) ms")
let best = min(seq4, conc4)
print("  best vs dense:                   \(String(format:"%.1f", dense/best))×")
print("  vs floor:                        \(String(format:"%.2f", best - topk)) ms gap")
print("")
if best < dense {
    print("  → Option A VIABLE. Proceed to build all 128 experts.")
} else {
    print("  → Option A NOT BETTER THAN DENSE. Per-call XPC dominates. Need _ANEClient.")
}
