// pf_ane_smoke_topk4.swift — compute-ceiling probe.
// Measure latency of a top-4-only MoE pack vs the 128-expert dense pack.
import CoreML
import Foundation
let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let MOE_128 = "\(DIR)/PF_packed_iverson_L0_N4_int8.mlmodelc"
let MOE_4   = "\(DIR)/PF_topk4_B64_int8.mlmodelc"
let D_MODEL = 640, BATCH = 64, WARMUP = 10, ITERS = 100

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted(); let n = s.count
    return n % 2 == 1 ? s[n/2] : 0.5 * (s[n/2 - 1] + s[n/2])
}
func loadOnANE(_ path: String) throws -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
}
func makeArr(_ shape: [Int], fill: Float16) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    let p = arr.dataPointer.bindMemory(to: Float16.self, capacity: arr.count)
    for i in 0..<arr.count { p[i] = fill }
    return arr
}
func bench(label: String, path: String, gExperts: Int) throws -> Double {
    let model = try loadOnANE(path)
    let xIn = try makeArr([BATCH, D_MODEL, 1, 1], fill: 0.01)
    let gIn = try makeArr([BATCH, gExperts, 1, 1], fill: 0)
    let gp = gIn.dataPointer.bindMemory(to: Float16.self, capacity: gIn.count)
    for b in 0..<BATCH {
        for k in 0..<min(4, gExperts) { gp[b * gExperts + k] = Float16(0.25) }
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: ["x_in": xIn, "g_in": gIn])
    for _ in 0..<WARMUP { _ = try model.prediction(from: provider) }
    var times: [Double] = []
    for _ in 0..<ITERS {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: provider)
        times.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
    }
    let m = median(times)
    print("[\(label)] G=\(gExperts) median=\(String(format:"%.3f", m)) ms  min=\(String(format:"%.3f", times.min()!)) max=\(String(format:"%.3f", times.max()!))")
    return m
}
do {
    let m128 = try bench(label: "moe_128", path: MOE_128, gExperts: 128)
    let m4   = try bench(label: "moe_4  ", path: MOE_4,   gExperts: 4)
    print("")
    print("[verdict] 128-expert pack: \(String(format:"%.2f", m128)) ms")
    print("[verdict]   4-expert pack: \(String(format:"%.2f", m4)) ms")
    print("[verdict] speedup: \(String(format:"%.1f", m128/m4))×  (theoretical max: 32×)")
    let floor = m4
    let computeFloor = m128 / 32.0
    print("[verdict] compute floor (linear scaling): \(String(format:"%.2f", computeFloor)) ms")
    print("[verdict] fixed overhead (load+layout+DMA): \(String(format:"%.2f", floor - computeFloor)) ms")
    if m4 < 5.0 {
        print("[verdict] WIN — gather-then-compute is the lever. Worth solving the dynamic-gather problem.")
    } else if m4 < 20.0 {
        print("[verdict] PARTIAL — fixed overhead exists but gather still pays.")
    } else {
        print("[verdict] DEAD END — fixed-cost floor too high; gather won't help.")
    }
} catch { print("ERROR: \(error)"); exit(1) }
