// pf_ane_smoke_gather_real.swift — bench the real gather MoE pack.
import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let GATHER  = "\(DIR)/PF_gather_moe_B64_fp16.mlmodelc"
let GATHER8 = "\(DIR)/PF_gather_moe_B64_int8.mlmodelc"
let DENSE   = "\(DIR)/PF_packed_iverson_L0_N4_int8.mlmodelc"
let TOPK4   = "\(DIR)/PF_topk4_B64_int8.mlmodelc"

let B = 64, K = 4, D_MODEL = 640, N_EXPERTS = 128
let WARMUP = 10, ITERS = 100

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
func makeI32(_ shape: [Int], fill: Int32) throws -> MLMultiArray {
    let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
    let p = a.dataPointer.bindMemory(to: Int32.self, capacity: a.count)
    for i in 0..<a.count { p[i] = fill }
    return a
}

func benchGather(_ path: String, _ label: String) throws -> Double {
    let model = try loadOnANE(path)
    let x  = try makeF16([B, D_MODEL], fill: Float16(0.01))
    let idx = try makeI32([B, K], fill: 0)
    let ip = idx.dataPointer.bindMemory(to: Int32.self, capacity: idx.count)
    // realistic scattered indices
    for b in 0..<B {
        for k in 0..<K {
            ip[b * K + k] = Int32((b * 13 + k * 31 + 7) % N_EXPERTS)
        }
    }
    let tw = try makeF16([B, K], fill: Float16(0.25))
    let prov = try MLDictionaryFeatureProvider(dictionary: ["x_in": x, "idx": idx, "topk_w": tw])
    for _ in 0..<WARMUP { _ = try model.prediction(from: prov) }
    var t: [Double] = []
    for _ in 0..<ITERS {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: prov)
        t.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
    }
    let m = median(t)
    print("[\(label)] median=\(String(format:"%.3f", m)) ms  min=\(String(format:"%.3f", t.min()!)) max=\(String(format:"%.3f", t.max()!))  p99=\(String(format:"%.3f", t.sorted()[Int(0.99 * Double(t.count))]))")
    return m
}
func benchPack(_ label: String, _ path: String, _ G: Int) throws -> Double {
    let m = try loadOnANE(path)
    let x = try makeF16([B, D_MODEL, 1, 1], fill: Float16(0.01))
    let g = try makeF16([B, G, 1, 1], fill: 0)
    let gp = g.dataPointer.bindMemory(to: Float16.self, capacity: g.count)
    for b in 0..<B { for k in 0..<min(4,G) { gp[b * G + k] = Float16(0.25) } }
    let prov = try MLDictionaryFeatureProvider(dictionary: ["x_in": x, "g_in": g])
    for _ in 0..<WARMUP { _ = try m.prediction(from: prov) }
    var t: [Double] = []
    for _ in 0..<ITERS {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try m.prediction(from: prov)
        t.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
    }
    let med = median(t)
    print("[\(label)]  median=\(String(format:"%.3f", med)) ms  min=\(String(format:"%.3f", t.min()!)) max=\(String(format:"%.3f", t.max()!))")
    return med
}

do {
    let dense  = try benchPack("dense_128 ", DENSE, 128)
    let topk   = try benchPack("topk4_stat", TOPK4, 4)
    let gather   = try benchGather(GATHER,  "gather_fp16")
    let gather8  = try benchGather(GATHER8, "gather_int8")
    print("")
    print("=== verdict ===")
    print("  dense 128-expert pack:           \(String(format:"%.2f", dense)) ms  (status quo)")
    print("  static top-4 ceiling:            \(String(format:"%.2f", topk)) ms  (silicon floor)")
    print("  REAL gather MoE fp16:            \(String(format:"%.2f", gather)) ms")
    print("  REAL gather MoE int8:            \(String(format:"%.2f", gather8)) ms")
    let best = min(gather, gather8)
    print("  best speedup vs dense:           \(String(format:"%.1f", dense/best))×")
    print("  gap to silicon floor:            \(String(format:"%.2f", best - topk)) ms (CPU gather + DMA)")

    // Full forward projection: 8 layers × 16 chunks × MoE call + 8 layers × 8 sentences × attn (~0.6ms)
    let fullChainGather = 8.0 * 16.0 * best + 8.0 * 8.0 * 0.6
    let fullChainDense  = 8.0 * 16.0 * dense  + 8.0 * 8.0 * 0.6
    print("")
    print("Full B=8/T=128 forward projection:")
    print("  with dense MoE:  \(String(format:"%.0f", fullChainDense)) ms")
    print("  with gather MoE: \(String(format:"%.0f", fullChainGather)) ms  (\(String(format:"%.1f", fullChainDense/fullChainGather))× faster)")
    print("  CPU baseline (measured): 8955 ms")
    print("  CPU/gather speedup: \(String(format:"%.1f", 8955.0/fullChainGather))×")
} catch { print("ERROR: \(error)"); exit(1) }
