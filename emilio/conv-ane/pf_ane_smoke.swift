// pf_ane_smoke.swift — measure raw ANE single-call latency for one PF layer.
// No Python, no opf, no full forward — just: how fast does ANE answer one call?
//
// Build:
//   swiftc -O -framework CoreML -o pf_ane_smoke pf_ane_smoke.swift
//
// Run:
//   ./pf_ane_smoke

import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let ATTN_PATH = "\(DIR)/PF_attn0_T128.mlmodelc"
let MOE_PATH  = "\(DIR)/PF_packed_iverson_L0_N4_int8.mlmodelc"

let D_MODEL = 640
let T_SEQ   = 128
let N_EXPERTS = 128
let MOE_BATCH = 64
let WARMUP = 10
let ITERS  = 100

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted(); let n = s.count
    return n % 2 == 1 ? s[n/2] : 0.5 * (s[n/2 - 1] + s[n/2])
}
func pct(_ xs: [Double], _ p: Double) -> Double {
    let s = xs.sorted()
    let r = max(0.0, min(Double(s.count - 1), p / 100.0 * Double(s.count - 1)))
    let lo = Int(r.rounded(.down)); let hi = Int(r.rounded(.up))
    let f = r - Double(lo)
    return s[lo] * (1.0 - f) + s[hi] * f
}

func loadOnANE(_ path: String) throws -> MLModel {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
}

func makeArr(_ shape: [Int], fill: Float16 = 0) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    let p = arr.dataPointer.bindMemory(to: Float16.self, capacity: arr.count)
    for i in 0..<arr.count { p[i] = fill }
    return arr
}

func benchAttn() throws -> [Double] {
    print("[smoke] loading attn: \(ATTN_PATH)")
    let model = try loadOnANE(ATTN_PATH)
    let xIn  = try makeArr([1, D_MODEL, 1, T_SEQ], fill: 0.01)
    let pad  = try makeArr([1, 1, 1, T_SEQ], fill: 0)
    let inputs: [String: Any] = ["x_in": xIn, "pad_add": pad]
    let provider = try MLDictionaryFeatureProvider(dictionary: inputs)

    print("[smoke] warmup attn (\(WARMUP))...")
    for _ in 0..<WARMUP { _ = try model.prediction(from: provider) }
    print("[smoke] timed  attn (\(ITERS))...")
    var times: [Double] = []
    for _ in 0..<ITERS {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: provider)
        let dt = DispatchTime.now().uptimeNanoseconds - t0
        times.append(Double(dt) / 1e6)
    }
    return times
}

func benchMoE() throws -> [Double] {
    print("[smoke] loading moe: \(MOE_PATH)")
    let model = try loadOnANE(MOE_PATH)
    let xIn = try makeArr([MOE_BATCH, D_MODEL, 1, 1], fill: 0.01)
    let gIn = try makeArr([MOE_BATCH, N_EXPERTS, 1, 1], fill: 0)
    // Set 4 active experts per row to ~0.25.
    let gp = gIn.dataPointer.bindMemory(to: Float16.self, capacity: gIn.count)
    for b in 0..<MOE_BATCH {
        for k in 0..<4 {
            gp[b * N_EXPERTS + k] = Float16(0.25)
        }
    }
    let inputs: [String: Any] = ["x_in": xIn, "g_in": gIn]
    let provider = try MLDictionaryFeatureProvider(dictionary: inputs)

    print("[smoke] warmup moe (\(WARMUP))...")
    for _ in 0..<WARMUP { _ = try model.prediction(from: provider) }
    print("[smoke] timed  moe (\(ITERS))...")
    var times: [Double] = []
    for _ in 0..<ITERS {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: provider)
        let dt = DispatchTime.now().uptimeNanoseconds - t0
        times.append(Double(dt) / 1e6)
    }
    return times
}

func report(_ label: String, _ ts: [Double]) {
    print("[\(label)] n=\(ts.count)  median=\(String(format: "%.3f", median(ts))) ms" +
          "  p25=\(String(format: "%.3f", pct(ts, 25)))" +
          "  p75=\(String(format: "%.3f", pct(ts, 75)))" +
          "  p99=\(String(format: "%.3f", pct(ts, 99)))" +
          "  min=\(String(format: "%.3f", ts.min() ?? 0))" +
          "  max=\(String(format: "%.3f", ts.max() ?? 0))")
}

do {
    let aTs = try benchAttn()
    report("attn", aTs)
    let mTs = try benchMoE()
    report("moe ", mTs)

    // Predicted full-chain wall: 8 sentences × 8 attn calls + 16 moe-chunks × 8 moe calls
    // (matches what the Python harness does, B=8 T=128 valid_tokens=151 batch=8).
    let mAttn = median(aTs)
    let mMoE  = median(mTs)
    let fullChain = 8.0 * 8.0 * mAttn + 16.0 * 8.0 * mMoE
    print("[predict] median full B=8/T=128 forward (Swift): \(String(format: "%.1f", fullChain)) ms")
    print("[predict] => Python harness was \(String(format: "%.1f", 20210.0 / fullChain))× slower than Swift floor")
} catch {
    print("ERROR: \(error)")
    exit(1)
}
