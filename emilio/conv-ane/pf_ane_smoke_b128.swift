// pf_ane_smoke_b128.swift — Stepanov associativity test (BOOK_ANALYSIS Exp 19).
// Compare per-call latency of B=64 vs B=128 MoE packs.
// If B=128 ≈ B=64 → XPC round-trip dominates → re-batching is the right lever.
// If B=128 ≈ 2 × B=64 → bandwidth/compute dominates → re-batching is moot.
//
// Build: swiftc -O -framework CoreML -o pf_ane_smoke_b128 pf_ane_smoke_b128.swift
import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let MOE_64  = "\(DIR)/PF_packed_iverson_L0_N4_int8.mlmodelc"
let MOE_128 = "\(DIR)/PF_packed_iverson_v2_N4_B128_int8.mlmodelc"

let D_MODEL = 640
let N_EXPERTS = 128
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
    return s[lo] * (1.0 - (r - Double(lo))) + s[hi] * (r - Double(lo))
}

func loadOnANE(_ path: String) throws -> MLModel {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
}

func makeArr(_ shape: [Int], fill: Float16) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    let p = arr.dataPointer.bindMemory(to: Float16.self, capacity: arr.count)
    for i in 0..<arr.count { p[i] = fill }
    return arr
}

func benchMoE(label: String, path: String, batch: Int) throws -> [Double] {
    print("[smoke] B=\(batch)  loading: \(path.components(separatedBy: "/").last!)")
    let model = try loadOnANE(path)
    let xIn = try makeArr([batch, D_MODEL, 1, 1], fill: 0.01)
    let gIn = try makeArr([batch, N_EXPERTS, 1, 1], fill: 0)
    let gp = gIn.dataPointer.bindMemory(to: Float16.self, capacity: gIn.count)
    for b in 0..<batch {
        for k in 0..<4 { gp[b * N_EXPERTS + k] = Float16(0.25) }
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
    print("[\(label)] n=\(times.count)  median=\(String(format:"%.3f",m)) ms" +
          "  p25=\(String(format:"%.3f",pct(times,25)))" +
          "  p75=\(String(format:"%.3f",pct(times,75)))" +
          "  p99=\(String(format:"%.3f",pct(times,99)))" +
          "  min=\(String(format:"%.3f",times.min()!))" +
          "  max=\(String(format:"%.3f",times.max()!))" +
          "  ms/row=\(String(format:"%.4f", m / Double(batch)))")
    return times
}

do {
    let t64  = try benchMoE(label: "moe_b64 ", path: MOE_64,  batch: 64)
    let t128 = try benchMoE(label: "moe_b128", path: MOE_128, batch: 128)
    let m64  = median(t64)
    let m128 = median(t128)
    print("")
    print("[verdict] B=64  median = \(String(format:"%.2f",m64)) ms  (\(String(format:"%.4f", m64/64.0)) ms/row)")
    print("[verdict] B=128 median = \(String(format:"%.2f",m128)) ms  (\(String(format:"%.4f", m128/128.0)) ms/row)")
    let ratio = m128 / m64
    print("[verdict] B=128/B=64 ratio = \(String(format:"%.3f", ratio))×")
    if ratio < 1.3 {
        print("[verdict] STEPANOV WINS — XPC overhead dominates. Re-batching amortizes it.")
        print("[verdict] Predicted MoE-call savings: ~\(String(format:"%.0f", (1.0 - ratio/2.0)*100.0))% on 16-chunk loop.")
    } else if ratio < 1.7 {
        print("[verdict] PARTIAL WIN — some XPC amortization, but compute is climbing.")
    } else {
        print("[verdict] BANDWIDTH-LIMITED — re-batching is not the lever; need fusion/chaining.")
    }
} catch {
    print("ERROR: \(error)"); exit(1)
}
