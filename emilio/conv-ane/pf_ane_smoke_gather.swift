// pf_ane_smoke_gather.swift — bench the M2 gather model.
// If gather + downstream-on-ANE total beats 75 ms, gather is viable.
import CoreML
import Foundation

let _BINDIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let DIR = "\(_BINDIR)/_gather_probe"
let M2 = "\(DIR)/M2_gather_reduce_combine.mlmodelc"
let MOE_128 = "\(_BINDIR)/PF_packed_iverson_L0_N4_int8.mlmodelc"
let MOE_4   = "\(_BINDIR)/PF_topk4_B64_int8.mlmodelc"
let B = 64, K = 4, D_FF = 640, N_EXPERTS = 128, D_MODEL = 640
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

func benchM2() throws -> Double {
    let model = try loadOnANE(M2)
    let idx = try makeI32([B, K], fill: 0)
    let ip = idx.dataPointer.bindMemory(to: Int32.self, capacity: idx.count)
    // realistic scattered indices
    for b in 0..<B {
        for k in 0..<K { ip[b * K + k] = Int32((b + k * 32) % N_EXPERTS) }
    }
    let gw = try makeF16([B, K], fill: Float16(0.25))
    let x  = try makeF16([B, D_FF], fill: Float16(0.01))
    let prov = try MLDictionaryFeatureProvider(dictionary: ["idx": idx, "gw": gw, "x": x])
    for _ in 0..<WARMUP { _ = try model.prediction(from: prov) }
    var t: [Double] = []
    for _ in 0..<ITERS {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: prov)
        t.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
    }
    let m = median(t)
    print("[M2 gather]   median=\(String(format:"%.3f", m)) ms  min=\(String(format:"%.3f", t.min()!)) max=\(String(format:"%.3f", t.max()!))")
    return m
}

func benchMoE(_ label: String, _ path: String, _ G: Int) throws -> Double {
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
    let dense  = try benchMoE("dense_128", MOE_128, 128)
    let topk   = try benchMoE("topk4_static", MOE_4, 4)
    let gather = try benchM2()
    print("")
    print("[verdict] dense 128-expert pack:           \(String(format:"%.2f", dense)) ms")
    print("[verdict] static top-4 ceiling:            \(String(format:"%.2f", topk)) ms")
    print("[verdict] M2 (gather + matmul + combine):  \(String(format:"%.2f", gather)) ms")
    if gather < dense {
        print("[verdict] GATHER WINS by \(String(format:"%.1f", dense/gather))× — viable architecture")
    } else {
        print("[verdict] GATHER LOSES — CPU gather + DMA dominates; use shard-dispatch instead")
    }
} catch { print("ERROR: \(error)"); exit(1) }
