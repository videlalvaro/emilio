// pf_cache_probe.swift
// Empirical probe: how does per-layer wall time scale with total mlmodelc
// working-set size (number of distinct experts resident in driver cache)?
//
// For each cache_size in {1, 2, 4, 8} layers (= 128, 256, 512, 1024 experts),
// load that many layers, then bench L0 only (always the same input).
// Holds # of distinct experts hit per call constant (always L0's 32 experts).
//
// Build: swiftc -O -framework CoreML -o pf_cache_probe pf_cache_probe.swift

import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let B = 64, K = 4, D_MODEL = 640, N_EXPERTS = 128
let TARGET_LAYER = 0
let WARMUP = 10, ITERS = 50

func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
func ms(_ a: UInt64, _ b: UInt64) -> Double { Double(b - a) / 1e6 }
func median(_ xs: [Double]) -> Double {
    let s = xs.sorted(); let n = s.count
    return n % 2 == 1 ? s[n/2] : 0.5 * (s[n/2 - 1] + s[n/2])
}
func p95(_ xs: [Double]) -> Double {
    let s = xs.sorted(); return s[Int(Double(s.count - 1) * 0.95)]
}
func loadOnANE(_ p: String) throws -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: URL(fileURLWithPath: p), configuration: cfg)
}
func readBinF16(_ path: String, _ count: Int) throws -> [Float16] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self).prefix(count)) }
}
func readBinI32(_ path: String, _ count: Int) throws -> [Int32] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self).prefix(count)) }
}

// Load L0 routing (always the experts we'll actually call)
let norm  = try readBinF16("\(DIR)/_pf_chain_L\(TARGET_LAYER)_norm_b64.bin",  B*D_MODEL)
let idx   = try readBinI32("\(DIR)/_pf_chain_L\(TARGET_LAYER)_idx_b64.bin",   B*K)
let w     = try readBinF16("\(DIR)/_pf_chain_L\(TARGET_LAYER)_w_b64.bin",     B*K)

var byExpert: [Int: [(Int, Float)]] = [:]
for r in 0..<B {
    for k in 0..<K {
        let eid = Int(idx[r*K + k]); let wv = Float(w[r*K + k])
        if wv == 0 { continue }
        byExpert[eid, default: []].append((r, wv))
    }
}
let distinct = Array(byExpert.keys).sorted()
print("[probe] target layer L\(TARGET_LAYER), distinct experts hit per call: \(distinct.count)")

let q = DispatchQueue.global(qos: .userInitiated)

func benchHotLayer(_ targetExperts: [MLModel]) throws -> (med: Double, p95: Double) {
    func step() throws {
        let group = DispatchGroup()
        var bufs: [UnsafeMutablePointer<Float16>] = []
        var keepAlive: [(MLDictionaryFeatureProvider)] = []
        for eid in distinct {
            let rows = byExpert[eid]!
            let xin = try MLMultiArray(shape: [NSNumber(value: B), NSNumber(value: D_MODEL),
                                               1, 1], dataType: .float16)
            let xp = xin.dataPointer.bindMemory(to: Float16.self, capacity: xin.count)
            memset(xp, 0, xin.count * MemoryLayout<Float16>.stride)
            for (slot, (r, _)) in rows.enumerated() {
                let src = r * D_MODEL; let dst = slot * D_MODEL
                for d in 0..<D_MODEL { xp[dst + d] = norm[src + d] }
            }
            let prov = try MLDictionaryFeatureProvider(dictionary: ["x_in": xin])
            keepAlive.append(prov)
            let outBuf = UnsafeMutablePointer<Float16>.allocate(capacity: rows.count * D_MODEL)
            bufs.append(outBuf)
            let m = targetExperts[eid]
            let rowCount = rows.count
            group.enter()
            q.async {
                do {
                    let pred = try m.prediction(from: prov)
                    let yArr = pred.featureValue(for: "y_out")!.multiArrayValue!
                    let yp = yArr.dataPointer.bindMemory(to: Float16.self, capacity: yArr.count)
                    let s0 = Int(truncating: yArr.strides[0])
                    let s1 = Int(truncating: yArr.strides[1])
                    for slot in 0..<rowCount {
                        let dst = slot * D_MODEL; let src = slot * s0
                        for d in 0..<D_MODEL { outBuf[dst + d] = yp[src + d * s1] }
                    }
                } catch { fatalError("predict: \(error)") }
                group.leave()
            }
        }
        group.wait()
        for b in bufs { b.deallocate() }
        _ = keepAlive.count  // silence
    }
    for _ in 0..<WARMUP { try step() }
    var times: [Double] = []
    for _ in 0..<ITERS {
        let t0 = nowNs(); try step(); times.append(ms(t0, nowNs()))
    }
    return (median(times), p95(times))
}

// Probe: progressively load more layers' worth of experts to inflate the
// driver's working-set, but always benchmark the same L0 experts.
print("\n[probe] N_layers_loaded  total_resident  L0_med_ms  L0_p95_ms")
print("        ---------------  --------------  ---------  ---------")

let cacheSizes = [1, 2, 4, 8]
var allLoaded: [[MLModel]] = []
for L in 0..<8 {
    var arr: [MLModel] = []
    for e in 0..<N_EXPERTS {
        arr.append(try loadOnANE("\(DIR)/PF_expert_L\(L)_\(e)_B\(B)_fp16.mlmodelc"))
    }
    allLoaded.append(arr)
    if cacheSizes.contains(L + 1) {
        let resident = (L + 1) * N_EXPERTS
        let (med, q95) = try benchHotLayer(allLoaded[TARGET_LAYER])
        print(String(format: "        %15d  %14d  %9.3f  %9.3f", L+1, resident, med, q95))
    }
}
