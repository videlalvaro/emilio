// pf_chain_pipelined.swift
// Variant A: pre-pack ALL 8 layers' per-expert inputs ONCE, then time only
// the dispatch+scatter loops. This isolates the pure-ANE chain time and
// tells us the absolute floor for the per-expert MoE chain on M4 Max.
//
// Build: swiftc -O -framework CoreML -o pf_chain_pipelined pf_chain_pipelined.swift

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
func readBinF16(_ path: String, _ count: Int) throws -> [Float16] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self).prefix(count)) }
}
func readBinI32(_ path: String, _ count: Int) throws -> [Int32] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self).prefix(count)) }
}

print("[load] loading \(N_LAYERS)x\(N_EXPERTS) experts on ANE ...")
let lt0 = nowNs()
var experts: [[MLModel]] = []
for L in 0..<N_LAYERS {
    var layer: [MLModel] = []
    for e in 0..<N_EXPERTS {
        layer.append(try loadOnANE("\(DIR)/PF_expert_L\(L)_\(e)_B\(B)_fp16.mlmodelc"))
    }
    experts.append(layer)
}
print(String(format: "[load] done in %.1fs", ms(lt0, nowNs())/1000))

// ---- Build per-layer dispatch packets up-front (out of the timed region) ----
struct Packet {
    let model: MLModel
    let xin:   MLMultiArray            // retained input (kept alive)
    let prov:  MLDictionaryFeatureProvider
    let rows:  [(Int, Float)]          // (target row, weight) per slot
}

var packets: [[Packet]] = Array(repeating: [], count: N_LAYERS)
for L in 0..<N_LAYERS {
    let norm  = try readBinF16("\(DIR)/_pf_chain_L\(L)_norm_b64.bin",  B*D_MODEL)
    let idx   = try readBinI32("\(DIR)/_pf_chain_L\(L)_idx_b64.bin",   B*K)
    let w     = try readBinF16("\(DIR)/_pf_chain_L\(L)_w_b64.bin",     B*K)
    var byExpert: [Int: [(Int, Float)]] = [:]
    for r in 0..<B {
        for k in 0..<K {
            let eid = Int(idx[r*K + k]); let wv = Float(w[r*K + k])
            if wv == 0 { continue }
            byExpert[eid, default: []].append((r, wv))
        }
    }
    for eid in byExpert.keys.sorted() {
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
        packets[L].append(Packet(model: experts[L][eid], xin: xin, prov: prov, rows: rows))
    }
    print("  L\(L): \(packets[L].count) packets")
}

// Per-layer scratch (reused across iters to avoid alloc churn).
let q = DispatchQueue.global(qos: .userInitiated)

func runLayerPrebuilt(_ L: Int) -> [Float] {
    let pkts = packets[L]
    let kScale = Float(K)
    let group = DispatchGroup()

    // Allocate per-packet output buffers (small; one per distinct expert)
    var outBufs: [UnsafeMutablePointer<Float16>] = []
    outBufs.reserveCapacity(pkts.count)
    for p in pkts {
        outBufs.append(UnsafeMutablePointer<Float16>.allocate(capacity: p.rows.count * D_MODEL))
    }

    for (i, p) in pkts.enumerated() {
        let m = p.model; let prov = p.prov
        let rowCount = p.rows.count
        let outBuf = outBufs[i]
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
            } catch { fatalError("predict L\(L): \(error)") }
            group.leave()
        }
    }
    group.wait()

    var delta = [Float](repeating: 0, count: B * D_MODEL)
    for (i, p) in pkts.enumerated() {
        let outBuf = outBufs[i]
        for (slot, (row, wv)) in p.rows.enumerated() {
            let src = slot * D_MODEL; let dst = row * D_MODEL
            let s = wv * kScale
            for d in 0..<D_MODEL { delta[dst + d] += s * Float(outBuf[src + d]) }
        }
        outBuf.deallocate()
    }
    return delta
}

print("\n[validate] vs golden ...")
for L in 0..<N_LAYERS {
    let pred = runLayerPrebuilt(L)
    let g = try readBinF16("\(DIR)/_pf_chain_L\(L)_delta_b64.bin", B*D_MODEL)
    var dot = 0.0, np = 0.0, ng = 0.0
    for i in 0..<(B*D_MODEL) {
        let p = Double(pred[i]); let gv = Double(g[i])
        dot += p*gv; np += p*p; ng += gv*gv
    }
    let cos = dot / (sqrt(np)*sqrt(ng) + 1e-30)
    print(String(format: "  L%d cos=%.6f  [%@]", L, cos,
                 (cos >= 0.97 ? "PASS" : "FAIL") as CVarArg))
}

print("\n[bench] warmup \(WARMUP)x ...")
for _ in 0..<WARMUP { for L in 0..<N_LAYERS { _ = runLayerPrebuilt(L) } }

print("[bench] timing \(ITERS)x full chain (pre-packed inputs) ...")
var chain: [Double] = []
var perL: [[Double]] = Array(repeating: [], count: N_LAYERS)
for _ in 0..<ITERS {
    let t0 = nowNs()
    for L in 0..<N_LAYERS {
        let s = nowNs()
        _ = runLayerPrebuilt(L)
        perL[L].append(ms(s, nowNs()))
    }
    chain.append(ms(t0, nowNs()))
}
print("\n=== Pre-packed chain (M4 Max, 8L) ===")
for L in 0..<N_LAYERS {
    print(String(format: "  L%d: %.3f ms", L, median(perL[L])))
}
print(String(format: "\n  total chain median: %.2f ms", median(chain)))
print(String(format: "  vs profiled prep+predict+scatter (25.79): %.1f%% reduction",
             100 * (25.79 - median(chain)) / 25.79))
