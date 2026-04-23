// pf_chain_profile.swift
// Phase-level profiler for the per-expert MoE chain.
// Decomposes each layer into:
//   t_prep    : alloc input buffer + Float16 copy of norm rows
//   t_predict : dispatch + group.wait()  (wall-time including ANE)
//   t_scatter : weighted accumulate into delta
// Build: swiftc -O -framework CoreML -o pf_chain_profile pf_chain_profile.swift
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
        let p = "\(DIR)/PF_expert_L\(L)_\(e)_B\(B)_fp16.mlmodelc"
        layer.append(try loadOnANE(p))
    }
    experts.append(layer)
}
print(String(format: "[load] done in %.1fs", ms(lt0, nowNs())/1000))

struct LayerData {
    let norm: [Float16]
    let delta:[Float16]
    let distinct: [Int]
    let perExpertRows: [Int: [(Int, Float)]]
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
            let eid = Int(idx[r*K + k]); let wv = Float(w[r*K + k])
            if wv == 0 { continue }
            byExpert[eid, default: []].append((r, wv))
        }
    }
    layers.append(LayerData(norm: norm, delta: delta,
                            distinct: Array(byExpert.keys).sorted(),
                            perExpertRows: byExpert))
}

let q = DispatchQueue.global(qos: .userInitiated)

// Returns (t_prep_ms, t_predict_ms, t_scatter_ms) for this layer.
func runLayerProfiled(_ L: Int) throws -> (Double, Double, Double) {
    let ld = layers[L]
    let layerExperts = experts[L]

    // ---- PREP ----
    let pPrep0 = nowNs()
    var prepared: [(model: MLModel, xin: MLMultiArray,
                    prov: MLDictionaryFeatureProvider,
                    rows: [(Int, Float)], outBuf: UnsafeMutablePointer<Float16>)] = []
    prepared.reserveCapacity(ld.distinct.count)
    for eid in ld.distinct {
        let rows = ld.perExpertRows[eid]!
        let xin = try MLMultiArray(shape: [NSNumber(value: B), NSNumber(value: D_MODEL),
                                           1, 1], dataType: .float16)
        let xp = xin.dataPointer.bindMemory(to: Float16.self, capacity: xin.count)
        memset(xp, 0, xin.count * MemoryLayout<Float16>.stride)
        for (slot, (r, _)) in rows.enumerated() {
            let src = r * D_MODEL; let dst = slot * D_MODEL
            for d in 0..<D_MODEL { xp[dst + d] = ld.norm[src + d] }
        }
        let prov = try MLDictionaryFeatureProvider(dictionary: ["x_in": xin])
        let outBuf = UnsafeMutablePointer<Float16>.allocate(capacity: rows.count * D_MODEL)
        prepared.append((layerExperts[eid], xin, prov, rows, outBuf))
    }
    let tPrep = ms(pPrep0, nowNs())

    // ---- PREDICT ----
    let pPred0 = nowNs()
    let group = DispatchGroup()
    for item in prepared {
        let m = item.model; let prov = item.prov
        let rows = item.rows; let outBuf = item.outBuf
        group.enter()
        q.async {
            do {
                let pred = try m.prediction(from: prov)
                let yArr = pred.featureValue(for: "y_out")!.multiArrayValue!
                let yp = yArr.dataPointer.bindMemory(to: Float16.self, capacity: yArr.count)
                let s0 = Int(truncating: yArr.strides[0])
                let s1 = Int(truncating: yArr.strides[1])
                for slot in 0..<rows.count {
                    let dst = slot * D_MODEL; let src = slot * s0
                    for d in 0..<D_MODEL { outBuf[dst + d] = yp[src + d * s1] }
                }
            } catch { fatalError("predict L\(L): \(error)") }
            group.leave()
        }
    }
    group.wait()
    let tPred = ms(pPred0, nowNs())

    // ---- SCATTER ----
    let pSc0 = nowNs()
    var delta = [Float](repeating: 0, count: B * D_MODEL)
    let kScale = Float(K)
    for item in prepared {
        for (slot, (row, wv)) in item.rows.enumerated() {
            let src = slot * D_MODEL; let dst = row * D_MODEL
            let s = wv * kScale
            for d in 0..<D_MODEL {
                delta[dst + d] += s * Float(item.outBuf[src + d])
            }
        }
        item.outBuf.deallocate()
    }
    let tSc = ms(pSc0, nowNs())
    return (tPrep, tPred, tSc)
}

print("[warmup] \(WARMUP)x ...")
for _ in 0..<WARMUP { for L in 0..<N_LAYERS { _ = try runLayerProfiled(L) } }

print("[profile] \(ITERS) iters ...")
var prep: [[Double]] = Array(repeating: [], count: N_LAYERS)
var pred: [[Double]] = Array(repeating: [], count: N_LAYERS)
var scat: [[Double]] = Array(repeating: [], count: N_LAYERS)
for _ in 0..<ITERS {
    for L in 0..<N_LAYERS {
        let (a, b, c) = try runLayerProfiled(L)
        prep[L].append(a); pred[L].append(b); scat[L].append(c)
    }
}

print("\n=== Per-layer phase breakdown (median ms) ===")
print("   L     prep   predict   scatter     total")
var sP = 0.0, sR = 0.0, sS = 0.0
for L in 0..<N_LAYERS {
    let p = median(prep[L]), r = median(pred[L]), s = median(scat[L])
    sP += p; sR += r; sS += s
    print(String(format: "%4d  %8.3f  %8.3f  %8.3f  %8.3f", L, p, r, s, p+r+s))
}
print(String(format: "   Σ  %8.3f  %8.3f  %8.3f  %8.3f", sP, sR, sS, sP+sR+sS))
print(String(format: "\nshare:  prep=%.0f%%  predict=%.0f%%  scatter=%.0f%%",
             100*sP/(sP+sR+sS), 100*sR/(sP+sR+sS), 100*sS/(sP+sR+sS)))
