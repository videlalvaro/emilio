// pf_attn4_fp32_probe.swift
// A/B compare PF_attn4_T128 (fp16) vs PF_attn4_T128_fp32 (fp32 compute) vs golden.
// Repo-local artifact paths only (no /tmp).
import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let REPO = URL(fileURLWithPath: DIR).deletingLastPathComponent().deletingLastPathComponent().path
let TMP = "\(REPO)/python/privacy/out/probe"
let D = 640, T = 128

func readF16(_ p: String, _ n: Int) -> [Float16] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: p))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self).prefix(n)) }
}

func loadModel(_ path: String, units: MLComputeUnits) throws -> MLModel {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = units
    return try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
}

func makeInputs(ain: [Float16], pad: [Float16]) throws
    -> MLDictionaryFeatureProvider
{
    let xin = try MLMultiArray(shape: [1, NSNumber(value: D), 1, NSNumber(value: T)],
                                dataType: .float16)
    let xS = xin.strides.map { Int(truncating: $0) }
    let xp = xin.dataPointer.bindMemory(to: Float16.self, capacity: xin.count)
    for d in 0..<D { for t in 0..<T { xp[d*xS[1] + t*xS[3]] = ain[t*D + d] } }
    let padIn = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: T)],
                                 dataType: .float16)
    let pS = padIn.strides.map { Int(truncating: $0) }
    let pp = padIn.dataPointer.bindMemory(to: Float16.self, capacity: padIn.count)
    for t in 0..<T { pp[t*pS[3]] = pad[t] }
    return try MLDictionaryFeatureProvider(dictionary: ["x_in": xin, "pad_add": padIn])
}

func runOne(model: MLModel, prov: MLDictionaryFeatureProvider) throws -> [Float] {
    _ = try model.prediction(from: prov)        // warm-up
    let pred = try model.prediction(from: prov)
    let yArr = pred.featureValue(for: "x_out")!.multiArrayValue!
    let yp = yArr.dataPointer.bindMemory(to: Float16.self, capacity: yArr.count)
    let yS = yArr.strides.map { Int(truncating: $0) }
    var out = [Float](repeating: 0, count: T*D)
    for d in 0..<D { for t in 0..<T {
        out[t*D + d] = Float(yp[d*yS[1] + t*yS[3]])
    } }
    return out
}

func cosVs(_ pred: [Float], _ gold: [Float16]) -> (Double, Double, Double) {
    var dot = 0.0, nP = 0.0, nG = 0.0
    for i in 0..<(T*D) {
        let p = Double(pred[i]); let g = Double(gold[i])
        dot += p*g; nP += p*p; nG += g*g
    }
    return (dot/(sqrt(nP)*sqrt(nG)+1e-30), sqrt(nP), sqrt(nG))
}

func benchMs(_ model: MLModel, prov: MLDictionaryFeatureProvider, iters: Int) throws -> Double {
    _ = try model.prediction(from: prov)        // warm-up
    var times: [Double] = []
    for _ in 0..<iters {
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try model.prediction(from: prov)
        let t1 = DispatchTime.now().uptimeNanoseconds
        times.append(Double(t1 - t0) / 1e6)
    }
    times.sort()
    return times[times.count / 2]
}

let ain = readF16("\(TMP)/_ai_L4.bin", T*D)
let aout = readF16("\(TMP)/_ao_L4.bin", T*D)
let pad = readF16("\(TMP)/_pad.bin", T)
let prov = try makeInputs(ain: ain, pad: pad)

print("=== fp16 baseline pack on .cpuAndNeuralEngine")
let m_fp16_ane = try loadModel("\(DIR)/PF_attn4_T128.mlmodelc", units: .cpuAndNeuralEngine)
let p1 = try runOne(model: m_fp16_ane, prov: prov)
let (c1, np1, ng1) = cosVs(p1, aout)
let t1 = try benchMs(m_fp16_ane, prov: prov, iters: 20)
print(String(format: "    cos=%.6f  ||pred||=%.1f  ||gold||=%.1f  time=%.3f ms",
             c1, np1, ng1, t1))

print("\n=== fp32-compute pack on .cpuAndNeuralEngine (CPU fallback if op-set forbids ANE)")
let m_fp32_ane = try loadModel("\(DIR)/PF_attn4_T128_fp32.mlmodelc", units: .cpuAndNeuralEngine)
let p2 = try runOne(model: m_fp32_ane, prov: prov)
let (c2, np2, ng2) = cosVs(p2, aout)
let t2 = try benchMs(m_fp32_ane, prov: prov, iters: 20)
print(String(format: "    cos=%.6f  ||pred||=%.1f  ||gold||=%.1f  time=%.3f ms",
             c2, np2, ng2, t2))

print("\n=== fp32-compute pack on .cpuOnly  (sanity baseline for residency proxy)")
let m_fp32_cpu = try loadModel("\(DIR)/PF_attn4_T128_fp32.mlmodelc", units: .cpuOnly)
let p3 = try runOne(model: m_fp32_cpu, prov: prov)
let (c3, np3, _) = cosVs(p3, aout)
let t3 = try benchMs(m_fp32_cpu, prov: prov, iters: 5)
print(String(format: "    cos=%.6f  ||pred||=%.1f  time=%.3f ms", c3, np3, t3))

print("\n=== MIXED pack (fp32 softmax only) on .cpuAndNeuralEngine")
let m_mix_ane = try loadModel("\(DIR)/PF_attn4_T128_mixed.mlmodelc", units: .cpuAndNeuralEngine)
let p4 = try runOne(model: m_mix_ane, prov: prov)
let (c4, np4, _) = cosVs(p4, aout)
let t4 = try benchMs(m_mix_ane, prov: prov, iters: 20)
print(String(format: "    cos=%.6f  ||pred||=%.1f  time=%.3f ms", c4, np4, t4))

print("\n=== MIXED pack on .cpuOnly (residency proxy)")
let m_mix_cpu = try loadModel("\(DIR)/PF_attn4_T128_mixed.mlmodelc", units: .cpuOnly)
_ = try runOne(model: m_mix_cpu, prov: prov)
let t4cpu = try benchMs(m_mix_cpu, prov: prov, iters: 5)
print(String(format: "    time=%.3f ms", t4cpu))

print("\n=== SAFE-NORM pack (fp16, ANE) on .cpuAndNeuralEngine")
let m_safe_ane = try loadModel("\(DIR)/PF_attn4_T128_safe.mlmodelc", units: .cpuAndNeuralEngine)
let p5 = try runOne(model: m_safe_ane, prov: prov)
let (c5, np5, _) = cosVs(p5, aout)
let t5 = try benchMs(m_safe_ane, prov: prov, iters: 20)
print(String(format: "    cos=%.6f  ||pred||=%.1f  time=%.3f ms", c5, np5, t5))

print("\n=== SAFE-NORM pack on .cpuOnly (residency proxy)")
let m_safe_cpu = try loadModel("\(DIR)/PF_attn4_T128_safe.mlmodelc", units: .cpuOnly)
_ = try runOne(model: m_safe_cpu, prov: prov)
let t5cpu = try benchMs(m_safe_cpu, prov: prov, iters: 5)
print(String(format: "    time=%.3f ms", t5cpu))

print("\n=== Verdict for L4")
print(String(format: "    fp16 baseline:    cos=%.4f  time=%.2f ms", c1, t1))
print(String(format: "    fp32 (CPU+ANE):   cos=%.4f  time=%.2f ms", c2, t2))
print(String(format: "    fp32 (CPU only):  cos=%.4f  time=%.2f ms", c3, t3))
print(String(format: "    mixed(CPU+ANE):   cos=%.4f  time=%.2f ms", c4, t4))
print(String(format: "    mixed(CPU only):  time=%.2f ms", t4cpu))
print(String(format: "    safe (CPU+ANE):   cos=%.4f  time=%.2f ms", c5, t5))
print(String(format: "    safe (CPU only):  time=%.2f ms", t5cpu))
let safeOnAne = (t5 < t5cpu * 0.6)
let safeQuality = c5 >= 0.97
print(String(format: "    SAFE quality (cos>=0.97):     %@", safeQuality ? "PASS" : "FAIL"))
print(String(format: "    SAFE residency (ANE<0.6×CPU): %@",
             safeOnAne ? "ANE LIKELY" : "CPU FALLBACK"))
