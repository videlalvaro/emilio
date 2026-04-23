// Per-layer attn pack probe: feed each layer's golden attn_in, compare attn_out.
import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let D = 640, T = 128

func readF16(_ p: String, _ n: Int) -> [Float16] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: p))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self).prefix(n)) }
}

let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
for L in 0..<8 {
    let m = try MLModel(contentsOf: URL(fileURLWithPath: "\(DIR)/PF_attn\(L)_T128.mlmodelc"),
                        configuration: cfg)
    let ain = readF16("/tmp/_ai_L\(L).bin", T*D)
    let aout = readF16("/tmp/_ao_L\(L).bin", T*D)
    let pad = readF16("/tmp/_pad.bin", T)

    let xin = try MLMultiArray(shape: [1, NSNumber(value: D), 1, NSNumber(value: T)],
                                dataType: .float16)
    let xS = xin.strides.map { Int(truncating: $0) }
    let xp = xin.dataPointer.bindMemory(to: Float16.self, capacity: xin.count)
    for d in 0..<D { for t in 0..<T { xp[d*xS[1] + t*xS[3]] = ain[t*D + d] } }
    let padIn = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: T)], dataType: .float16)
    let pS = padIn.strides.map { Int(truncating: $0) }
    let pp = padIn.dataPointer.bindMemory(to: Float16.self, capacity: padIn.count)
    for t in 0..<T { pp[t*pS[3]] = pad[t] }
    let prov = try MLDictionaryFeatureProvider(dictionary: ["x_in": xin, "pad_add": padIn])
    let pred = try m.prediction(from: prov)
    let yArr = pred.featureValue(for: "x_out")!.multiArrayValue!
    let yp = yArr.dataPointer.bindMemory(to: Float16.self, capacity: yArr.count)
    let yS = yArr.strides.map { Int(truncating: $0) }
    var dot=0.0, nP=0.0, nG=0.0
    for d in 0..<D { for t in 0..<T {
        let p = Double(yp[d*yS[1] + t*yS[3]]); let g = Double(aout[t*D + d])
        dot += p*g; nP += p*p; nG += g*g
    } }
    let cos = dot / (sqrt(nP)*sqrt(nG) + 1e-30)
    print(String(format: "L%d  cos=%.6f  ||pred||=%.1f  ||gold||=%.1f", L, cos, sqrt(nP), sqrt(nG)))
}
