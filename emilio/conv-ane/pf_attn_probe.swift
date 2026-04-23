// pf_attn_probe.swift — verify Swift attn I/O convention vs golden.
import CoreML
import Foundation

let DIR = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let D = 640, T = 128

func readF16(_ p: String, _ n: Int) -> [Float16] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: p))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self).prefix(n)) }
}

let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
let m = try MLModel(contentsOf: URL(fileURLWithPath: "\(DIR)/PF_attn0_T128.mlmodelc"),
                    configuration: cfg)

// Inspect strides
let probe = try MLMultiArray(shape: [1, 640, 1, 128], dataType: .float16)
print("input shape:", probe.shape, "strides:", probe.strides)

let ain = readF16("/tmp/_ai.bin", T*D)         // [T, D] fp16
let aout = readF16("/tmp/_ao.bin", T*D)        // [T, D] fp16 golden
let pad = readF16("/tmp/_pad.bin", T)

// Build x_in = [1, D, 1, T]:  x_in[0, d, 0, t] = ain[t, d]
let xin = try MLMultiArray(shape: [1, NSNumber(value: D), 1, NSNumber(value: T)],
                            dataType: .float16)
let xS = xin.strides.map { Int(truncating: $0) }
print("xin strides:", xS)
let xp = xin.dataPointer.bindMemory(to: Float16.self, capacity: xin.count)
for d in 0..<D {
    for t in 0..<T {
        xp[d * xS[1] + t * xS[3]] = ain[t * D + d]
    }
}
let padIn = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: T)], dataType: .float16)
let pS = padIn.strides.map { Int(truncating: $0) }
let pp = padIn.dataPointer.bindMemory(to: Float16.self, capacity: padIn.count)
for t in 0..<T { pp[t * pS[3]] = pad[t] }

let prov = try MLDictionaryFeatureProvider(dictionary: ["x_in": xin, "pad_add": padIn])
let pred = try m.prediction(from: prov)
let yArr = pred.featureValue(for: "x_out")!.multiArrayValue!
print("output shape:", yArr.shape, "strides:", yArr.strides)
let yp = yArr.dataPointer.bindMemory(to: Float16.self, capacity: yArr.count)
let yS = yArr.strides.map { Int(truncating: $0) }

// Read back as [T, D]: read[t, d] = yp[d * yS[1] + t * yS[3]]
var pred_TD = [Float](repeating: 0, count: T*D)
for d in 0..<D {
    for t in 0..<T {
        pred_TD[t * D + d] = Float(yp[d * yS[1] + t * yS[3]])
    }
}

// Cosine vs golden (whole tensor, all 128 tokens)
var dot = 0.0, nP = 0.0, nG = 0.0
for i in 0..<(T*D) {
    let p = Double(pred_TD[i]); let g = Double(aout[i])
    dot += p*g; nP += p*p; nG += g*g
}
let cos = dot / (sqrt(nP) * sqrt(nG) + 1e-30)
print(String(format: "cos vs golden L0_attn_out: %.6f  ||p||=%.2f  ||g||=%.2f", cos, sqrt(nP), sqrt(nG)))

// Also per-token cos for first 5 tokens
print("\nper-token cos (first 10 tokens):")
for t in 0..<10 {
    var d2=0.0, p2=0.0, g2=0.0
    for d in 0..<D {
        let pv = Double(pred_TD[t*D+d]); let gv = Double(aout[t*D+d])
        d2 += pv*gv; p2 += pv*pv; g2 += gv*gv
    }
    print(String(format: "  t%d cos=%.6f  ||p||=%.2f  ||g||=%.2f", t, d2/(sqrt(p2)*sqrt(g2)+1e-30), sqrt(p2), sqrt(g2)))
}

print("\npred row 0 first 5:", (0..<5).map { pred_TD[$0] })
print("gold row 0 first 5:", (0..<5).map { Float(aout[$0]) })
