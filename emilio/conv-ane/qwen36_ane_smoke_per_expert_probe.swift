import CoreML
import Foundation

struct Args {
    var layer = 0
    var expertIds = [0]
    var warmup = 20
    var iters = 200
    var fanout = 4
    var summaryPath: String? = nil
}

func parseArgs() -> Args {
    var args = Args()
    let cli = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < cli.count {
        switch cli[i] {
        case "--layer":
            args.layer = Int(cli[i + 1]) ?? args.layer
            i += 2
        case "--experts":
            let parsed = cli[i + 1]
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            if !parsed.isEmpty {
                args.expertIds = parsed
            }
            i += 2
        case "--warmup":
            args.warmup = Int(cli[i + 1]) ?? args.warmup
            i += 2
        case "--iters":
            args.iters = Int(cli[i + 1]) ?? args.iters
            i += 2
        case "--fanout":
            args.fanout = max(1, Int(cli[i + 1]) ?? args.fanout)
            i += 2
        case "--summary":
            args.summaryPath = cli[i + 1]
            i += 2
        default:
            i += 1
        }
    }
    return args
}

func median(_ xs: [Double]) -> Double {
    let sorted = xs.sorted()
    let n = sorted.count
    return n % 2 == 1 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
}

func minMs(_ xs: [Double]) -> Double { xs.min() ?? 0.0 }
func maxMs(_ xs: [Double]) -> Double { xs.max() ?? 0.0 }

func loadOnANE(_ path: String) throws -> MLModel {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
}

func findRepoRoot() throws -> String {
    let fm = FileManager.default
    let execDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
    let starts = [fm.currentDirectoryPath, execDir]
    for start in starts {
        var url = URL(fileURLWithPath: start)
        while true {
            let candidate = url.appendingPathComponent("python/moe", isDirectory: true).path
            if fm.fileExists(atPath: candidate) {
                return url.path
            }
            let parent = url.deletingLastPathComponent()
            if parent.path == url.path {
                break
            }
            url = parent
        }
    }
    throw NSError(domain: "qwen36_probe", code: 1, userInfo: [NSLocalizedDescriptionKey: "could not locate repo root"])
}

func makeArray(shape: [NSNumber], dataType: MLMultiArrayDataType, fill: Double) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape, dataType: dataType)
    switch dataType {
    case .float16:
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        let value = Float16(fill)
        for i in 0..<array.count { ptr[i] = value }
    case .float32:
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        let value = Float(fill)
        for i in 0..<array.count { ptr[i] = value }
    case .double:
        let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
        for i in 0..<array.count { ptr[i] = fill }
    default:
        throw NSError(domain: "qwen36_probe", code: 2, userInfo: [NSLocalizedDescriptionKey: "unsupported input dtype \(dataType.rawValue)"])
    }
    return array
}

func inputShape(_ constraint: MLMultiArrayConstraint) -> [Int] {
    return constraint.shape.map { $0.intValue }
}

func dtypeName(_ dataType: MLMultiArrayDataType) -> String {
    switch dataType {
    case .float16: return "float16"
    case .float32: return "float32"
    case .double: return "double"
    case .int32: return "int32"
    default: return "raw_\(dataType.rawValue)"
    }
}

let args = parseArgs()
let repoRoot = try findRepoRoot()
let expertsDir = "\(repoRoot)/python/moe/out/qwen36/experts"
let layerTag = String(format: "L%02d", args.layer)

let modelPaths = args.expertIds.map {
    "\(expertsDir)/qwen36_\(layerTag)_expert\(String(format: "%03d", $0))_int4.mlmodelc"
}

print("[qwen36] repo_root=\(repoRoot)")
print("[qwen36] experts_dir=\(expertsDir)")
print("[qwen36] layer=\(args.layer) experts=\(args.expertIds) fanout=\(args.fanout)")

for path in modelPaths {
    guard FileManager.default.fileExists(atPath: path) else {
        throw NSError(domain: "qwen36_probe", code: 3, userInfo: [NSLocalizedDescriptionKey: "missing expert model: \(path)"])
    }
}

let models = try modelPaths.map(loadOnANE)
guard let firstModel = models.first else {
    throw NSError(domain: "qwen36_probe", code: 4, userInfo: [NSLocalizedDescriptionKey: "no models loaded"])
}

guard let inputName = firstModel.modelDescription.inputDescriptionsByName.keys.sorted().first,
      let inputDesc = firstModel.modelDescription.inputDescriptionsByName[inputName],
      let inputConstraint = inputDesc.multiArrayConstraint else {
    throw NSError(domain: "qwen36_probe", code: 5, userInfo: [NSLocalizedDescriptionKey: "missing multi-array input description"])
}

let shape = inputShape(inputConstraint)
let dtype = inputConstraint.dataType
let probeInput = try makeArray(shape: inputConstraint.shape, dataType: dtype, fill: 0.01)
let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: probeInput])

var dispatchModels = models
if dispatchModels.count < args.fanout {
    let base = dispatchModels
    while dispatchModels.count < args.fanout {
        dispatchModels.append(base[dispatchModels.count % base.count])
    }
} else if dispatchModels.count > args.fanout {
    dispatchModels = Array(dispatchModels.prefix(args.fanout))
}

print("[qwen36] input_name=\(inputName) shape=\(shape) dtype=\(dtypeName(dtype))")
print("[qwen36] loaded_models=\(models.count) dispatch_models=\(dispatchModels.count)")

func runPrediction(_ model: MLModel) throws {
    _ = try model.prediction(from: provider)
}

for _ in 0..<args.warmup {
    try runPrediction(firstModel)
}

var singleTimes = [Double]()
singleTimes.reserveCapacity(args.iters)
for _ in 0..<args.iters {
    let t0 = DispatchTime.now().uptimeNanoseconds
    try runPrediction(firstModel)
    let t1 = DispatchTime.now().uptimeNanoseconds
    singleTimes.append(Double(t1 - t0) / 1e6)
}
let singleMedian = median(singleTimes)
print("[single] median=\(String(format: "%.3f", singleMedian)) ms min=\(String(format: "%.3f", minMs(singleTimes))) max=\(String(format: "%.3f", maxMs(singleTimes)))")

for _ in 0..<args.warmup {
    for model in dispatchModels {
        try runPrediction(model)
    }
}

var seqTimes = [Double]()
seqTimes.reserveCapacity(args.iters)
for _ in 0..<args.iters {
    let t0 = DispatchTime.now().uptimeNanoseconds
    for model in dispatchModels {
        try runPrediction(model)
    }
    let t1 = DispatchTime.now().uptimeNanoseconds
    seqTimes.append(Double(t1 - t0) / 1e6)
}
let seqMedian = median(seqTimes)
print("[seq]    median=\(String(format: "%.3f", seqMedian)) ms per_call=\(String(format: "%.3f", seqMedian / Double(dispatchModels.count)))")

let queue = DispatchQueue.global(qos: .userInitiated)
let group = DispatchGroup()
for _ in 0..<args.warmup {
    for model in dispatchModels {
        group.enter()
        queue.async {
            _ = try? model.prediction(from: provider)
            group.leave()
        }
    }
    group.wait()
}

var concTimes = [Double]()
concTimes.reserveCapacity(args.iters)
for _ in 0..<args.iters {
    let t0 = DispatchTime.now().uptimeNanoseconds
    for model in dispatchModels {
        group.enter()
        queue.async {
            _ = try? model.prediction(from: provider)
            group.leave()
        }
    }
    group.wait()
    let t1 = DispatchTime.now().uptimeNanoseconds
    concTimes.append(Double(t1 - t0) / 1e6)
}
let concMedian = median(concTimes)
print("[conc]   median=\(String(format: "%.3f", concMedian)) ms per_call=\(String(format: "%.3f", concMedian / Double(dispatchModels.count)))")

if let summaryPath = args.summaryPath {
    let summary: [String: Any] = [
        "repo_root": repoRoot,
        "experts_dir": expertsDir,
        "layer": args.layer,
        "expert_ids": args.expertIds,
        "dispatch_count": dispatchModels.count,
        "dispatch_reuses_models": dispatchModels.count > models.count,
        "model_paths": modelPaths,
        "input_name": inputName,
        "input_shape": shape,
        "input_dtype": dtypeName(dtype),
        "warmup": args.warmup,
        "iters": args.iters,
        "single_median_ms": singleMedian,
        "single_min_ms": minMs(singleTimes),
        "single_max_ms": maxMs(singleTimes),
        "seq_median_ms": seqMedian,
        "conc_median_ms": concMedian,
        "seq_per_call_ms": seqMedian / Double(dispatchModels.count),
        "conc_per_call_ms": concMedian / Double(dispatchModels.count),
    ]
    let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: URL(fileURLWithPath: summaryPath))
    print("[summary] \(summaryPath)")
}