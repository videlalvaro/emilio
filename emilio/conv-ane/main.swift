// Conv-ANE: Conway's Game of Life on Apple Neural Engine
//
// Every generation is a 3×3 convolution + ReLU-based rule,
// stacked N layers deep and pipelined through the ANE.
//
// Build: swiftc -O -framework CoreML -o conv_ane main.swift
// Usage: ./conv_ane [n_generations] [--cpu-only]

import CoreML
import Foundation

// MARK: - Grid

typealias Grid = [[Float]]

func emptyGrid(_ h: Int, _ w: Int) -> Grid {
    Array(repeating: Array(repeating: Float(0), count: w), count: h)
}

func setCell(_ g: inout Grid, _ r: Int, _ c: Int) {
    guard r >= 0, r < g.count, c >= 0, c < g[0].count else { return }
    g[r][c] = 1
}

// MARK: - Classic Patterns

func placeGlider(_ g: inout Grid, _ r: Int, _ c: Int) {
    //  .X.
    //  ..X
    //  XXX
    for (dr, dc) in [(0,1), (1,2), (2,0), (2,1), (2,2)] {
        setCell(&g, r+dr, c+dc)
    }
}

func placeBlinker(_ g: inout Grid, _ r: Int, _ c: Int) {
    // XXX (period-2 oscillator)
    for dc in 0...2 { setCell(&g, r, c+dc) }
}

func placeBlock(_ g: inout Grid, _ r: Int, _ c: Int) {
    // XX  (still life)
    // XX
    for dr in 0...1 { for dc in 0...1 { setCell(&g, r+dr, c+dc) } }
}

func placeRPentomino(_ g: inout Grid, _ r: Int, _ c: Int) {
    // .XX  (chaotic, stabilizes at gen 1103)
    // XX.
    // .X.
    for (dr, dc) in [(0,1), (0,2), (1,0), (1,1), (2,1)] {
        setCell(&g, r+dr, c+dc)
    }
}

func placeGliderGun(_ g: inout Grid, _ r: Int, _ c: Int) {
    // Gosper glider gun (period 30)
    let cells = [
        (0,24),
        (1,22), (1,24),
        (2,12), (2,13), (2,20), (2,21), (2,34), (2,35),
        (3,11), (3,15), (3,20), (3,21), (3,34), (3,35),
        (4,0), (4,1), (4,10), (4,16), (4,20), (4,21),
        (5,0), (5,1), (5,10), (5,14), (5,16), (5,17), (5,22), (5,24),
        (6,10), (6,16), (6,24),
        (7,11), (7,15),
        (8,12), (8,13),
    ]
    for (dr, dc) in cells { setCell(&g, r+dr, c+dc) }
}

// MARK: - CPU Reference

func golStep(_ g: Grid) -> Grid {
    let h = g.count, w = g[0].count
    var next = emptyGrid(h, w)
    for r in 0..<h {
        for c in 0..<w {
            var n: Float = 0
            for dr in -1...1 {
                for dc in -1...1 {
                    if dr == 0 && dc == 0 { continue }
                    let nr = r + dr, nc = c + dc
                    if nr >= 0, nr < h, nc >= 0, nc < w {
                        n += g[nr][nc]
                    }
                }
            }
            if g[r][c] > 0.5 {
                if n == 2 || n == 3 { next[r][c] = 1 }
            } else {
                if n == 3 { next[r][c] = 1 }
            }
        }
    }
    return next
}

func golN(_ g: Grid, _ n: Int) -> Grid {
    var g = g
    for _ in 0..<n { g = golStep(g) }
    return g
}

// MARK: - Display

func countAlive(_ g: Grid) -> Int {
    g.flatMap { $0 }.filter { $0 > 0.5 }.count
}

func printGrid(_ g: Grid, _ label: String) {
    print("\n\(label):")
    // Bounding box of live cells
    var minR = g.count, maxR = 0, minC = g[0].count, maxC = 0
    for r in 0..<g.count {
        for c in 0..<g[0].count {
            if g[r][c] > 0.5 {
                minR = min(minR, r); maxR = max(maxR, r)
                minC = min(minC, c); maxC = max(maxC, c)
            }
        }
    }
    if minR > maxR { print("  (empty)"); return }
    let pad = 2
    let r0 = max(0, minR - pad), r1 = min(g.count - 1, maxR + pad)
    let c0 = max(0, minC - pad), c1 = min(g[0].count - 1, maxC + pad)
    for r in r0...r1 {
        var line = "  "
        for c in c0...c1 { line += g[r][c] > 0.5 ? "█" : "·" }
        print(line)
    }
    print("  [\(countAlive(g)) live cells]")
}

// MARK: - CoreML I/O

func toMLArray(_ g: Grid) throws -> MLMultiArray {
    let h = g.count, w = g[0].count
    let arr = try MLMultiArray(
        shape: [1, 1, NSNumber(value: h), NSNumber(value: w)],
        dataType: .float32)
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    for r in 0..<h {
        for c in 0..<w {
            ptr[r * w + c] = g[r][c]
        }
    }
    return arr
}

func fromMLArray(_ a: MLMultiArray, _ h: Int, _ w: Int) -> Grid {
    var g = emptyGrid(h, w)
    let ptr = a.dataPointer.assumingMemoryBound(to: Float.self)
    for r in 0..<h {
        for c in 0..<w {
            g[r][c] = ptr[r * w + c]
        }
    }
    return g
}

// MARK: - Benchmark Helper

func bench(_ label: String, iterations: Int, _ body: () throws -> MLFeatureProvider) throws -> (ms: Double, last: MLFeatureProvider) {
    // Warm up
    let _ = try body()

    let start = CFAbsoluteTimeGetCurrent()
    var result: MLFeatureProvider!
    for _ in 0..<iterations {
        result = try body()
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    return (ms: elapsed / Double(iterations) * 1000, last: result)
}

// MARK: - Main

func main() throws {
    // Parse args
    let rawArgs = Array(CommandLine.arguments.dropFirst())
    let cpuOnly = rawArgs.contains("--cpu-only")
    let nGens = rawArgs.first(where: { $0.first != "-" }).flatMap(Int.init) ?? 32
    let benchN = 200

    // Load model
    let modelURL = URL(fileURLWithPath: "GOL.mlmodelc")
    guard FileManager.default.fileExists(atPath: "GOL.mlmodelc") else {
        print("Error: GOL.mlmodelc not found. Run build.sh first.")
        return
    }

    let aneConfig = MLModelConfiguration()
    aneConfig.computeUnits = .cpuAndNeuralEngine

    let cpuConfig = MLModelConfiguration()
    cpuConfig.computeUnits = .cpuOnly

    print("Loading GOL.mlmodelc...")
    let aneModel = try MLModel(contentsOf: modelURL, configuration: aneConfig)
    let cpuModel = cpuOnly ? nil : try MLModel(contentsOf: modelURL, configuration: cpuConfig)

    // Read grid dimensions from model
    let shape = aneModel.modelDescription.inputDescriptionsByName["grid"]!
        .multiArrayConstraint!.shape
    let H = shape[2].intValue, W = shape[3].intValue
    let outputKey = aneModel.modelDescription.outputDescriptionsByName.keys.first!

    print("Grid: \(H)×\(W), \(nGens) generations per call")
    print("Compute: \(cpuOnly ? "CPU only" : "ANE + CPU comparison")")

    // Initial grid with patterns
    var grid = emptyGrid(H, W)
    placeGlider(&grid, 1, 1)
    placeBlinker(&grid, 10, 10)
    placeBlock(&grid, 10, 20)
    if H >= 48 && W >= 48 {
        placeRPentomino(&grid, H / 2, W / 2)
    }
    if H >= 48 && W >= 48 {
        placeGliderGun(&grid, H / 2 - 20, 2)
    }
    printGrid(grid, "Initial (gen 0)")

    // Prepare input
    let input = try toMLArray(grid)
    let provider = try MLDictionaryFeatureProvider(
        dictionary: ["grid": MLFeatureValue(multiArray: input)])

    // Benchmark ANE (or CPU if --cpu-only)
    let label = cpuOnly ? "CPU" : "ANE"
    let primaryModel = cpuOnly ? (try MLModel(contentsOf: modelURL, configuration: cpuConfig)) : aneModel

    print("\nBenchmarking \(label) (\(benchN) iterations)...")
    let (primaryMs, primaryResult) = try bench(label, iterations: benchN) {
        try primaryModel.prediction(from: provider)
    }

    // Extract result
    let output = primaryResult.featureValue(for: outputKey)!.multiArrayValue!
    let resultGrid = fromMLArray(output, H, W)
    printGrid(resultGrid, "\(label) output (gen \(nGens))")

    // Benchmark CPU via CoreML (for comparison)
    var cpuCoreMLMs: Double? = nil
    if !cpuOnly, let cpuM = cpuModel {
        print("Benchmarking CPU (\(benchN) iterations)...")
        let (ms, _) = try bench("CPU", iterations: benchN) {
            try cpuM.prediction(from: provider)
        }
        cpuCoreMLMs = ms
    }

    // CPU reference verification
    print("\nVerifying against CPU reference...")
    let cpuRefStart = CFAbsoluteTimeGetCurrent()
    let cpuRef = golN(grid, nGens)
    let cpuRefMs = (CFAbsoluteTimeGetCurrent() - cpuRefStart) * 1000

    var mismatches = 0
    for r in 0..<H {
        for c in 0..<W {
            if (resultGrid[r][c] > 0.5) != (cpuRef[r][c] > 0.5) {
                mismatches += 1
            }
        }
    }

    // Report
    let gensPerSec = Double(nGens * benchN) / (primaryMs * Double(benchN) / 1000)
    let cellGensPerSec = gensPerSec * Double(H * W)

    print("""

    ╔════════════════════════════════════════════════╗
    ║          Conv-ANE  ·  Results                  ║
    ╚════════════════════════════════════════════════╝

      Grid:            \(H)×\(W) (\(H * W) cells)
      Generations:     \(nGens) per call
      Iterations:      \(benchN)

      \(label) latency:    \(String(format: "%.3f", primaryMs)) ms/call\
    \(cpuCoreMLMs.map { "\n      CPU latency:    \(String(format: "%.3f", $0)) ms/call" } ?? "")\
    \(!cpuOnly && cpuCoreMLMs != nil ? "\n      Speedup:        \(String(format: "%.1f", cpuCoreMLMs! / primaryMs))×" : "")

      Throughput:      \(String(format: "%.0f", gensPerSec)) gen/sec
      Cell-gens/sec:   \(String(format: "%.2e", cellGensPerSec))

      CPU ref (Swift):  \(String(format: "%.1f", cpuRefMs)) ms for \(nGens) gens
      Verification:     \(mismatches == 0 ? "✓ PASS" : "✗ FAIL (\(mismatches) mismatches)")
    """)
}

try main()
