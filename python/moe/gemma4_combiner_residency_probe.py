"""ANE residency probe for pre-built Gemma-4 FFN combiner/partial shards.

Loads existing .mlpackage, compiles, and reports per-op compute-unit placement
via MLComputePlan (macOS 14.4+ / coremltools 9).
"""
import shutil
import sys
from pathlib import Path

import coremltools as ct

SHARDS = [
    "gemma4_shard5_6_real_ffn_combine_q8.mlpackage",
    "gemma4_shard0_1_real_ffn_combine_q8.mlpackage",
    "gemma4_shard5_6_real_ffn_p0of2_q8.mlpackage",
]

OUT_DIR = Path(__file__).parent / "out"
TMP_DIR = Path(__file__).parent.parent / "tmp" / "combiner_residency"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COMPUTE_HEAVY = {"linear", "conv", "matmul", "batch_norm", "layer_norm",
                 "instance_norm", "gelu", "silu", "relu", "sigmoid",
                 "softmax", "add", "mul", "reduce_mean", "reduce_sum",
                 "constexpr_blockwise_shift_scale"}


def _base_op(name: str) -> str:
    """Strip ios16./ios18. prefix to get base op name."""
    for prefix in ("ios18.", "ios16.", "ios17.", "ios15."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def probe_shard(mlpackage_path: Path):
    name = mlpackage_path.stem
    print(f"\n{'='*60}")
    print(f"Shard: {mlpackage_path.name}")

    # Get weight file size
    weights_dir = mlpackage_path / "Data" / "com.apple.CoreML" / "weights"
    total_weight_mb = 0.0
    if weights_dir.exists():
        for f in weights_dir.iterdir():
            total_weight_mb += f.stat().st_size / (1024 * 1024)
    print(f"Weight size: {total_weight_mb:.1f} MB")

    # Compile
    compiled_path = TMP_DIR / f"{name}.mlmodelc"
    if compiled_path.exists():
        shutil.rmtree(compiled_path)
    print("Compiling...")
    compiled_path = Path(ct.utils.compile_model(str(mlpackage_path), str(compiled_path)))

    # Load compute plan
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(
        path=str(compiled_path),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    if program is None:
        print("  ERROR: no program structure")
        return

    counts = {"ANE": 0, "GPU": 0, "CPU": 0, "unknown": 0}
    heavy_cpu = []
    heavy_gpu = []
    rows = []

    for fn_name, fn in program.functions.items():
        for op in fn.block.operations:
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if dev is None:
                    dev_name = "unknown"
                else:
                    raw = dev.preferred_compute_device.__class__.__name__
                    if "NeuralEngine" in raw:
                        dev_name = "ANE"
                    elif "GPU" in raw:
                        dev_name = "GPU"
                    elif "CPU" in raw:
                        dev_name = "CPU"
                    else:
                        dev_name = raw
            except Exception:
                dev_name = "unknown"

            counts[dev_name] = counts.get(dev_name, 0) + 1
            rows.append((op.operator_name, dev_name))

            base = _base_op(op.operator_name)
            if base in COMPUTE_HEAVY:
                if dev_name == "CPU":
                    heavy_cpu.append(op.operator_name)
                elif dev_name == "GPU":
                    heavy_gpu.append(op.operator_name)

    # Print table
    print(f"\n| {'op':<40} | {'placement':>9} |")
    print(f"|{'-'*42}|{'-'*11}|")
    for op_name, dev in rows:
        print(f"| {op_name:<40} | {dev:>9} |")

    print(f"\nSummary: ANE={counts.get('ANE',0)}  GPU={counts.get('GPU',0)}  "
          f"CPU={counts.get('CPU',0)}  unknown={counts.get('unknown',0)}")

    if heavy_cpu:
        print(f"*** FAIL: compute-heavy ops on CPU: {heavy_cpu}")
    if heavy_gpu:
        print(f"*** WARN: compute-heavy ops on GPU: {heavy_gpu}")
    if not heavy_cpu and not heavy_gpu:
        print("PASS: all compute-heavy ops on ANE")

    return counts, heavy_cpu, heavy_gpu


def main():
    results = {}
    for shard_name in SHARDS:
        p = OUT_DIR / shard_name
        if not p.exists():
            print(f"SKIP: {p} not found")
            continue
        results[shard_name] = probe_shard(p)

    # Final verdict
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    all_pass = True
    for name, r in results.items():
        if r is None:
            print(f"  {name}: ERROR")
            all_pass = False
            continue
        counts, hcpu, hgpu = r
        if hcpu:
            print(f"  {name}: FAIL (CPU ops: {hcpu})")
            all_pass = False
        elif hgpu:
            print(f"  {name}: WARN (GPU ops: {hgpu})")
        else:
            print(f"  {name}: PASS (ANE={counts.get('ANE',0)})")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
