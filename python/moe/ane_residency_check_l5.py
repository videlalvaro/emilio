"""ANE residency check for L5 rebuilt FFN shards.

Loads pre-built .mlpackage, compiles, and uses MLComputePlan to report
per-op compute device placement. Flags any compute-heavy op on CPU/GPU.
"""
import sys
from pathlib import Path
import coremltools as ct

COMPUTE_HEAVY_OPS = {
    "linear", "conv", "matmul", "batch_norm", "layer_norm",
    "instance_norm", "gelu", "silu", "relu", "softmax",
    "mul", "add", "sub",  # element-wise that can be fused
}

OUT = Path(__file__).resolve().parent / "out"

SHARDS = [
    "gemma4_shard5_6_real_ffn_p0of2_q8.mlpackage",
    "gemma4_shard5_6_real_ffn_p1of2_q8.mlpackage",
    "gemma4_shard5_6_real_ffn_combine_q8.mlpackage",
    "gemma4_shard0_1_real_ffn_combine_q8.mlpackage",
]


def check_shard(name):
    pkg = OUT / name
    compiled_name = name.replace(".mlpackage", ".mlmodelc")
    compiled = OUT / compiled_name

    if not pkg.exists():
        print(f"  MISSING: {pkg}")
        return None

    # compile if needed
    if not compiled.exists():
        print(f"  Compiling {name} ...")
        compiled = Path(ct.utils.compile_model(str(pkg), str(compiled)))
    else:
        print(f"  Using existing {compiled_name}")

    # weight size
    weight_bytes = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file())
    weight_mb = weight_bytes / (1024 * 1024)

    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(
        path=str(compiled),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    if program is None:
        print("  ERROR: no program structure")
        return None

    counts = {"ANE": 0, "GPU": 0, "CPU": 0, "?": 0}
    flagged = []
    rows = []

    for fn_name, fn in program.functions.items():
        for op in fn.block.operations:
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if dev is None:
                    dev_name = "?"
                else:
                    dev_name = dev.preferred_compute_device.__class__.__name__
                    # MLNeuralEngineComputeDevice -> ANE
                    if "NeuralEngine" in dev_name:
                        dev_name = "ANE"
                    elif "GPU" in dev_name:
                        dev_name = "GPU"
                    elif "CPU" in dev_name:
                        dev_name = "CPU"
            except Exception:
                dev_name = "?"

            counts[dev_name] = counts.get(dev_name, 0) + 1
            op_name = op.operator_name
            rows.append((op_name, dev_name))

            # Strip namespace prefix (e.g. ios18.linear -> linear)
            base_op = op_name.rsplit(".", 1)[-1] if "." in op_name else op_name
            if dev_name in ("CPU", "GPU") and base_op in COMPUTE_HEAVY_OPS:
                flagged.append((op_name, dev_name))

    return {
        "weight_mb": weight_mb,
        "counts": counts,
        "rows": rows,
        "flagged": flagged,
    }


def main():
    all_pass = True
    for name in SHARDS:
        print(f"\n{'='*60}")
        print(f"Shard: {name}")
        print(f"{'='*60}")
        result = check_shard(name)
        if result is None:
            all_pass = False
            continue

        print(f"  Package size: {result['weight_mb']:.1f} MB")
        print(f"  Op counts: ANE={result['counts'].get('ANE',0)}  "
              f"GPU={result['counts'].get('GPU',0)}  "
              f"CPU={result['counts'].get('CPU',0)}  "
              f"?={result['counts'].get('?',0)}")
        print()
        print(f"  {'op':<30} {'device':<8}")
        print(f"  {'-'*30} {'-'*8}")
        for op_name, dev in result["rows"]:
            base = op_name.rsplit(".", 1)[-1] if "." in op_name else op_name
            marker = " <<< FAIL" if (dev in ("CPU", "GPU") and base in COMPUTE_HEAVY_OPS) else ""
            print(f"  {op_name:<30} {dev:<8}{marker}")

        if result["flagged"]:
            all_pass = False
            print(f"\n  *** FAIL: {len(result['flagged'])} compute-heavy op(s) NOT on ANE:")
            for op_name, dev in result["flagged"]:
                print(f"      {op_name} -> {dev}")
        else:
            print(f"\n  PASS: all compute-heavy ops on ANE")

    print(f"\n{'='*60}")
    if all_pass:
        print("OVERALL: PASS — all 4 shards fully ANE-resident")
    else:
        print("OVERALL: FAIL — see flagged ops above")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
