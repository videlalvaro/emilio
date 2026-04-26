"""Probe: ANE residency for scatter-based MoE routing shard.

Loads the pre-compiled .mlmodelc and reports per-op compute unit placement
via MLComputePlan (macOS 14.4+ / coremltools 9).
"""
import sys
from pathlib import Path

import coremltools as ct

COMPILED = Path(__file__).resolve().parent / "out" / "gemma4_shard0_1_real_ffn_p0of2_q8.mlmodelc"

# Ops we skip in the table (compile-time constants, not runtime ops)
SKIP_OPS = {"const", "constexpr_blockwise_shift_scale"}


def main():
    if not COMPILED.exists():
        print(f"ERROR: compiled model not found at {COMPILED}")
        sys.exit(1)

    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        path=str(COMPILED),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    if program is None:
        print("ERROR: no program structure found")
        sys.exit(1)

    print(f"Artifact: {COMPILED.name}")
    print(f"{'op':<30} {'MIL name':<50} {'device':<12}")
    print("-" * 92)

    counts = {"ANE": 0, "CPU": 0, "GPU": 0, "?": 0}
    cpu_ops = []
    gpu_ops = []

    for fn_name, fn in program.functions.items():
        for op in fn.block.operations:
            if op.operator_name in SKIP_OPS:
                continue
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if dev is None:
                    dev_name = "?"
                else:
                    dev_name = dev.preferred_compute_device.__class__.__name__
                    # Normalize names
                    if "NeuralEngine" in dev_name:
                        dev_name = "ANE"
                    elif "CPU" in dev_name:
                        dev_name = "CPU"
                    elif "GPU" in dev_name:
                        dev_name = "GPU"
            except Exception as e:
                dev_name = f"err:{e!r}"

            mil_name = getattr(op, "name", "?")
            print(f"  {op.operator_name:<28} {mil_name:<50} {dev_name:<12}")
            counts[dev_name] = counts.get(dev_name, 0) + 1
            if dev_name == "CPU":
                cpu_ops.append((op.operator_name, mil_name))
            elif dev_name == "GPU":
                gpu_ops.append((op.operator_name, mil_name))

    print("-" * 92)
    print(f"Total runtime ops: {sum(counts.values())}")
    for dev, n in sorted(counts.items()):
        if n:
            print(f"  {dev}: {n}")

    if cpu_ops:
        print("\nCPU fallback ops:")
        for op_type, name in cpu_ops:
            print(f"  - {op_type} ({name})")
    if gpu_ops:
        print("\nGPU fallback ops:")
        for op_type, name in gpu_ops:
            print(f"  - {op_type} ({name})")

    if not cpu_ops and not gpu_ops:
        print("\nVerdict: PASS — all runtime ops on ANE")
    else:
        print(f"\nVerdict: FAIL — {len(cpu_ops)} CPU + {len(gpu_ops)} GPU fallback ops")


if __name__ == "__main__":
    main()
