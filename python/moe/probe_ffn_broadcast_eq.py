"""ANE residency probe for FFN shard with broadcast == routing.

Checks per-op compute-unit placement via MLComputePlan for the compiled
gemma4_shard0_1_real_ffn_p0of2_q8 model.
"""
import sys
from pathlib import Path
import coremltools as ct

COMPILED = Path(__file__).resolve().parent / "out" / "gemma4_shard0_1_real_ffn_p0of2_q8.mlmodelc"
PACKAGE = COMPILED.with_suffix(".mlpackage")


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

    print(f"{'op_name':<60} {'op_type':<30} {'device':<20}")
    print("-" * 110)

    cpu_ops = []
    gpu_ops = []
    total = 0

    def _op_name(op):
        if op.outputs:
            return op.outputs[0].name
        return "?"

    for fn_name, fn in program.functions.items():
        for op in fn.block.operations:
            op_type = op.operator_name
            # skip const ops — they are just weight/param declarations
            if op_type in ("const", "constexpr_blockwise_shift_scale"):
                continue
            total += 1
            name = _op_name(op)
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if dev is None:
                    dev_name = "unknown"
                else:
                    dev_name = dev.preferred_compute_device.__class__.__name__
            except Exception as e:
                dev_name = f"error:{e!r}"

            # Map class names
            if "NeuralEngine" in dev_name:
                label = "ANE"
            elif "CPU" in dev_name or "cpu" in dev_name.lower():
                label = "CPU"
                cpu_ops.append((name, op_type))
            elif "GPU" in dev_name or "gpu" in dev_name.lower():
                label = "GPU"
                gpu_ops.append((name, op_type))
            else:
                label = dev_name

            print(f"{name:<60} {op_type:<30} {label:<20}")

    print("-" * 110)
    print(f"Total compute ops: {total}")
    print(f"CPU fallbacks:     {len(cpu_ops)}")
    print(f"GPU fallbacks:     {len(gpu_ops)}")
    if cpu_ops:
        print("\nCPU ops:")
        for name, ot in cpu_ops:
            print(f"  {name}: {ot}")
    if gpu_ops:
        print("\nGPU ops:")
        for name, ot in gpu_ops:
            print(f"  {name}: {ot}")

    if cpu_ops or gpu_ops:
        print("\nVERDICT: FAIL — ops fell back to CPU/GPU")
        sys.exit(1)
    else:
        print("\nVERDICT: PASS — all compute ops on ANE")


if __name__ == "__main__":
    main()
