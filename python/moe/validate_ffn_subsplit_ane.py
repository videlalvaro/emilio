"""ANE residency validator for Gemma-4 INT8 FFN sub-split artifacts.

Uses MLComputePlan (macOS 14.4+) to inspect per-op compute-unit placement
on pre-compiled .mlmodelc bundles. Reports PASS/FAIL per artifact.

Run with Xcode python3 (only env with coremltools 9):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 python/moe/validate_ffn_subsplit_ane.py
"""
import sys
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

ARTIFACTS = [
    ("ffn_p0of2", "gemma4_shard0_1_real_ffn_p0of2_q8.mlmodelc"),
    ("ffn_combine", "gemma4_shard0_1_real_ffn_combine_q8.mlmodelc"),
    ("attn", "gemma4_shard0_1_real_attn_q8.mlmodelc"),
]

OUT_DIR = Path(__file__).parent / "out"


def inspect_artifact(label: str, filename: str) -> bool:
    path = OUT_DIR / filename
    if not path.exists():
        print(f"\n# {label}: SKIP — {path} not found")
        return True  # don't fail on missing

    print(f"\n# {label}: {path.name}  ({path.stat().st_size / 1e6:.0f} MB)")

    plan = MLComputePlan.load_from_path(
        path=str(path),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    if program is None:
        print("  ERROR: no program structure")
        return False

    total_ops = 0
    ane_ops = 0
    cpu_ops = 0
    gpu_ops = 0
    unknown_ops = 0
    failures = []

    for fn_name, fn in program.functions.items():
        for op in fn.block.operations:
            total_ops += 1
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if dev is None:
                    dev_name = "unknown"
                    unknown_ops += 1
                else:
                    dev_name = dev.preferred_compute_device.__class__.__name__
                    if "NeuralEngine" in dev_name:
                        ane_ops += 1
                    elif "CPU" in dev_name.upper():
                        cpu_ops += 1
                        failures.append(op.operator_name)
                    elif "GPU" in dev_name.upper():
                        gpu_ops += 1
                        failures.append(op.operator_name)
                    else:
                        unknown_ops += 1
            except Exception as e:
                unknown_ops += 1
                dev_name = f"error({e!r})"

            print(f"  {op.operator_name:30s} -> {dev_name}")

    print(f"\n  Total ops: {total_ops}")
    print(f"  ANE: {ane_ops}  CPU: {cpu_ops}  GPU: {gpu_ops}  unknown: {unknown_ops}")

    if cpu_ops > 0 or gpu_ops > 0:
        print(f"  VERDICT: FAIL — non-ANE ops: {failures}")
        return False
    else:
        print(f"  VERDICT: PASS — 100% ANE")
        return True


def main():
    all_pass = True
    for label, filename in ARTIFACTS:
        ok = inspect_artifact(label, filename)
        if not ok:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("OVERALL: PASS — all artifacts 100% ANE")
    else:
        print("OVERALL: FAIL — some artifacts have non-ANE ops")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
