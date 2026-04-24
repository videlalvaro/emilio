"""ane-validator probe for PF_experts1_smoke.mlpackage (MultiFunctionDescriptor pack).

Single function `expert_0`, x_in [64,640,1,1] fp16 -> x_out [64,640,1,1] fp16.
Op chain: conv1x1 (640->1280) + clamp + sigmoid + mul + conv1x1 (640->640).
~1.6 MB weights => below 12 MB ANE residency floor; CPU placement is acceptable.
We mostly want: load + predict round-trip + per-op placement table.
"""
import shutil, sys, time
from pathlib import Path
import numpy as np
import coremltools as ct

PKG = Path("/Users/alvarovidela/Code/em2/emilio/conv-ane/PF_experts1_smoke.mlpackage")
COMPILED = Path("/Users/alvarovidela/Code/em2/python/moe/tmp/pf_smoke.mlmodelc")
FN = "expert_0"
SHAPE = (64, 640, 1, 1)


def main():
    print(f"package: {PKG}")
    print(f"function: {FN}")
    print(f"input shape: {SHAPE} fp16")

    # 1) Load test
    t0 = time.time()
    try:
        m = ct.models.MLModel(
            str(PKG),
            function_name=FN,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    except Exception as e:
        print(f"LOAD: FAIL ({e!r})")
        return 1
    print(f"LOAD: OK  ({time.time()-t0:.2f}s)")

    # 2) Predict round-trip
    x = np.random.default_rng(0).standard_normal(SHAPE).astype(np.float16)
    spec = m.get_spec()
    # MFD: inputs/outputs live under spec.description.functions[fn].
    desc = spec.description
    fn_desc = None
    if hasattr(desc, "functions") and len(desc.functions) > 0:
        for f in desc.functions:
            if f.name == FN:
                fn_desc = f
                break
        if fn_desc is None:
            fn_desc = desc.functions[0]
        in_name = fn_desc.input[0].name
        out_name = fn_desc.output[0].name
    else:
        in_name = desc.input[0].name
        out_name = desc.output[0].name
    print(f"input feature name: {in_name}")
    print(f"output feature name: {out_name}")

    try:
        t0 = time.time()
        y = m.predict({in_name: x})
        dt = (time.time() - t0) * 1000
        out = y[out_name]
        print(f"PREDICT: OK  shape={out.shape} dtype={out.dtype}  latency={dt:.2f} ms")
        print(f"  out stats: min={float(np.min(out)):.4f} max={float(np.max(out)):.4f} mean={float(np.mean(out)):.4f}")
    except Exception as e:
        print(f"PREDICT: FAIL ({e!r})")
        return 2

    # warm + repeat for steadier latency
    lat = []
    for _ in range(5):
        t0 = time.time(); m.predict({in_name: x}); lat.append((time.time()-t0)*1000)
    print(f"PREDICT (5x warm): min={min(lat):.2f} ms  mean={sum(lat)/len(lat):.2f} ms")

    # 3) Compute-plan per-op placement
    if COMPILED.exists():
        shutil.rmtree(COMPILED)
    COMPILED.parent.mkdir(parents=True, exist_ok=True)
    try:
        compiled_path = ct.utils.compile_model(str(PKG), str(COMPILED))
    except Exception as e:
        print(f"COMPILE: FAIL ({e!r})")
        return 3
    print(f"COMPILE: OK  -> {compiled_path}")

    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(
            path=str(compiled_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            function_name=FN,
        )
    except TypeError:
        # older signature without function_name
        plan = MLComputePlan.load_from_path(
            path=str(compiled_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    except Exception as e:
        print(f"COMPUTE_PLAN: FAIL ({e!r})")
        return 0  # not fatal

    program = plan.model_structure.program
    if program is None:
        print("COMPUTE_PLAN: no program structure (not mlprogram?)")
        return 0
    for fn_name, fn in program.functions.items():
        print(f"\nfunction {fn_name}:")
        print(f"  {'op':<28} {'device':<22} {'cost':>8}")
        for op in fn.block.operations:
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                if dev is None:
                    devn = "?"
                else:
                    devn = dev.preferred_compute_device.__class__.__name__
                cn = f"{cost.weight:.4f}" if cost is not None else "-"
                print(f"  {op.operator_name:<28} {devn:<22} {cn:>8}")
            except Exception as e:
                print(f"  {op.operator_name:<28} ERR {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
