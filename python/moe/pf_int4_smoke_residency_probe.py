"""ane-validator probe for PF_expert000_int4_smoke.mlpackage (single-function CoreML 9).

INT4 per-block (block_size=32) symmetric quantized MoE expert (layer0/expert0).
Op chain: conv1x1 (640->1280) + clip x2 + sigmoid + mul x3 + conv1x1 (640->640).
Pack: ~0.70 MB (FP16 was 2.47 MB). Per-conv weight ~0.5 MB INT4.
This is the T3.5 derisk: do INT4 weight-quantized convs land on ANE here?
"""
import shutil, sys, time
from pathlib import Path
import numpy as np
import coremltools as ct

PKG = Path("/Users/alvarovidela/Code/em2/emilio/conv-ane/PF_expert000_int4_smoke.mlpackage")
COMPILED = Path("/Users/alvarovidela/Code/em2/python/moe/tmp/pf_int4_smoke.mlmodelc")
SHAPE = (64, 640, 1, 1)


def main():
    print(f"package: {PKG}")
    print(f"input shape: {SHAPE} fp16  (single-function 'main')")

    t0 = time.time()
    try:
        m = ct.models.MLModel(str(PKG), compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e:
        print(f"LOAD: FAIL ({e!r})")
        return 1
    print(f"LOAD: OK  ({time.time()-t0:.2f}s)")

    spec = m.get_spec()
    desc = spec.description
    in_name = desc.input[0].name
    out_name = desc.output[0].name
    print(f"input feature: {in_name}   output feature: {out_name}")

    x = np.zeros(SHAPE, dtype=np.float16)

    try:
        t0 = time.time()
        y = m.predict({in_name: x})
        cold = (time.time() - t0) * 1000
        out = y[out_name]
        print(f"PREDICT cold: shape={out.shape} dtype={out.dtype}  latency={cold:.2f} ms")
        print(f"  out stats: min={float(np.min(out)):.4f} max={float(np.max(out)):.4f} mean={float(np.mean(out)):.4f}")
    except Exception as e:
        print(f"PREDICT: FAIL ({e!r})")
        return 2

    lat = []
    for _ in range(5):
        t0 = time.time(); m.predict({in_name: x}); lat.append((time.time()-t0)*1000)
    print(f"PREDICT (5x warm): min={min(lat):.2f} ms  mean={sum(lat)/len(lat):.2f} ms")

    if COMPILED.exists():
        shutil.rmtree(COMPILED)
    COMPILED.parent.mkdir(parents=True, exist_ok=True)
    try:
        compiled_path = ct.utils.compile_model(str(PKG), str(COMPILED))
    except Exception as e:
        print(f"COMPILE: FAIL ({e!r})")
        return 3
    print(f"COMPILE: OK -> {compiled_path}")

    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(
            path=str(compiled_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    except Exception as e:
        print(f"COMPUTE_PLAN: FAIL ({e!r})")
        return 0

    program = plan.model_structure.program
    if program is None:
        print("COMPUTE_PLAN: no program structure")
        return 0

    conv_devs = []
    for fn_name, fn in program.functions.items():
        print(f"\nfunction {fn_name}:")
        print(f"  {'op':<28} {'device':<26} {'cost':>8}")
        for op in fn.block.operations:
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                devn = dev.preferred_compute_device.__class__.__name__ if dev else "?"
                cn = f"{cost.weight:.4f}" if cost is not None else "-"
                print(f"  {op.operator_name:<28} {devn:<26} {cn:>8}")
                if "conv" in op.operator_name.lower():
                    conv_devs.append((op.operator_name, devn))
            except Exception as e:
                print(f"  {op.operator_name:<28} ERR {e!r}")

    print("\n=== VERDICT ===")
    if not conv_devs:
        print("FAIL: no conv ops found in program")
        return 4
    all_ane = all("NeuralEngine" in d for _, d in conv_devs)
    for n, d in conv_devs:
        print(f"  {n}: {d}")
    if all_ane:
        print("PASS: all conv ops on MLNeuralEngineComputeDevice")
        return 0
    else:
        print("FAIL: at least one conv op fell to CPU/GPU")
        return 5


if __name__ == "__main__":
    sys.exit(main())
