"""ane-validator: residency probe for PF_expert000_layoutfix_{fp16,int4}.mlpackage.

ANE-canonical activation layout: x_in/x_out [1, 640, 1, 64] fp16 (seq on W).
Probes both packages back-to-back to isolate "layout" vs "INT4 quant".
"""
import shutil, sys, time
from pathlib import Path
import numpy as np
import coremltools as ct

BASE = Path("/Users/alvarovidela/Code/em2/emilio/conv-ane")
PKGS = [
    ("FP16", BASE / "PF_expert000_layoutfix_fp16.mlpackage"),
    ("INT4", BASE / "PF_expert000_layoutfix_int4.mlpackage"),
]
TMP = Path("/Users/alvarovidela/Code/em2/python/moe/tmp")
SHAPE = (1, 640, 1, 64)


def probe(label, pkg):
    print(f"\n############ {label}: {pkg.name} ############")
    if not pkg.exists():
        print(f"MISSING: {pkg}")
        return False, None
    t0 = time.time()
    try:
        m = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e:
        print(f"LOAD: FAIL ({e!r})")
        return False, None
    print(f"LOAD: OK ({time.time()-t0:.2f}s)")

    spec = m.get_spec()
    in_name = spec.description.input[0].name
    out_name = spec.description.output[0].name
    print(f"input {in_name}  output {out_name}  shape {SHAPE}")

    x = np.zeros(SHAPE, dtype=np.float16)
    try:
        t0 = time.time()
        y = m.predict({in_name: x})
        cold = (time.time() - t0) * 1000
        out = y[out_name]
        print(f"PREDICT cold: shape={out.shape} dtype={out.dtype}  latency={cold:.2f} ms")
    except Exception as e:
        print(f"PREDICT: FAIL ({e!r})")
        return False, None

    lat = []
    for _ in range(5):
        t0 = time.time(); m.predict({in_name: x}); lat.append((time.time()-t0)*1000)
    print(f"PREDICT (5x warm): min={min(lat):.2f} ms  mean={sum(lat)/len(lat):.2f} ms")

    compiled = TMP / f"pf_layoutfix_{label.lower()}.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)
    compiled.parent.mkdir(parents=True, exist_ok=True)
    try:
        compiled_path = ct.utils.compile_model(str(pkg), str(compiled))
    except Exception as e:
        print(f"COMPILE: FAIL ({e!r})")
        return False, None
    print(f"COMPILE: OK -> {compiled_path}")

    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(
            path=str(compiled_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    except Exception as e:
        print(f"COMPUTE_PLAN: FAIL ({e!r})")
        return False, None

    program = plan.model_structure.program
    if program is None:
        print("COMPUTE_PLAN: no program")
        return False, None

    conv_devs = []
    rows = []
    for fn_name, fn in program.functions.items():
        print(f"\nfunction {fn_name}:")
        print(f"  {'op':<32} {'device':<28} {'cost':>8}")
        for op in fn.block.operations:
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                devn = dev.preferred_compute_device.__class__.__name__ if dev else "?"
                cn = f"{cost.weight:.4f}" if cost is not None else "-"
                print(f"  {op.operator_name:<32} {devn:<28} {cn:>8}")
                rows.append((op.operator_name, devn, cn))
                if "conv" in op.operator_name.lower():
                    conv_devs.append((op.operator_name, devn))
            except Exception as e:
                print(f"  {op.operator_name:<32} ERR {e!r}")

    print(f"\n--- {label} VERDICT ---")
    if not conv_devs:
        print("FAIL: no conv ops")
        return False, (cold, lat, rows, conv_devs)
    all_ane = all("NeuralEngine" in d for _, d in conv_devs)
    for n, d in conv_devs:
        print(f"  {n}: {d}")
    print("PASS" if all_ane else "FAIL")
    return all_ane, (cold, lat, rows, conv_devs)


def main():
    results = {}
    for label, pkg in PKGS:
        ok, info = probe(label, pkg)
        results[label] = (ok, info)

    print("\n================ SUMMARY ================")
    for label, (ok, info) in results.items():
        if info is None:
            print(f"  {label}: ERROR (no info)")
            continue
        cold, lat, rows, conv_devs = info
        wm = min(lat) if lat else float("nan")
        convs = ", ".join(f"{n}->{d}" for n, d in conv_devs)
        print(f"  {label}: {'PASS' if ok else 'FAIL'}  cold={cold:.2f}ms warm_min={wm:.2f}ms  convs: {convs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
