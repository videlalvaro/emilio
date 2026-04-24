"""ane-validator probe for PF_experts128_layer0.mlpackage (full 128-fn MFD pack).

301 MB FP16 pack, expected CPU placement (above 96 MB ANE cliff).
Probes load+predict for expert_000 and expert_064; per-op placement table.
"""
import shutil, sys, time
from pathlib import Path
import numpy as np
import coremltools as ct

PKG = Path("/Users/alvarovidela/Code/em2/emilio/conv-ane/PF_experts128_layer0.mlpackage")
COMPILED = Path("/Users/alvarovidela/Code/em2/python/moe/tmp/pf_experts128.mlmodelc")
SHAPE = (64, 640, 1, 1)
FNS = ["expert_000", "expert_064"]


def load_and_predict(fn: str, x: np.ndarray, do_warm: bool = True):
    print(f"\n=== function {fn} ===")
    t0 = time.time()
    try:
        m = ct.models.MLModel(
            str(PKG),
            function_name=fn,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    except Exception as e:
        print(f"LOAD: FAIL ({e!r})")
        return None, None, None
    load_dt = time.time() - t0
    print(f"LOAD: OK  ({load_dt:.2f}s)")

    spec = m.get_spec()
    desc = spec.description
    fn_desc = None
    if hasattr(desc, "functions") and len(desc.functions) > 0:
        for f in desc.functions:
            if f.name == fn:
                fn_desc = f; break
        if fn_desc is None:
            fn_desc = desc.functions[0]
        in_name = fn_desc.input[0].name
        out_name = fn_desc.output[0].name
    else:
        in_name = desc.input[0].name
        out_name = desc.output[0].name

    try:
        t0 = time.time()
        y = m.predict({in_name: x})
        cold_ms = (time.time() - t0) * 1000
        out = y[out_name]
        print(f"PREDICT cold: OK  shape={out.shape} dtype={out.dtype}  {cold_ms:.2f} ms")
    except Exception as e:
        print(f"PREDICT: FAIL ({e!r})")
        return load_dt, None, None

    warm = []
    if do_warm:
        for _ in range(5):
            t0 = time.time(); m.predict({in_name: x}); warm.append((time.time()-t0)*1000)
        print(f"PREDICT warm 5x: min={min(warm):.2f} mean={sum(warm)/len(warm):.2f} ms")
    return load_dt, cold_ms, warm


def per_op_placement(compiled_path: Path, fn: str):
    from coremltools.models.compute_plan import MLComputePlan
    try:
        plan = MLComputePlan.load_from_path(
            path=str(compiled_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            function_name=fn,
        )
    except TypeError:
        plan = MLComputePlan.load_from_path(
            path=str(compiled_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    program = plan.model_structure.program
    if program is None:
        print("COMPUTE_PLAN: no program structure")
        return {}
    counts = {}
    for fn_name, fnobj in program.functions.items():
        if fn_name != fn:
            continue
        print(f"\nfunction {fn_name}:")
        print(f"  {'op':<28} {'device':<22} {'cost':>8}")
        for op in fnobj.block.operations:
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                devn = dev.preferred_compute_device.__class__.__name__ if dev is not None else "?"
                cn = f"{cost.weight:.4f}" if cost is not None else "-"
                print(f"  {op.operator_name:<28} {devn:<22} {cn:>8}")
                counts[devn] = counts.get(devn, 0) + 1
            except Exception as e:
                print(f"  {op.operator_name:<28} ERR {e!r}")
    return counts


def main():
    print(f"package: {PKG}")
    sz_mb = sum(p.stat().st_size for p in PKG.rglob("*") if p.is_file()) / (1024*1024)
    print(f"package size: {sz_mb:.2f} MB")
    print(f"input shape: {SHAPE} fp16")

    x = np.random.default_rng(0).standard_normal(SHAPE).astype(np.float16)

    results = {}
    for fn in FNS:
        results[fn] = load_and_predict(fn, x)

    # Compile once, query plan per function
    if COMPILED.exists():
        shutil.rmtree(COMPILED)
    COMPILED.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nCOMPILE: {PKG.name} -> {COMPILED}")
    t0 = time.time()
    try:
        compiled_path = ct.utils.compile_model(str(PKG), str(COMPILED))
    except Exception as e:
        print(f"COMPILE: FAIL ({e!r})")
        return 3
    print(f"COMPILE: OK  ({time.time()-t0:.2f}s)  -> {compiled_path}")

    all_counts = {}
    for fn in FNS:
        all_counts[fn] = per_op_placement(Path(compiled_path), fn)

    print("\n=== summary ===")
    for fn, c in all_counts.items():
        print(f"  {fn}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
