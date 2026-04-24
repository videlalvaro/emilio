"""Sweep N-function multi-function .mlpackages to find the procedure cap.

Hypothesis from `_ANEModelInstanceParameters` and `VirtANEModel`'s `[64 I]`
arrays: the firmware caps procedures per loaded model at 64.

Empirical test: build N tiny experts in one .mlpackage, load each with
function_name and run on CPU_AND_NE. First N where load or predict fails
is the cap.
"""
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

OUT_DIR = Path(__file__).parent.parent / "tmp" / "mfn_cap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_one_expert(expert_id: int, out_path: Path):
    rng = np.random.default_rng(seed=expert_id)
    W = rng.standard_normal((8, 8)).astype(np.float16)
    b = np.zeros((8,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 8), dtype=ct.converters.mil.mil.types.fp16)])
    def prog(x):
        y = mb.linear(x=x, weight=W, bias=b, name="lin")
        z = mb.relu(x=y, name="out")
        return z

    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.save(str(out_path))


def try_n(n: int) -> dict:
    work = OUT_DIR / f"n{n}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    parts = []
    t0 = time.time()
    for i in range(n):
        p = work / f"expert_{i}.mlpackage"
        build_one_expert(i, p)
        parts.append(p)
    t_build = time.time() - t0

    multi_path = work / "experts_multi.mlpackage"
    desc = MultiFunctionDescriptor()
    for i, p in enumerate(parts):
        desc.add_function(str(p), "main", f"expert_{i}")
    desc.default_function_name = "expert_0"

    t1 = time.time()
    save_multifunction(desc, str(multi_path))
    t_save = time.time() - t1

    # Try to load + predict each function. Note: this dispatches potentially N model
    # loads. If the daemon caps at 64 procedures we'll see a failure on load OR predict.
    x = np.arange(8, dtype=np.float16).reshape(1, 8).astype(np.float32)
    failures = []
    t2 = time.time()
    for i in range(n):
        try:
            m = ct.models.MLModel(
                str(multi_path),
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                function_name=f"expert_{i}",
            )
            in_name = list(m.input_description)[0] if list(m.input_description) else "x"
            out = m.predict({in_name: x})
            out_name = next(iter(out))
            _ = np.array(out[out_name]).reshape(-1)[:1]
        except Exception as e:
            failures.append((i, repr(e)[:160]))
            if len(failures) >= 3:
                break
    t_run = time.time() - t2

    pkg_size = sum(p.stat().st_size for p in multi_path.rglob("*") if p.is_file())

    return {
        "n": n,
        "build_s": round(t_build, 1),
        "save_s": round(t_save, 1),
        "run_s": round(t_run, 1),
        "pkg_MB": round(pkg_size / 1e6, 2),
        "ok": len(failures) == 0,
        "failures": failures,
    }


def main():
    sweep = [4, 16, 32, 64, 96, 128, 192, 256]
    results = []
    for n in sweep:
        print(f"\n=== N={n} ===", flush=True)
        try:
            r = try_n(n)
        except Exception as e:
            print("BUILD/SAVE FAILED:", repr(e))
            traceback.print_exc()
            results.append({"n": n, "fatal": repr(e)[:200]})
            break
        print(r)
        results.append(r)
        if not r["ok"]:
            print(f"  -> first failures: {r['failures']}")
            # don't break: maybe predict failed transiently. Continue to see pattern.

    print("\n=== SUMMARY ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
