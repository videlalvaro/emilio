"""Probe 2: cached MLModel vs fresh re-instantiation.

If `MLModel(path, function_name="expert_k")` always re-streams weights from
disk, then the multi-function plan dies — 60 cached instances would each cost
a full load. If the daemon shares a resident _ANEModel across instances, we
can cache cheaply.

Method: compare per-call latency for
  (A) one MLModel instance, predict 200x
  (B) re-instantiate MLModel for each of 200 predicts
and report the delta as cold-load amortized cost.
"""
import shutil
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

OUT_DIR = Path(__file__).parent.parent / "tmp" / "cache_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_one(in_dim, out_dim, out_path):
    rng = np.random.default_rng(0)
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float16)
    b = np.zeros((out_dim,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, in_dim), dtype=ct.converters.mil.mil.types.fp16)])
    def prog(x):
        return mb.linear(x=x, weight=W, bias=b, name="out")

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    m.save(str(out_path))


def main():
    # ~16 MB weights — small enough to converge fast but big enough to feel a re-load.
    in_dim, out_dim = 4096, 2048
    path = OUT_DIR / "lin.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    build_one(in_dim, out_dim, path)

    n_iter = 200
    x = np.zeros((1, in_dim), dtype=np.float32)

    # (A) Cached instance.
    m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    in_name = list(m.input_description)[0]
    for _ in range(20):
        m.predict({in_name: x})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        m.predict({in_name: x})
    cached = (time.perf_counter() - t0) / n_iter

    # (B) Fresh instance per call.
    t0 = time.perf_counter()
    for _ in range(n_iter):
        mm = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        in_name2 = list(mm.input_description)[0]
        mm.predict({in_name2: x})
    fresh = (time.perf_counter() - t0) / n_iter

    # (C) Just the load (no predict)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_only = (time.perf_counter() - t0) / n_iter

    print(f"cached predict:        {cached*1e3:8.3f} ms")
    print(f"fresh load + predict:  {fresh*1e3:8.3f} ms")
    print(f"load only (no predict):{load_only*1e3:8.3f} ms")
    print(f"implied cold-load tax: {(fresh - cached)*1e3:8.3f} ms")


if __name__ == "__main__":
    main()
