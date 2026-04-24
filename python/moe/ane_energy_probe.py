"""Probe D': energy per token, ANE vs GPU.

Two workload runners (ANE, MPS) that each execute a Gemma-shape SwiGLU expert
in a tight loop for a fixed wall-clock window. Each prints:
    BEGIN_AT <epoch_seconds>
    END_AT   <epoch_seconds>
    ITERS    <count>

Capture power separately in another terminal:

    sudo powermetrics --samplers cpu_power,gpu_power,ane_power \\
        -i 1000 -n 60 --format plist > /tmp/power.plist

Then parse with `python python/moe/ane_energy_parse.py`.

Usage:
    python python/moe/ane_energy_probe.py ane     # 30 s ANE loop
    python python/moe/ane_energy_probe.py mps     # 30 s Metal/MPS loop
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np

D_MODEL = 2304
D_FFN = 9216
DURATION_S = 30.0


def run_ane():
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    from coremltools.converters.mil import Builder as mb

    out = Path(__file__).parent.parent / "tmp" / "energy" / "expert_int4.mlpackage"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        rng = np.random.default_rng(0)
        Wg = rng.standard_normal((D_FFN, D_MODEL)).astype(np.float16)
        Wu = rng.standard_normal((D_FFN, D_MODEL)).astype(np.float16)
        Wd = rng.standard_normal((D_MODEL, D_FFN)).astype(np.float16)
        bg = np.zeros((D_FFN,), dtype=np.float16)
        bu = np.zeros((D_FFN,), dtype=np.float16)
        bd = np.zeros((D_MODEL,), dtype=np.float16)
        fp16 = ct.converters.mil.mil.types.fp16

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, D_MODEL), dtype=fp16)])
        def prog(x):
            g = mb.linear(x=x, weight=Wg, bias=bg, name="gate")
            u = mb.linear(x=x, weight=Wu, bias=bu, name="up")
            sg = mb.silu(x=g, name="silu")
            h = mb.mul(x=sg, y=u, name="mul")
            return mb.linear(x=h, weight=Wd, bias=bd, name="down")

        m = ct.convert(prog, convert_to="mlprogram",
                       compute_precision=ct.precision.FLOAT16,
                       compute_units=ct.ComputeUnit.CPU_AND_NE,
                       minimum_deployment_target=ct.target.macOS15)
        m = cto.linear_quantize_weights(
            m, config=cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(
                    mode="linear_symmetric", weight_threshold=0, dtype="int4")))
        m.save(str(out))

    m = ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_AND_NE)
    in_name = list(m.input_description)[0]
    x = np.zeros((1, D_MODEL), dtype=np.float32)
    # warmup
    for _ in range(50):
        m.predict({in_name: x})
    print("READY: starting ANE loop in 3s...", flush=True)
    time.sleep(3)
    t0 = time.perf_counter()
    print(f"BEGIN_AT {time.time():.3f}", flush=True)
    iters = 0
    while time.perf_counter() - t0 < DURATION_S:
        m.predict({in_name: x})
        iters += 1
    print(f"END_AT   {time.time():.3f}", flush=True)
    print(f"ITERS    {iters}", flush=True)
    print(f"throughput {iters / DURATION_S:.1f} expert-evals/s", flush=True)


def run_mps():
    import torch
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available", file=sys.stderr)
        sys.exit(2)
    dev = torch.device("mps")
    rng = np.random.default_rng(0)
    Wg = torch.from_numpy(rng.standard_normal((D_MODEL, D_FFN)).astype(np.float16)).to(dev)
    Wu = torch.from_numpy(rng.standard_normal((D_MODEL, D_FFN)).astype(np.float16)).to(dev)
    Wd = torch.from_numpy(rng.standard_normal((D_FFN, D_MODEL)).astype(np.float16)).to(dev)
    x = torch.zeros((1, D_MODEL), dtype=torch.float16, device=dev)

    def expert(x):
        g = x @ Wg
        u = x @ Wu
        h = torch.nn.functional.silu(g) * u
        return h @ Wd

    # warmup
    for _ in range(50):
        y = expert(x)
    torch.mps.synchronize()
    print("READY: starting MPS loop in 3s...", flush=True)
    time.sleep(3)
    t0 = time.perf_counter()
    print(f"BEGIN_AT {time.time():.3f}", flush=True)
    iters = 0
    while time.perf_counter() - t0 < DURATION_S:
        y = expert(x)
        iters += 1
    torch.mps.synchronize()
    print(f"END_AT   {time.time():.3f}", flush=True)
    print(f"ITERS    {iters}", flush=True)
    print(f"throughput {iters / DURATION_S:.1f} expert-evals/s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("ane", "mps"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "ane":
        run_ane()
    else:
        run_mps()
