"""Probe 1: ANE effective bandwidth.

Build models that are pure weight-streaming: a single linear op of varying
constant size, run repeatedly. Slope of (time vs bytes) gives effective BW
visible to ANE on this M4 Max.

We sweep weight sizes from 1 MB to ~256 MB. Bigger than that and the .mlpackage
build itself starts to dominate.
"""
import shutil
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

OUT_DIR = Path(__file__).parent.parent / "tmp" / "bw_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_pure_linear(in_dim: int, out_dim: int, out_path: Path):
    """y = x @ W where W is (in_dim, out_dim) fp16. No bias, no activation."""
    rng = np.random.default_rng(seed=0)
    # mb.linear expects weight shape (out_dim, in_dim)
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


def time_predict(model, x, in_name, n_iter=200, warmup=20):
    for _ in range(warmup):
        model.predict({in_name: x})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.predict({in_name: x})
    return (time.perf_counter() - t0) / n_iter


def main():
    # (in_dim, out_dim, label_MB).  weight bytes = in*out*2 (fp16).
    # 1 MB -> 1024*512  ; 4 MB -> 2048*1024 ; 16 MB -> 4096*2048 ;
    # 64 MB -> 8192*4096 ; 256 MB -> 16384*8192
    cases = [
        (1024, 512),    # 1 MB
        (2048, 1024),   # 4 MB
        (4096, 2048),   # 16 MB
        (8192, 4096),   # 64 MB
        (16384, 8192),  # 256 MB
    ]

    print(f"{'W_MB':>8} {'in':>6} {'out':>6} {'lat_ms':>9} {'GB/s':>8}")
    rows = []
    for in_dim, out_dim in cases:
        w_bytes = in_dim * out_dim * 2  # fp16
        w_mb = w_bytes / 1e6
        path = OUT_DIR / f"linear_{in_dim}x{out_dim}.mlpackage"
        if path.exists():
            shutil.rmtree(path)
        build_pure_linear(in_dim, out_dim, path)
        m = ct.models.MLModel(
            str(path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        in_name = list(m.input_description)[0]
        x = np.zeros((1, in_dim), dtype=np.float32)
        lat = time_predict(m, x, in_name, n_iter=100, warmup=10)
        gbps = (w_bytes / 1e9) / lat
        print(f"{w_mb:>8.1f} {in_dim:>6} {out_dim:>6} {lat*1e3:>9.3f} {gbps:>8.1f}")
        rows.append((w_mb, lat * 1e3, gbps))

    print("\n=== summary ===")
    for w_mb, lat_ms, gbps in rows:
        print(f"  {w_mb:7.1f} MB   {lat_ms:7.3f} ms   {gbps:6.1f} GB/s")


if __name__ == "__main__":
    main()
