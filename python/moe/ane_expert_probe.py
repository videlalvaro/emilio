"""Probe 3: realistic Gemma 4 expert-sized timing.

One Gemma 4 26B-A4B expert ≈ 4M params (active=4B / 128 experts × 30 layers
implied by the A4B math; in practice each expert MLP at hidden=2048,
ffn=8192 is ~50M params). Use both shapes and INT4 storage size to bracket.

We build:
  - "small" : 2048 -> 8192 -> 2048 (gate+up+down style as a single linear chain)
  - "large" : per-expert ffn at fp16 weight = ~50 MB
At INT4 these would be ~12.5 MB each, but coremltools fp16 weights are the
fair stand-in for a measured ANE BW comparison (we know int4 unpacks to fp16
on the wire to the engine).

Then estimate per-token decode latency for a Gemma-shaped MoE:
  per-layer-cost = attention_cost + 8 * expert_cost + router_cost
  per-token-cost = 30 * per-layer-cost
"""
import shutil
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

OUT_DIR = Path(__file__).parent.parent / "tmp" / "expert_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_expert_mlp(d_model: int, d_ffn: int, out_path: Path):
    """A Gemma-style SwiGLU MLP block:
       gate = x @ Wg
       up   = x @ Wu
       y    = (silu(gate) * up) @ Wd
    All fp16. Returns a single-input single-output mlprogram."""
    rng = np.random.default_rng(0)
    # mb.linear expects weight shape (out_dim, in_dim)
    Wg = rng.standard_normal((d_ffn, d_model)).astype(np.float16)
    Wu = rng.standard_normal((d_ffn, d_model)).astype(np.float16)
    Wd = rng.standard_normal((d_model, d_ffn)).astype(np.float16)
    bg = np.zeros((d_ffn,), dtype=np.float16)
    bu = np.zeros((d_ffn,), dtype=np.float16)
    bd = np.zeros((d_model,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, d_model), dtype=ct.converters.mil.mil.types.fp16)])
    def prog(x):
        g = mb.linear(x=x, weight=Wg, bias=bg, name="gate")
        u = mb.linear(x=x, weight=Wu, bias=bu, name="up")
        sg = mb.silu(x=g, name="silu")
        h = mb.mul(x=sg, y=u, name="mul")
        y = mb.linear(x=h, weight=Wd, bias=bd, name="down")
        return y

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    m.save(str(out_path))


def time_predict(m, x, in_name, n_iter=200, warmup=20):
    for _ in range(warmup):
        m.predict({in_name: x})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        m.predict({in_name: x})
    return (time.perf_counter() - t0) / n_iter


def measure(d_model, d_ffn, label):
    weight_bytes = (d_model * d_ffn * 2 + d_model * d_ffn * 2 + d_ffn * d_model * 2)
    weight_MB = weight_bytes / 1e6
    path = OUT_DIR / f"expert_{label}.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    print(f"\n[{label}] d_model={d_model} d_ffn={d_ffn}  weights={weight_MB:.1f} MB (fp16)")
    build_expert_mlp(d_model, d_ffn, path)
    m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    in_name = list(m.input_description)[0]
    x = np.zeros((1, d_model), dtype=np.float32)
    lat = time_predict(m, x, in_name, n_iter=100, warmup=10)
    gbps = weight_bytes / 1e9 / lat
    print(f"   per-call latency: {lat*1e3:.3f} ms   ({gbps:.1f} GB/s effective)")
    return weight_MB, lat


def main():
    # Bracket Gemma 4 expert shape uncertainty.
    # Gemma 4 26B-A4B has 4B active params over 30 layers x 8 active experts.
    # That's ~4B/(30*8) = ~16M params per active expert call worth of MLP-related streaming.
    # Per-expert MLP at typical Gemma sizes:
    #   d_model=2304, d_ffn ≈ 4*d_model=9216 -> 3 mats * 2304*9216*2B ≈ 127 MB at fp16
    # At INT4 storage the on-disk is ~32 MB but the engine likely materializes fp16 internally.
    # We measure both fp16 (worst-case BW) and a smaller proxy.
    results = []
    results.append(("tiny",   measure(1024, 4096, "tiny")))    # ~25 MB
    results.append(("medium", measure(2048, 4096, "medium")))  # ~50 MB
    results.append(("gemma",  measure(2304, 9216, "gemma")))   # ~127 MB

    print("\n=== Gemma 4 26B-A4B per-token decode projection ===")
    n_layers = 30
    n_active_experts = 8
    # Pick the gemma row.
    _, (_, expert_lat_s) = results[-1]
    per_layer_ms = (expert_lat_s * n_active_experts) * 1e3
    per_token_ms = per_layer_ms * n_layers
    tps = 1000.0 / per_token_ms
    print(f"  expert latency (fp16 weights):    {expert_lat_s*1e3:.2f} ms")
    print(f"  per-layer (8 active experts):     {per_layer_ms:.1f} ms")
    print(f"  per-token (30 layers, MLP only):  {per_token_ms:.0f} ms")
    print(f"  decode throughput (MLP-only):     ~{tps:.1f} tok/s")
    print("  NOTE: ignores attention cost; fp16 weights are upper-bound BW")
    print("        true INT4 should be ~4x less data = ~4x faster")


if __name__ == "__main__":
    main()
