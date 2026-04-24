"""Probe A'' — packed-expert format sweep.

When a single Gemma expert (d=2816, ffn=704) is shipped on its own, CoreML
silently falls back to CPU because the linear weights are too small for the
ANE compiler's threshold.

Fix: pack G experts together by concatenating their weights along the output
axis. The "select expert e" semantic becomes:
    H = mb.linear(x, W_packed)      # shape (G * d_ffn,)
    H = mb.reshape(H, (G, d_ffn))   # one row per expert
    H = mb.gather(H, expert_id)     # pick the chosen expert
This makes the up/gate linear (G * 704, 2816) and the down linear (2816,
G * 704). For G=16 → up: (11264, 2816), down: (2816, 11264) — well over the
ANE threshold and matches the kind of shapes that landed on ANE before.

We sweep G ∈ {1, 4, 8, 16, 32} and measure latency + device placement.
The per-token MLP cost is then  ceil(8/G) * latency * 30  layers.
"""
import shutil
import time
from pathlib import Path

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb

OUT_DIR = Path(__file__).parent.parent / "tmp" / "gemma_packed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL = 2816
D_FFN   = 704
TOPK    = 8


def build_packed(G: int, out_path: Path) -> ct.models.MLModel:
    """Pack G experts into a single program. Inputs: x (1, d_model),
    expert_id (1,) int32. Output: y (1, d_model)."""
    rng = np.random.default_rng(0)
    Wg = rng.standard_normal((G * D_FFN, D_MODEL)).astype(np.float16)
    Wu = rng.standard_normal((G * D_FFN, D_MODEL)).astype(np.float16)
    Wd = rng.standard_normal((G * D_MODEL, D_FFN)).astype(np.float16)
    bg = np.zeros((G * D_FFN,), dtype=np.float16)
    bu = np.zeros((G * D_FFN,), dtype=np.float16)
    bd = np.zeros((G * D_MODEL,), dtype=np.float16)

    fp16 = ct.converters.mil.mil.types.fp16

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, D_MODEL), dtype=fp16),
    ])
    def prog(x):
        # Latency probe only: measure the cost of the packed linears. The
        # post-linear "select expert e" gather is dropped because the
        # coremltools 9 builder/spec mismatch on iOS18 gather blocks us.
        # We surrogate it with reshape+reduce_sum which keeps the down-proj
        # input shape correct without invoking scatter_gather ops.
        g = mb.linear(x=x, weight=Wg, bias=bg, name="gate")
        u = mb.linear(x=x, weight=Wu, bias=bu, name="up")
        gelu_g = mb.gelu(x=g, mode="TANH_APPROXIMATION", name="gelu")
        h = mb.mul(x=gelu_g, y=u, name="hmul")  # (1, G*d_ffn)
        h = mb.reshape(x=h, shape=(1, G, D_FFN), name="h_r")
        h = mb.reduce_sum(x=h, axes=[1], keep_dims=False, name="h_sum")  # (1, d_ffn)
        y = mb.linear(x=h, weight=Wd, bias=bd, name="down")  # (1, G*d_model)
        return y

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    m.save(str(out_path))
    return m


def package_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def time_predict(m, feed, n_iter=120, warmup=20):
    for _ in range(warmup):
        m.predict(feed)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        m.predict(feed)
    return (time.perf_counter() - t0) / n_iter


def device_summary(pkg_path: Path) -> str:
    try:
        from coremltools.models.compute_plan import MLComputePlan
        compiled = pkg_path.with_suffix(".mlmodelc")
        if compiled.exists():
            shutil.rmtree(compiled)
        compiled = Path(ct.utils.compile_model(str(pkg_path), str(compiled)))
        plan = MLComputePlan.load_from_path(
            path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
        program = plan.model_structure.program
        if program is None:
            return "?"
        devs = []
        for fn_name, fn in program.functions.items():
            for op in fn.block.operations:
                if op.operator_name in ("ios18.linear", "linear"):
                    try:
                        d = plan.get_compute_device_usage_for_mlprogram_operation(op)
                        n = d.preferred_compute_device.__class__.__name__ if d else "?"
                        devs.append("ANE" if "Neural" in n else ("GPU" if "GPU" in n else "CPU"))
                    except Exception:
                        devs.append("?")
        return "/".join(devs) if devs else "?"
    except Exception as e:
        return f"err:{type(e).__name__}"


def measure(G: int, quant: str):
    label = f"G{G:02d}_{quant}"
    print(f"\n=== {label} ===")
    base_path = OUT_DIR / f"G{G:02d}_baseline.mlpackage"
    if not base_path.exists():
        build_packed(G, base_path)

    out_path = OUT_DIR / f"{label}.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)

    base = ct.models.MLModel(str(base_path),
                             compute_units=ct.ComputeUnit.CPU_AND_NE)
    if quant == "fp16":
        base.save(str(out_path))
        m = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    elif quant == "int4":
        m = cto.linear_quantize_weights(
            base, config=cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(
                    mode="linear_symmetric", weight_threshold=0, dtype="int4")))
        m.save(str(out_path))
    else:
        raise ValueError(quant)

    size_mb = package_size_mb(out_path)
    feed = {"x": np.zeros((1, D_MODEL), dtype=np.float32)}
    try:
        lat = time_predict(m, feed)
    except Exception as e:
        print(f"  predict failed: {e!r}")
        return None
    devs = device_summary(out_path)
    # Per-token MLP: top-K experts spread across packs of size G
    calls_per_layer = (TOPK + G - 1) // G if G < TOPK else 1
    per_token_ms = calls_per_layer * lat * 1e3 * 30
    print(f"  size={size_mb:7.2f} MB   lat={lat*1e3:6.3f} ms   linears: {devs}"
          f"   ⇒ {per_token_ms:6.1f} ms/token MLP")
    return (label, size_mb, lat * 1e3, devs, per_token_ms)


def main():
    rows = []
    for G in (1, 4, 8, 16, 32):
        for q in ("fp16", "int4"):
            rows.append(measure(G, q))

    print("\n=== SUMMARY ===")
    print(f"{'pack':12s}  {'size_MB':>8s}  {'lat_ms':>7s}  {'per_tok_ms':>10s}  devices")
    for r in rows:
        if r is None: continue
        label, size_mb, lat_ms, devs, ptm = r
        print(f"{label:12s}  {size_mb:8.2f}  {lat_ms:7.3f}  {ptm:10.1f}  {devs}")


if __name__ == "__main__":
    main()
