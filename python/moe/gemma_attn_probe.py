"""Probe B (Gemma-real): attention block decode latency for both regimes.

Real Gemma-4-26B-A4B attention from config.json:

  Common
    d_model            = 2816
    n_heads            = 16
    attention_k_eq_v   = True       (single proj used as both K and V)

  Sliding attention (24/30 layers, theta=10000, default RoPE)
    n_kv_heads = 8     head_dim = 256
    KV cache capped at sliding_window = 1024 tokens

  Global attention (6/30 layers, theta=1e6, partial_rotary_factor=0.25)
    n_kv_heads = 2     head_dim = 512
    KV cache = full sequence length

We measure each regime separately at relevant seq_lens. INT4 weight quant
on all linears. Stateless inputs for benchmarking (KV passed in).
"""
import shutil
import time
from pathlib import Path

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb

OUT_DIR = Path(__file__).parent.parent / "tmp" / "gemma_attn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL = 2816
N_HEADS = 16


def build_attn(name: str, n_kv_heads: int, head_dim: int, seq_len: int,
               out_path: Path):
    KV_DIM = n_kv_heads * head_dim
    Q_DIM  = N_HEADS * head_dim
    SCALE  = float(head_dim ** -0.5)

    rng = np.random.default_rng(0)
    Wq  = rng.standard_normal((Q_DIM, D_MODEL)).astype(np.float16)
    Wkv = rng.standard_normal((KV_DIM, D_MODEL)).astype(np.float16)  # k_eq_v
    Wo  = rng.standard_normal((D_MODEL, Q_DIM)).astype(np.float16)
    bq  = np.zeros((Q_DIM,),  dtype=np.float16)
    bkv = np.zeros((KV_DIM,), dtype=np.float16)
    bo  = np.zeros((D_MODEL,), dtype=np.float16)

    fp16 = ct.converters.mil.mil.types.fp16

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, D_MODEL), dtype=fp16),
        mb.TensorSpec(shape=(n_kv_heads, seq_len, head_dim), dtype=fp16),
        mb.TensorSpec(shape=(n_kv_heads, seq_len, head_dim), dtype=fp16),
    ])
    def prog(x, K, V):
        q = mb.linear(x=x, weight=Wq, bias=bq, name="q_proj")
        # k_eq_v: same projection weights drive both K and V at the new token.
        # We compute it twice symbolically (CoreML will optimize/reuse).
        _kv_now = mb.linear(x=x, weight=Wkv, bias=bkv, name="kv_proj")
        # (We don't actually append _kv_now to K/V here — KV is passed in
        # already up-to-date for the benchmark.)
        q = mb.reshape(x=q, shape=(N_HEADS, 1, head_dim), name="q_r")
        repeats = N_HEADS // n_kv_heads
        if repeats > 1:
            Kt = mb.tile(x=K, reps=(repeats, 1, 1), name="K_tile")
            Vt = mb.tile(x=V, reps=(repeats, 1, 1), name="V_tile")
        else:
            Kt, Vt = K, V
        scores = mb.matmul(x=q, y=Kt, transpose_y=True, name="qk")
        scores = mb.mul(x=scores, y=np.float16(SCALE), name="scale")
        attn = mb.softmax(x=scores, axis=-1, name="softmax")
        ctx = mb.matmul(x=attn, y=Vt, name="ctx")
        ctx = mb.reshape(x=ctx, shape=(1, Q_DIM), name="ctx_r")
        # add the kv_proj into the residual to keep the op alive (otherwise
        # CoreML may dead-code-eliminate it and we'd under-measure)
        ctx = mb.add(x=ctx, y=mb.reduce_sum(x=_kv_now, axes=[-1], keep_dims=True), name="keepalive")
        y = mb.linear(x=ctx, weight=Wo, bias=bo, name="o_proj")
        return y

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    m = cto.linear_quantize_weights(
        m, config=cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", weight_threshold=0, dtype="int4")))
    m.save(str(out_path))


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
        counts = {"ANE": 0, "GPU": 0, "CPU": 0, "?": 0}
        for fn in program.functions.values():
            for op in fn.block.operations:
                try:
                    d = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    n = d.preferred_compute_device.__class__.__name__ if d else "?"
                    if "Neural" in n: counts["ANE"] += 1
                    elif "GPU" in n:  counts["GPU"] += 1
                    elif "CPU" in n:  counts["CPU"] += 1
                    else:             counts["?"] += 1
                except Exception:
                    counts["?"] += 1
        return " ".join(f"{k}={v}" for k, v in counts.items() if v)
    except Exception as e:
        return f"err:{type(e).__name__}"


def measure(name: str, n_kv_heads: int, head_dim: int, seq_len: int):
    label = f"{name}_s{seq_len}"
    print(f"\n=== {label} (n_kv={n_kv_heads}, head_dim={head_dim}) ===")
    path = OUT_DIR / f"{label}.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    build_attn(name, n_kv_heads, head_dim, seq_len, path)
    devs = device_summary(path)
    m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    feed = {
        "x": np.zeros((1, D_MODEL), dtype=np.float32),
        "K": np.zeros((n_kv_heads, seq_len, head_dim), dtype=np.float32),
        "V": np.zeros((n_kv_heads, seq_len, head_dim), dtype=np.float32),
    }
    for _ in range(15):
        m.predict(feed)
    t0 = time.perf_counter()
    for _ in range(80):
        m.predict(feed)
    lat = (time.perf_counter() - t0) / 80
    kv_bytes = 2 * n_kv_heads * seq_len * head_dim * 2
    print(f"  lat={lat*1e3:6.2f} ms   KV={kv_bytes/1e6:5.1f} MB   {devs}")
    return name, seq_len, lat * 1e3


def main():
    rows = []
    # sliding attention: KV is bounded at 1024 tokens regardless of context
    for s in (256, 1024):
        rows.append(measure("sliding", n_kv_heads=8, head_dim=256, seq_len=s))
    # global attention: KV grows with sequence length
    for s in (1024, 2048, 4096, 8192):
        rows.append(measure("global", n_kv_heads=2, head_dim=512, seq_len=s))

    print("\n=== SUMMARY ===")
    print(f"{'regime':10s}  {'seq_len':>8s}  {'lat_ms':>7s}")
    for r in rows:
        print(f"{r[0]:10s}  {r[1]:8d}  {r[2]:7.2f}")

    # Per-token decode projection
    # Gemma-4: 24 sliding layers (always at sliding_window=1024) + 6 global layers
    print("\n=== Per-token attention cost (24 sliding @ s=1024 + 6 global @ ctx) ===")
    sliding_lat = next(lat for n, s, lat in rows if n == "sliding" and s == 1024)
    for n, s, lat in rows:
        if n == "global":
            total = 24 * sliding_lat + 6 * lat
            print(f"  ctx={s:5d}  sliding 24 * {sliding_lat:.2f} + global 6 * {lat:.2f} = {total:6.1f} ms")


if __name__ == "__main__":
    main()
