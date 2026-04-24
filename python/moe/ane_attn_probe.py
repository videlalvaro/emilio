"""Probe B: Gemma 4 attention block decode latency.

We measure one attention layer's cost during single-token decode at varying
KV cache sizes (seq_len = 256, 1024, 2048, 4096). Projections are INT4-quantized
(the format probe A' picked).

Gemma 4 26B-A4B plausible config (best public guess):
  d_model    = 2304
  n_heads    = 16        (so head_dim = 144)
  n_kv_heads = 4         (GQA 4:1)
  head_dim   = 144

For benchmarking we treat K and V as model inputs of shape
(n_kv_heads, seq_len, head_dim) rather than stateful buffers. This costs the
same BW (the cache must be read either way) and avoids fighting state
plumbing in the probe.
"""
import shutil
import time
from pathlib import Path

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb

OUT_DIR = Path(__file__).parent.parent / "tmp" / "attn_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL = 2304
N_HEADS = 16
N_KV_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS   # 144
KV_DIM = N_KV_HEADS * HEAD_DIM  # 576
SCALE = float(HEAD_DIM ** -0.5)


def build_attn(seq_len: int, out_path: Path):
    """One attention layer for decode (1 query token, seq_len cached KVs).

    Inputs:
      x : (1, d_model)               new token activation
      K : (n_kv_heads, seq_len, head_dim)
      V : (n_kv_heads, seq_len, head_dim)
    Output:
      y : (1, d_model)
    """
    rng = np.random.default_rng(0)
    Wq = rng.standard_normal((N_HEADS * HEAD_DIM, D_MODEL)).astype(np.float16)
    Wk = rng.standard_normal((KV_DIM, D_MODEL)).astype(np.float16)
    Wv = rng.standard_normal((KV_DIM, D_MODEL)).astype(np.float16)
    Wo = rng.standard_normal((D_MODEL, N_HEADS * HEAD_DIM)).astype(np.float16)
    bq = np.zeros((N_HEADS * HEAD_DIM,), dtype=np.float16)
    bk = np.zeros((KV_DIM,), dtype=np.float16)
    bv = np.zeros((KV_DIM,), dtype=np.float16)
    bo = np.zeros((D_MODEL,), dtype=np.float16)

    fp16 = ct.converters.mil.mil.types.fp16

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, D_MODEL), dtype=fp16),
        mb.TensorSpec(shape=(N_KV_HEADS, seq_len, HEAD_DIM), dtype=fp16),
        mb.TensorSpec(shape=(N_KV_HEADS, seq_len, HEAD_DIM), dtype=fp16),
    ])
    def prog(x, K, V):
        q = mb.linear(x=x, weight=Wq, bias=bq, name="q_proj")  # (1, n_heads*head_dim)
        # reshape q -> (n_heads, 1, head_dim)
        q = mb.reshape(x=q, shape=(N_HEADS, 1, HEAD_DIM), name="q_r")
        # broadcast K,V across n_heads / n_kv_heads (GQA): repeat each kv head 4x
        # K,V are (n_kv_heads, seq_len, head_dim) -> need (n_heads, seq_len, head_dim)
        repeats = N_HEADS // N_KV_HEADS
        Kt = mb.tile(x=K, reps=(repeats, 1, 1), name="K_tile")  # (n_heads, seq_len, head_dim)
        Vt = mb.tile(x=V, reps=(repeats, 1, 1), name="V_tile")
        # scores = q @ K^T  ->  (n_heads, 1, seq_len)
        scores = mb.matmul(x=q, y=Kt, transpose_y=True, name="qk")
        scores = mb.mul(x=scores, y=np.float16(SCALE), name="scale")
        attn = mb.softmax(x=scores, axis=-1, name="softmax")
        # ctx = attn @ V  ->  (n_heads, 1, head_dim)
        ctx = mb.matmul(x=attn, y=Vt, name="ctx")
        # reshape -> (1, n_heads*head_dim)
        ctx = mb.reshape(x=ctx, shape=(1, N_HEADS * HEAD_DIM), name="ctx_r")
        y = mb.linear(x=ctx, weight=Wo, bias=bo, name="o_proj")
        return y

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    # INT4 quantize the linear weights (the dominant BW cost)
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
                    if "Neural" in n:
                        counts["ANE"] += 1
                    elif "GPU" in n:
                        counts["GPU"] += 1
                    elif "CPU" in n:
                        counts["CPU"] += 1
                    else:
                        counts["?"] += 1
                except Exception:
                    counts["?"] += 1
        return " ".join(f"{k}={v}" for k, v in counts.items() if v)
    except Exception as e:
        return f"err:{type(e).__name__}"


def measure(seq_len: int):
    print(f"\n=== seq_len = {seq_len} ===")
    path = OUT_DIR / f"attn_s{seq_len}.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    build_attn(seq_len, path)
    devs = device_summary(path)
    m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    in_names = list(m.input_description)
    x = np.zeros((1, D_MODEL), dtype=np.float32)
    K = np.zeros((N_KV_HEADS, seq_len, HEAD_DIM), dtype=np.float32)
    V = np.zeros((N_KV_HEADS, seq_len, HEAD_DIM), dtype=np.float32)
    feed = {}
    for n in in_names:
        if n in ("x", "input_1"):
            feed[n] = x
        elif n == "K":
            feed[n] = K
        elif n == "V":
            feed[n] = V
        else:
            feed[n] = x if "x" not in feed else (K if "K" not in feed else V)
    # warmup
    for _ in range(15):
        m.predict(feed)
    t0 = time.perf_counter()
    for _ in range(80):
        m.predict(feed)
    lat = (time.perf_counter() - t0) / 80
    # KV bytes streamed (fp16, both K and V)
    kv_bytes = 2 * N_KV_HEADS * seq_len * HEAD_DIM * 2
    proj_bytes = (Wq_w := N_HEADS * HEAD_DIM * D_MODEL) + 2 * (KV_DIM * D_MODEL) + (D_MODEL * N_HEADS * HEAD_DIM)
    proj_bytes_int4 = proj_bytes // 2  # int4 = 0.5 byte/param
    print(f"  lat={lat*1e3:6.2f} ms   KV={kv_bytes/1e6:5.1f} MB   projW(int4)={proj_bytes_int4/1e6:5.1f} MB   {devs}")
    return seq_len, lat * 1e3


def main():
    rows = []
    for s in (256, 1024, 2048, 4096):
        rows.append(measure(s))

    print("\n=== SUMMARY (per-layer attention decode) ===")
    print(f"{'seq_len':>8s}  {'lat_ms':>7s}")
    for s, lat in rows:
        print(f"{s:8d}  {lat:7.2f}")

    # Project to per-token decode for Gemma 4 (30 layers)
    print("\n=== Per-token decode projection (attention only, 30 layers) ===")
    for s, lat in rows:
        print(f"  seq={s:5d}  per-token attention = {lat*30:6.1f} ms")


if __name__ == "__main__":
    main()
