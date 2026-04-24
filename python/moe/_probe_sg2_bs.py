"""_probe_sg2_bs.py — re-quant the SG2 traced model at different block sizes
to isolate whether o_proj noise is the bottleneck.

Reuses: the existing converted+saved torch ref. Re-traces and re-converts
WITHOUT INT4 (fp16 baseline) and with block_size=8 (finer than 16).
Compares hidden cos for each.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
import coremltools.optimize.coreml as cto

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaMixedStackWrap, _load_layer_weights, _state_shape,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM,
)

OUT_DIR = Path("python/moe/out")
LAYER_TYPES = ["sliding", "global"]
NPZS = [OUT_DIR / "gemma_layer4_packed.npz",
        OUT_DIR / "gemma_layer5_packed.npz"]
MAX_CTX = 1024
SEED = 0xA1E


def _real_rope(theta: float, dh: int, pos: int):
    half = dh // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half) / half))
    freqs = pos * inv_freq
    full = np.concatenate([freqs, freqs])
    return np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)


def cos(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def trace_ref():
    ref = GemmaMixedStackWrap(MAX_CTX, LAYER_TYPES)
    ref.half().eval()
    for i, p in enumerate(NPZS):
        _load_layer_weights(ref.layers[i], p)
    dh_s = SLD_D_HEAD
    x_ex   = torch.randn(1, 1, D_MODEL,    dtype=torch.float16) * 0.1
    cos_s  = torch.randn(1, 1, dh_s,        dtype=torch.float16)
    sin_s  = torch.randn(1, 1, dh_s,        dtype=torch.float16)
    cos_g  = torch.randn(1, 1, GLB_ROT_DIM, dtype=torch.float16)
    sin_g  = torch.randn(1, 1, GLB_ROT_DIM, dtype=torch.float16)
    am_ex  = torch.full((1, 1, 1, MAX_CTX), -1e4, dtype=torch.float16); am_ex[..., 0] = 0.0
    wm_ex  = torch.zeros(1, 1, MAX_CTX, 1, dtype=torch.float16); wm_ex[0, 0, 0, 0] = 1.0
    traced = torch.jit.trace(ref, (x_ex, cos_s, sin_s, cos_g, sin_g, am_ex, wm_ex))
    with torch.no_grad():
        for i in range(2):
            getattr(ref, f"k_cache_{i}").zero_()
            getattr(ref, f"v_cache_{i}").zero_()
    return ref, traced


def convert(traced, *, quant_bs: int | None, tag: str):
    pkg = OUT_DIR / f"gemma4_sg2_{tag}.mlpackage"
    mlc = OUT_DIR / f"gemma4_sg2_{tag}.mlmodelc"
    if mlc.exists():
        print(f"  reusing {mlc.name}")
        return mlc
    dh_s = SLD_D_HEAD
    ct_inputs = [
        ct.TensorType(name="x",             shape=(1, 1, D_MODEL),    dtype=np.float16),
        ct.TensorType(name="cos_s",         shape=(1, 1, dh_s),       dtype=np.float16),
        ct.TensorType(name="sin_s",         shape=(1, 1, dh_s),       dtype=np.float16),
        ct.TensorType(name="cos_g",         shape=(1, 1, GLB_ROT_DIM),dtype=np.float16),
        ct.TensorType(name="sin_g",         shape=(1, 1, GLB_ROT_DIM),dtype=np.float16),
        ct.TensorType(name="attn_mask",     shape=(1, 1, 1, MAX_CTX), dtype=np.float16),
        ct.TensorType(name="kv_write_mask", shape=(1, 1, MAX_CTX, 1), dtype=np.float16),
    ]
    ct_outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
        ct.TensorType(name="k_new",  dtype=np.float16),
        ct.TensorType(name="v_new",  dtype=np.float16),
    ]
    ct_states = []
    for i, t in enumerate(LAYER_TYPES):
        shp = _state_shape(t == "global", MAX_CTX)
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=shp, dtype=np.float16),
            name=f"k_cache_{i}"))
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=shp, dtype=np.float16),
            name=f"v_cache_{i}"))
    print(f"  ct.convert ({tag})...")
    t0 = time.perf_counter()
    m = ct.convert(
        traced, inputs=ct_inputs, outputs=ct_outputs, states=ct_states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  convert wall: {time.perf_counter()-t0:.1f}s")
    if quant_bs is not None:
        print(f"  INT4 per_block bs={quant_bs}...")
        t0 = time.perf_counter()
        m = cto.linear_quantize_weights(
            m, config=cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(
                    mode="linear_symmetric", dtype="int4",
                    granularity="per_block", block_size=quant_bs,
                    weight_threshold=0)))
        print(f"  quant wall: {time.perf_counter()-t0:.1f}s")
    if pkg.exists(): shutil.rmtree(pkg)
    m.save(str(pkg))
    if mlc.exists(): shutil.rmtree(mlc)
    Path(ct.utils.compile_model(str(pkg), str(mlc)))
    pkg_mb = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1e6
    print(f"  saved {pkg.name} ({pkg_mb:.1f} MB)")
    return mlc


def run(mlc, inputs):
    m = ct.models.CompiledMLModel(str(mlc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = m.make_state()
    _ = m.predict(inputs, state=state)
    state = m.make_state()
    out = m.predict(inputs, state=state)
    return (np.asarray(out["hidden"]).astype(np.float32).reshape(D_MODEL),
            np.asarray(out["k_new"]).astype(np.float32),
            np.asarray(out["v_new"]).astype(np.float32))


def main():
    rng = np.random.default_rng(SEED)
    x_np = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)
    cos_s_f, sin_s_f = _real_rope(10000.0,    SLD_D_HEAD, 0)
    cos_g_f, sin_g_f = _real_rope(1_000_000., GLB_ROT_DIM, 0)
    cos_s = cos_s_f.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    sin_s = sin_s_f.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    cos_g = cos_g_f.astype(np.float16).reshape(1, 1, GLB_ROT_DIM)
    sin_g = sin_g_f.astype(np.float16).reshape(1, 1, GLB_ROT_DIM)
    am = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16); am[..., 0] = 0.0
    wm = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16); wm[0, 0, 0, 0] = 1.0
    inputs = {"x": x_np, "cos_s": cos_s, "sin_s": sin_s,
              "cos_g": cos_g, "sin_g": sin_g,
              "attn_mask": am, "kv_write_mask": wm}

    print("=== sg2 quantization sensitivity ===")
    print("  building torch ref...")
    ref, traced = trace_ref()
    with torch.no_grad():
        ref_hid, ref_k, ref_v = ref(
            torch.from_numpy(x_np),
            torch.from_numpy(cos_s), torch.from_numpy(sin_s),
            torch.from_numpy(cos_g), torch.from_numpy(sin_g),
            torch.from_numpy(am), torch.from_numpy(wm))
    ref_hid_np = ref_hid.float().numpy().reshape(D_MODEL)
    ref_k_np = ref_k.float().numpy()
    ref_v_np = ref_v.float().numpy()

    results = []
    for tag, bs in [("fp16",  None), ("int4_bs8",  8), ("int4_bs16", 16), ("int4_bs32", 32)]:
        print(f"\n--- variant: {tag} ---")
        # Re-trace fresh each time (state buffers were zeroed already, but
        # convert mutates internal stuff). Skip re-trace; just convert.
        mlc = convert(traced, quant_bs=bs, tag=tag)
        cml_hid, cml_k, cml_v = run(mlc, inputs)
        ch = cos(ref_hid_np, cml_hid)
        ck = cos(ref_k_np,   cml_k)
        cv = cos(ref_v_np,   cml_v)
        # Get pkg size
        pkg = OUT_DIR / f"gemma4_sg2_{tag}.mlpackage"
        sz = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1e6
        print(f"  cos(hidden)={ch:.6f}  cos(k)={ck:.6f}  cos(v)={cv:.6f}  pkg={sz:.0f}MB")
        results.append((tag, ch, ck, cv, sz))

    print("\n=== summary ===")
    print(f"{'variant':12s} {'cos(hidden)':>12s} {'cos(k)':>10s} {'cos(v)':>10s} {'pkg MB':>10s}")
    for tag, ch, ck, cv, sz in results:
        print(f"{tag:12s} {ch:12.6f} {ck:10.6f} {cv:10.6f} {sz:10.1f}")


if __name__ == "__main__":
    main()
