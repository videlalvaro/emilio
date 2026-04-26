"""gemma_split_golden.py — structural golden gate for the split (attn + FFN
sub-shard) architecture.

Compares one-layer CoreML split shards against PyTorch fp16 reference.
Chain: attn_shard(x) → ffn_partial_0(h) → ... → last_partial(h, prior_sum) → hidden

The last FFN partial shard includes the combiner (dense MLP + norms + residual)
merged in, so there is no separate combiner shard.

Usage (Xcode python only — coremltools 9):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_split_golden.py --layer 0 --ffn-shards 2 --quant-bits 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaSlidingLayer, GemmaGlobalLayer,
    _make_layer_for_type, _load_layer_weights, _layer_types_from_config,
    _state_shape,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM, N_PACKS,
)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "python" / "moe" / "out"
SEED = 0xA1E
THETA_SLD = 10_000.0
THETA_GLB = 1_000_000.0


def _real_rope(theta: float, dh: int, pos: int):
    half = dh // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half) / half))
    freqs = pos * inv_freq
    full = np.concatenate([freqs, freqs])
    return np.cos(full).astype(np.float16), np.sin(full).astype(np.float16)


def _cos_sim(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description="Split-shard structural golden gate")
    ap.add_argument("--layer", type=int, required=True,
                    help="Global layer index (e.g. 0, 5)")
    ap.add_argument("--ffn-shards", type=int, default=2)
    ap.add_argument("--quant-bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--n-layers", type=int, default=30,
                    help="Total model layers (for config.json lookup)")
    ap.add_argument("--max-ctx", type=int, default=1024,
                    help="Max context length (must match attn shard)")
    ap.add_argument("--cos-floor", type=float, default=0.95,
                    help="Cosine similarity floor for PASS (single layer, "
                         "should be very high)")
    args = ap.parse_args()

    layer_idx = args.layer
    ffn_shards = args.ffn_shards
    qsuf = f"_q{args.quant_bits}"
    tag = f"shard{layer_idx}_{layer_idx + 1}"

    layer_types = _layer_types_from_config(args.n_layers)
    layer_type = layer_types[layer_idx]
    is_global = layer_type == "global"
    max_ctx = args.max_ctx

    # --- Locate artifacts --------------------------------------------------
    attn_path   = OUT_DIR / f"gemma4_{tag}_real_attn{qsuf}.mlmodelc"
    # First (ffn_shards - 1) are regular partials, last one is merged
    partial_paths = [
        OUT_DIR / f"gemma4_{tag}_real_ffn_p{k}of{ffn_shards}{qsuf}.mlmodelc"
        for k in range(ffn_shards - 1)
    ]
    last_path = OUT_DIR / f"gemma4_{tag}_real_ffn_p{ffn_shards - 1}of{ffn_shards}{qsuf}.mlmodelc"
    npz_path  = OUT_DIR / f"gemma_layer{layer_idx}_packed.npz"

    print(f"=== split golden gate: layer {layer_idx} ({layer_type}) ===")
    print(f"  ffn_shards={ffn_shards}  quant=int{args.quant_bits}")
    print(f"  cos_floor={args.cos_floor}")
    all_paths = [attn_path] + partial_paths + [last_path, npz_path]
    for p in all_paths:
        if not p.exists():
            print(f"FATAL missing: {p}", file=sys.stderr)
            sys.exit(2)
        print(f"  ✓ {p.name}")

    # --- Build PyTorch reference -------------------------------------------
    print("\n  building PyTorch fp16 reference...")
    layer = _make_layer_for_type(layer_type, max_ctx)
    layer.half().eval()
    _load_layer_weights(layer, npz_path)
    layer.fuse_norm_scales_for_ane()  # match the CoreML conversion pipeline

    rng = np.random.default_rng(SEED)
    x_np = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)

    # Build rope for the layer's attention type
    dh = GLB_ROT_DIM if is_global else SLD_D_HEAD
    theta = THETA_GLB if is_global else THETA_SLD
    cos_np, sin_np = _real_rope(theta, dh, pos=0)
    cos_rope = cos_np.reshape(1, 1, dh)
    sin_rope = sin_np.reshape(1, 1, dh)

    # Build both rope variants for the CoreML attn shard (which takes all 4)
    cos_s_np, sin_s_np = _real_rope(THETA_SLD, SLD_D_HEAD, pos=0)
    cos_s = cos_s_np.reshape(1, 1, SLD_D_HEAD)
    sin_s = sin_s_np.reshape(1, 1, SLD_D_HEAD)
    cos_g_np, sin_g_np = _real_rope(THETA_GLB, GLB_ROT_DIM, pos=0)
    cos_g = cos_g_np.reshape(1, 1, GLB_ROT_DIM)
    sin_g = sin_g_np.reshape(1, 1, GLB_ROT_DIM)

    kv_shape = _state_shape(is_global, max_ctx)
    k_cache = np.zeros(kv_shape, dtype=np.float16)
    v_cache = np.zeros(kv_shape, dtype=np.float16)
    attn_mask = np.full((1, 1, 1, max_ctx), -1e4, dtype=np.float16)
    attn_mask[..., 0] = 0.0
    wmask = np.zeros((1, 1, max_ctx, 1), dtype=np.float16)
    wmask[0, 0, 0, 0] = 1.0

    with torch.no_grad():
        x_t = torch.from_numpy(x_np)
        ref_out, _, _ = layer(
            x_t,
            torch.from_numpy(cos_rope), torch.from_numpy(sin_rope),
            torch.from_numpy(k_cache), torch.from_numpy(v_cache),
            torch.from_numpy(attn_mask), torch.from_numpy(wmask),
        )
    ref_hidden = ref_out.float().numpy().reshape(D_MODEL)
    print(f"  pytorch ‖hidden‖ = {np.linalg.norm(ref_hidden):.4f}")

    # --- Also get PyTorch intermediate after attn (for sub-gate) ----------
    with torch.no_grad():
        ref_attn_out, _, _ = layer.forward_attn(
            x_t,
            torch.from_numpy(cos_rope), torch.from_numpy(sin_rope),
            torch.from_numpy(k_cache), torch.from_numpy(v_cache),
            torch.from_numpy(attn_mask), torch.from_numpy(wmask),
        )
    ref_attn_hidden = ref_attn_out.float().numpy().reshape(D_MODEL)

    # --- Run CoreML attn shard --------------------------------------------
    print(f"\n  loading attn shard: {attn_path.name}...")
    m_attn = ct.models.CompiledMLModel(
        str(attn_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state_attn = m_attn.make_state()
    attn_inputs = {
        "x": x_np,
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_g": cos_g, "sin_g": sin_g,
        "attn_mask": attn_mask,
        "kv_write_mask": wmask,
    }
    # warmup
    _ = m_attn.predict(attn_inputs, state=m_attn.make_state())
    state_attn = m_attn.make_state()
    t0 = time.perf_counter()
    attn_out = m_attn.predict(attn_inputs, state=state_attn)
    attn_ms = (time.perf_counter() - t0) * 1e3
    cml_attn_hidden = np.asarray(attn_out["hidden"]).astype(np.float32).reshape(D_MODEL)
    print(f"  attn fwd: {attn_ms:.1f} ms, ‖h‖={np.linalg.norm(cml_attn_hidden):.4f}")

    cos_attn = _cos_sim(ref_attn_hidden, cml_attn_hidden)
    print(f"  cos(attn) = {cos_attn:.6f}")

    # --- Run CoreML FFN partial shards ------------------------------------
    # First (ffn_shards - 1) partials produce partial_moe, accumulated in
    # partial_sum.  The last shard is a merged partial + combiner that takes
    # both x and the accumulated prior_partial_moe and returns hidden.
    h_for_ffn = np.asarray(attn_out["hidden"]).astype(np.float16).reshape(1, 1, D_MODEL)
    partial_sum = np.zeros((1, D_MODEL), dtype=np.float32)

    for k, pp in enumerate(partial_paths):
        print(f"\n  loading FFN partial {k}: {pp.name}...")
        m_p = ct.models.CompiledMLModel(
            str(pp), compute_units=ct.ComputeUnit.CPU_AND_NE)
        t0 = time.perf_counter()
        p_out = m_p.predict({"x": h_for_ffn})
        p_ms = (time.perf_counter() - t0) * 1e3
        p_moe = np.asarray(p_out["partial_moe"]).astype(np.float32).reshape(1, D_MODEL)
        partial_sum += p_moe
        print(f"  partial {k} fwd: {p_ms:.1f} ms, "
              f"‖partial‖={np.linalg.norm(p_moe):.4f}")
        del m_p

    # --- Run last partial + combiner (merged) -----------------------------
    print(f"\n  loading last partial + combiner: {last_path.name}...")
    m_last = ct.models.CompiledMLModel(
        str(last_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    last_x = h_for_ffn  # (1, 1, D_MODEL)
    prior_moe = partial_sum.astype(np.float16).reshape(1, 1, D_MODEL)
    t0 = time.perf_counter()
    last_out = m_last.predict({"x": last_x, "prior_partial_moe": prior_moe})
    last_ms = (time.perf_counter() - t0) * 1e3
    cml_hidden = np.asarray(last_out["hidden"]).astype(np.float32).reshape(D_MODEL)
    print(f"  last partial + combiner fwd: {last_ms:.1f} ms, "
          f"‖hidden‖={np.linalg.norm(cml_hidden):.4f}")

    # === Finite check =====================================================
    for name, arr in [("attn_hidden", cml_attn_hidden), ("final_hidden", cml_hidden)]:
        if not np.all(np.isfinite(arr)):
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            print(f"FATAL non-finite in {name}: nan={n_nan} inf={n_inf}",
                  file=sys.stderr)
            sys.exit(2)

    # === Cosine & top-K ===================================================
    cos_h = _cos_sim(ref_hidden, cml_hidden)
    K = 32
    ref_top = set(np.argsort(-np.abs(ref_hidden))[:K].tolist())
    cml_top = set(np.argsort(-np.abs(cml_hidden))[:K].tolist())
    overlap = len(ref_top & cml_top) / K

    rmse = float(np.sqrt(np.mean((ref_hidden - cml_hidden) ** 2)))

    print(f"\n{'='*60}")
    print(f"  cos(hidden)   = {cos_h:.6f}   (floor {args.cos_floor})")
    print(f"  cos(attn_mid) = {cos_attn:.6f}")
    print(f"  top-{K} overlap = {overlap:.3f}")
    print(f"  RMSE          = {rmse:.6f}")
    print(f"{'='*60}")

    ok = cos_h >= args.cos_floor
    if ok:
        print(f"\n# SPLIT GOLDEN (L{layer_idx} {layer_type}): PASS")
        sentinel = OUT_DIR / f".gemma_split_L{layer_idx}_{args.quant_bits}b_golden_PASS"
        sentinel.write_text(
            f"cos_h={cos_h:.6f} cos_attn={cos_attn:.6f} "
            f"overlap={overlap:.3f} rmse={rmse:.6f}\n")
        print(f"  sentinel: {sentinel}")
        sys.exit(0)
    else:
        print(f"\n# SPLIT GOLDEN (L{layer_idx} {layer_type}): FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
