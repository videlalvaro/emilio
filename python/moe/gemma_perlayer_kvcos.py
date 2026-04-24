"""gemma_perlayer_kvcos.py — per-layer K/V cosine diagnostic on an existing
mixed-N compiled model. Read-only; no convert. Validates the TurboQuant+
hypothesis: K is the failure axis, and degradation may be concentrated at
specific layers (boundaries / globals).

Approach: run one forward at slot 0 (wmask points at 0) on both the PyTorch
fp16 reference and the CoreML INT4 stateful model, then read every per-layer
k_cache_i / v_cache_i state, take slice [..., 0, :], and compute cosine.

Usage:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_perlayer_kvcos.py --n-layers 12
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
    GemmaMixedStackWrap, _load_layer_weights,
    _layer_types_from_config,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM,
)

OUT_DIR = Path("python/moe/out")
MAX_CTX = 1024
SEED = 0xA1E


def _real_rope(theta: float, dh: int, pos: int):
    half = dh // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half) / half))
    freqs = pos * inv_freq
    full = np.concatenate([freqs, freqs])
    return np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, required=True)
    args = ap.parse_args()
    N = args.n_layers

    layer_types = _layer_types_from_config(N)
    npz_paths = [OUT_DIR / f"gemma_layer{i}_packed.npz" for i in range(N)]
    mlmodelc = OUT_DIR / f"gemma4_mixed{N}_real.mlmodelc"
    print(f"=== mixed{N} per-layer K/V cosine ===")
    print(f"  layer_types: {layer_types}")
    for p in npz_paths:
        if not p.exists():
            print(f"FATAL missing pack: {p}", file=sys.stderr); sys.exit(2)
    if not mlmodelc.exists():
        print(f"FATAL missing mlmodelc: {mlmodelc}", file=sys.stderr); sys.exit(2)

    rng = np.random.default_rng(SEED)
    x_np = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)
    cos_s_f, sin_s_f = _real_rope(theta=10000.0, dh=SLD_D_HEAD, pos=0)
    cos_s = cos_s_f.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    sin_s = sin_s_f.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    cos_g_f, sin_g_f = _real_rope(theta=1_000_000.0, dh=GLB_ROT_DIM, pos=0)
    cos_g = cos_g_f.astype(np.float16).reshape(1, 1, GLB_ROT_DIM)
    sin_g = sin_g_f.astype(np.float16).reshape(1, 1, GLB_ROT_DIM)
    attn_mask_np = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    attn_mask_np[..., 0] = 0.0
    wmask_np = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    wmask_np[0, 0, 0, 0] = 1.0

    print("  building PyTorch reference...")
    ref = GemmaMixedStackWrap(MAX_CTX, layer_types)
    ref.half().eval()
    for i, p in enumerate(npz_paths):
        _load_layer_weights(ref.layers[i], p)
    with torch.no_grad():
        for i in range(N):
            getattr(ref, f"k_cache_{i}").zero_()
            getattr(ref, f"v_cache_{i}").zero_()
        t0 = time.perf_counter()
        _ = ref(
            torch.from_numpy(x_np),
            torch.from_numpy(cos_s), torch.from_numpy(sin_s),
            torch.from_numpy(cos_g), torch.from_numpy(sin_g),
            torch.from_numpy(attn_mask_np),
            torch.from_numpy(wmask_np),
        )
        ref_t = time.perf_counter() - t0
    print(f"  pytorch fwd: {ref_t*1e3:.1f} ms")

    print(f"  loading {mlmodelc}...")
    m = ct.models.CompiledMLModel(str(mlmodelc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = m.make_state()
    inputs = {
        "x": x_np,
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_g": cos_g, "sin_g": sin_g,
        "attn_mask": attn_mask_np, "kv_write_mask": wmask_np,
    }
    # warmup + actual
    _ = m.predict(inputs, state=state)
    state = m.make_state()
    t0 = time.perf_counter()
    _ = m.predict(inputs, state=state)
    cml_t = time.perf_counter() - t0
    print(f"  coreml fwd: {cml_t*1e3:.1f} ms")

    print()
    print(f"  {'layer':>5} {'type':>8} {'cos_K':>10} {'cos_V':>10} {'‖K_ref‖':>10} {'‖K_cml‖':>10}")
    print(f"  {'-'*5:>5} {'-'*8:>8} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")
    rows = []
    for i in range(N):
        t = layer_types[i]
        ref_k = getattr(ref, f"k_cache_{i}").float().numpy()
        ref_v = getattr(ref, f"v_cache_{i}").float().numpy()
        # State shape: (1, n_kv, max_ctx, head_dim). Slot 0 has the new token.
        ref_k0 = ref_k[..., 0, :]
        ref_v0 = ref_v[..., 0, :]
        try:
            cml_k_full = np.asarray(state.read_state(f"k_cache_{i}")).astype(np.float32)
            cml_v_full = np.asarray(state.read_state(f"v_cache_{i}")).astype(np.float32)
        except Exception as e:
            print(f"  [layer {i}] read_state failed: {e}", file=sys.stderr)
            sys.exit(2)
        cml_k0 = cml_k_full[..., 0, :]
        cml_v0 = cml_v_full[..., 0, :]
        ck = _cos(ref_k0, cml_k0)
        cv = _cos(ref_v0, cml_v0)
        nrk = float(np.linalg.norm(ref_k0))
        nrk_c = float(np.linalg.norm(cml_k0))
        rows.append((i, t, ck, cv, nrk, nrk_c))
        print(f"  {i:>5d} {t:>8s} {ck:>10.6f} {cv:>10.6f} {nrk:>10.3f} {nrk_c:>10.3f}")

    # Summary statistics
    print()
    sliding_K = [r[2] for r in rows if r[1] == "sliding"]
    global_K = [r[2] for r in rows if r[1] == "global"]
    sliding_V = [r[3] for r in rows if r[1] == "sliding"]
    global_V = [r[3] for r in rows if r[1] == "global"]
    if sliding_K:
        print(f"  sliding K cos: min={min(sliding_K):.4f} max={max(sliding_K):.4f} mean={np.mean(sliding_K):.4f}  (n={len(sliding_K)})")
        print(f"  sliding V cos: min={min(sliding_V):.4f} max={max(sliding_V):.4f} mean={np.mean(sliding_V):.4f}")
    if global_K:
        print(f"  global  K cos: min={min(global_K):.4f} max={max(global_K):.4f} mean={np.mean(global_K):.4f}  (n={len(global_K)})")
        print(f"  global  V cos: min={min(global_V):.4f} max={max(global_V):.4f} mean={np.mean(global_V):.4f}")
    # Boundary probe
    boundary_idx = {0, 1, N-2, N-1}
    bd_K = [r[2] for r in rows if r[0] in boundary_idx]
    int_K = [r[2] for r in rows if r[0] not in boundary_idx]
    if bd_K and int_K:
        print(f"  boundary(first 2 + last 2) K cos: mean={np.mean(bd_K):.4f}")
        print(f"  interior                   K cos: mean={np.mean(int_K):.4f}")


if __name__ == "__main__":
    main()
