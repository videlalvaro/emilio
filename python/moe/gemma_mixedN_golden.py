"""gemma_mixedN_golden.py — parameterized structural gate for any mixed-N
conversion. CLI: --n-layers N. Reads gemma_layer{0..N-1}_packed.npz +
gemma4_mixed{N}_real.mlmodelc.

Per gatekeeper C3: HARD ABORT (exit 2) on any NaN/Inf in hidden/k/v.
Cosine threshold is loose (random-input compounding through deep INT4 stack
is mostly noise); real quality gate is T4.1.4 vs HF golden.

Usage (Xcode python only — coremltools 9):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_mixedN_golden.py --n-layers 12
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
# Loose thresholds; depth amplifies INT4 noise on random inputs.
# Empirical depth model (validated by gemma_quant_ablation.py + measurements):
#   - hidden cos floor: ~0.98^L on random inputs
#   - K cos floor:      ~0.95^L (compounded through O/FFN/EXP, not K-proj itself;
#                                ablation showed only-K-quant gives cos_K=0.95
#                                at N=12, while ALL-projs gives 0.78)
# Real ship gate is T4.1.4 cos(logits) vs HF on real text — these floors only
# detect catastrophic structural failure (NaN/Inf/sign-flip), not quality.
def _cos_floor_h(n: int) -> float:
    return max(0.30, 0.98 ** n - 0.05)


def _cos_floor_k(n: int) -> float:
    return max(0.30, 0.95 ** n - 0.10)


def _topk_floor(n: int) -> float:
    # Top-K-by-magnitude shuffle is noisy as INT4 noise compounds.
    # Empirical: N=6 ~1.0, N=12 ~0.69, N=30 ~0.25  =>  ~0.95^L
    return max(0.10, 0.95 ** n - 0.05)


def _real_rope(theta: float, dh: int, pos: int):
    half = dh // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half) / half))
    freqs = pos * inv_freq
    full = np.concatenate([freqs, freqs])
    return np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--topk-floor", type=float, default=0.30)
    args = ap.parse_args()
    N = args.n_layers
    cos_floor = _cos_floor_h(N)
    k_floor = _cos_floor_k(N)
    topk_floor = args.topk_floor

    layer_types = _layer_types_from_config(N)
    npz_paths = [OUT_DIR / f"gemma_layer{i}_packed.npz" for i in range(N)]
    mlmodelc = OUT_DIR / f"gemma4_mixed{N}_real.mlmodelc"
    print(f"=== mixed{N} structural golden ===")
    print(f"  layer_types ({N}): {layer_types}")
    print(f"  cos_floor_h={cos_floor:.3f}  cos_floor_k={k_floor:.3f}  topk_floor={topk_floor:.2f}")
    for p in npz_paths:
        if not p.exists():
            print(f"FATAL missing pack: {p}", file=sys.stderr)
            sys.exit(2)
    if not mlmodelc.exists():
        print(f"FATAL missing mlmodelc: {mlmodelc}", file=sys.stderr)
        sys.exit(2)

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
        ref_hidden, ref_k, ref_v = ref(
            torch.from_numpy(x_np),
            torch.from_numpy(cos_s), torch.from_numpy(sin_s),
            torch.from_numpy(cos_g), torch.from_numpy(sin_g),
            torch.from_numpy(attn_mask_np),
            torch.from_numpy(wmask_np),
        )
        ref_t = time.perf_counter() - t0
    ref_hidden = ref_hidden.float().numpy().reshape(D_MODEL)
    ref_k_np = ref_k.float().numpy()
    ref_v_np = ref_v.float().numpy()
    print(f"  pytorch fp16 fwd: {ref_t*1e3:.1f} ms, ‖hidden‖={np.linalg.norm(ref_hidden):.3f}")

    print(f"  loading {mlmodelc}...")
    m = ct.models.CompiledMLModel(str(mlmodelc), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = m.make_state()
    inputs = {
        "x": x_np,
        "cos_s": cos_s, "sin_s": sin_s,
        "cos_g": cos_g, "sin_g": sin_g,
        "attn_mask": attn_mask_np, "kv_write_mask": wmask_np,
    }
    _ = m.predict(inputs, state=state)
    state = m.make_state()
    t0 = time.perf_counter()
    out = m.predict(inputs, state=state)
    cml_t = time.perf_counter() - t0
    cml_hidden = np.asarray(out["hidden"]).astype(np.float32).reshape(D_MODEL)
    cml_k = np.asarray(out["k_new"]).astype(np.float32)
    cml_v = np.asarray(out["v_new"]).astype(np.float32)
    print(f"  coreml int4 fwd: {cml_t*1e3:.1f} ms, ‖hidden‖={np.linalg.norm(cml_hidden):.3f}")

    # === C3: HARD finite-only abort ===
    for name, arr in [("hidden", cml_hidden), ("k_new", cml_k), ("v_new", cml_v)]:
        if not np.all(np.isfinite(arr)):
            n_nan = int(np.isnan(arr).sum()); n_inf = int(np.isinf(arr).sum())
            print(f"FATAL non-finite in {name}: nan={n_nan} inf={n_inf}", file=sys.stderr)
            sys.exit(2)

    def cos(a, b):
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cos_h = cos(ref_hidden, cml_hidden)
    cos_k = cos(ref_k_np, cml_k)
    cos_v = cos(ref_v_np, cml_v)
    K = 32
    ref_top = set(np.argsort(-np.abs(ref_hidden))[:K].tolist())
    cml_top = set(np.argsort(-np.abs(cml_hidden))[:K].tolist())
    overlap = len(ref_top & cml_top) / K

    print()
    print(f"  cos(hidden) = {cos_h:.6f}  (floor {cos_floor:.3f})")
    print(f"  cos(k_last) = {cos_k:.6f}  (floor {k_floor:.3f}, layer {N-1}, type={layer_types[-1]})")
    print(f"  cos(v_last) = {cos_v:.6f}")
    print(f"  top-{K} overlap = {overlap:.3f}  (floor {topk_floor:.2f})")

    ok = (cos_h >= cos_floor) and (cos_k >= k_floor) and (overlap >= topk_floor)
    if ok:
        print(f"\n# mixed{N} STRUCTURAL GOLDEN: PASS")
        sentinel = OUT_DIR / f".gemma_mixed{N}_golden_PASS"
        sentinel.write_text(f"cos_h={cos_h:.6f} cos_k={cos_k:.6f} overlap={overlap:.3f}\n")
        print(f"  sentinel: {sentinel}")
        sys.exit(0)
    else:
        print(f"\n# mixed{N} STRUCTURAL GOLDEN: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
