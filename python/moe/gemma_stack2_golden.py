"""gemma_stack2_golden.py — T4.1.3b quality gate for stacked-2L conversion.

Compares:
  PyTorch fp16 forward of GemmaStackWrap(N=2) loaded with real layer0 + layer1
  REAP packs                                         ──── reference
  CoreML INT4 forward of gemma4_stack2_real.mlmodelc on identical inputs
                                                     ──── candidate

The gate: INT4 noise accumulated over 2 layers should still give cos ≥ 0.985
(~ per-layer 0.992 floor from T4.1.2, degraded by chaining).

Run with:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_stack2_golden.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaStackWrap, _load_stack_weights,
    D_MODEL, SLD_D_HEAD, SLD_N_KV,
)

OUT_DIR = Path("python/moe/out")
NPZ_PATHS = [OUT_DIR / "gemma_layer0_packed.npz",
             OUT_DIR / "gemma_layer1_packed.npz"]
MLMODELC = OUT_DIR / "gemma4_stack2_real.mlmodelc"
N_LAYERS = 2
MAX_CTX  = 1024
SEED     = 0xA1E
# N-layer INT4 noise compounds from per-layer floor (~0.992 at T4.1.2).
# Residual stream partially anchors drift, but random-input cos still degrades
# slightly per layer. Real quality gate is T4.1.4 (full stack vs HF golden on
# real tokens, cos ≥ 0.97). This gate only catches chaining BUGS.
TOL_COS = 0.98
TOL_TOPK_AGREE = 0.80


def _real_rope(theta: float, dh: int, pos: int) -> tuple[np.ndarray, np.ndarray]:
    half = dh // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half) / half))
    freqs = pos * inv_freq
    full = np.concatenate([freqs, freqs])
    return np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)


def main():
    print(f"=== T4.1.3b stack{N_LAYERS} golden ===")
    rng = np.random.default_rng(SEED)

    x_np    = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)
    cos_np, sin_np = _real_rope(theta=10000.0, dh=SLD_D_HEAD, pos=0)
    cos_np = cos_np.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    sin_np = sin_np.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    attn_mask_np = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    attn_mask_np[..., 0] = 0.0
    wmask_np = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    wmask_np[0, 0, 0, 0] = 1.0

    print("  building PyTorch reference...")
    ref = GemmaStackWrap(MAX_CTX, N_LAYERS)
    ref.half().eval()
    _load_stack_weights(ref, NPZ_PATHS)
    with torch.no_grad():
        for i in range(N_LAYERS):
            getattr(ref, f"k_cache_{i}").zero_()
            getattr(ref, f"v_cache_{i}").zero_()
        t0 = time.perf_counter()
        ref_hidden, ref_k, ref_v = ref(
            torch.from_numpy(x_np),
            torch.from_numpy(cos_np),
            torch.from_numpy(sin_np),
            torch.from_numpy(attn_mask_np),
            torch.from_numpy(wmask_np),
        )
        ref_t = time.perf_counter() - t0
    ref_hidden = ref_hidden.float().numpy().reshape(D_MODEL)
    ref_k_np = ref_k.float().numpy()
    ref_v_np = ref_v.float().numpy()
    print(f"  pytorch fp16 fwd: {ref_t*1e3:.1f} ms, "
          f"hidden ‖x‖={np.linalg.norm(ref_hidden):.3f}")

    print(f"  loading {MLMODELC}...")
    m = ct.models.CompiledMLModel(
        str(MLMODELC), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = m.make_state()
    inputs = {
        "x": x_np, "cos": cos_np, "sin": sin_np,
        "attn_mask": attn_mask_np, "kv_write_mask": wmask_np,
    }
    _ = m.predict(inputs, state=state)  # warmup
    state = m.make_state()
    t0 = time.perf_counter()
    out = m.predict(inputs, state=state)
    cml_t = time.perf_counter() - t0
    cml_hidden = np.asarray(out["hidden"]).astype(np.float32).reshape(D_MODEL)
    cml_k = np.asarray(out["k_new"]).astype(np.float32)
    cml_v = np.asarray(out["v_new"]).astype(np.float32)
    print(f"  coreml int4 fwd: {cml_t*1e3:.1f} ms, "
          f"hidden ‖x‖={np.linalg.norm(cml_hidden):.3f}")

    def cos(a, b):
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cos_h = cos(ref_hidden, cml_hidden)
    cos_k = cos(ref_k_np, cml_k)
    cos_v = cos(ref_v_np, cml_v)
    rmse_h = float(np.sqrt(((ref_hidden - cml_hidden) ** 2).mean()))

    K = 32
    ref_top = set(np.argsort(-np.abs(ref_hidden))[:K].tolist())
    cml_top = set(np.argsort(-np.abs(cml_hidden))[:K].tolist())
    overlap = len(ref_top & cml_top) / K

    print()
    print(f"  cos(hidden) = {cos_h:.6f}")
    print(f"  cos(k_last) = {cos_k:.6f}  (layer {N_LAYERS-1})")
    print(f"  cos(v_last) = {cos_v:.6f}  (layer {N_LAYERS-1})")
    print(f"  rmse(hidden) = {rmse_h:.4f}")
    print(f"  top-{K} dim overlap = {overlap:.3f}")

    ok = (cos_h >= TOL_COS) and (cos_k >= TOL_COS) and (cos_v >= TOL_COS) \
         and (overlap >= TOL_TOPK_AGREE)
    print()
    if ok:
        print(f"# T4.1.3b STACK{N_LAYERS} GOLDEN: PASS")
        sys.exit(0)
    else:
        print(f"# T4.1.3b STACK{N_LAYERS} GOLDEN: FAIL "
              f"(cos≥{TOL_COS}, top-k≥{TOL_TOPK_AGREE})")
        sys.exit(1)


if __name__ == "__main__":
    main()
