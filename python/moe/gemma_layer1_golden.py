"""gemma_layer1_golden.py — T4.1.2 quality gate for layer1 conversion.

Compares:
  PyTorch fp16 forward of GemmaLayer1Wrap (real REAP weights from
  gemma_layer0_packed.npz)            ──── reference
  CoreML INT4 forward of gemma4_layer1_real.mlmodelc on identical inputs
                                       ──── candidate

The gap is INT4 weight quantization + any conversion bug. Bug-free should
sit close to pure-INT4 noise, which for Gemma-style weights is typically
cos ≥ 0.99 per layer.

Inputs are sampled deterministically from a real prompt embedding distribution:
  - x: realistic post-input_ln residual scale  (we use random N(0, 0.5²) → fp16
    ROUGHLY matching post-RMSNorm magnitudes; for a single-layer test the
    distribution doesn't have to match a real corpus)
  - cos, sin: real RoPE values for sliding layer (theta=10000) at position 0
  - attn_mask, kv_write_mask: position 0 only (decode-style)
  - State buffers initialized to zero

Run with the only python that has coremltools 9 + torch:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_layer1_golden.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

# Reuse the converter's PyTorch module + weight loader.
sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaLayer1Wrap, _load_weights,
    D_MODEL, SLD_D_HEAD, SLD_N_KV,
)

OUT_DIR = Path("python/moe/out")
NPZ     = OUT_DIR / "gemma_layer0_packed.npz"
MLMODELC = OUT_DIR / "gemma4_layer1_real.mlmodelc"
MAX_CTX = 1024
SEED    = 0xA1E
# Single-layer cos vs same-weights fp16 reference — measures INT4 quant noise.
# Empirical floor for per_block=16 INT4 on Gemma-MoE shapes: ~0.99.
# True quality gate is full-stack T4.1.4 (cos ≥ 0.97 vs HF golden).
TOL_COS = 0.99
TOL_TOPK_AGREE = 0.90  # 32-dim overlap on random input is noisy


def _real_rope(theta: float, dh: int, pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard RoPE cos/sin for one position. Returns (cos, sin) shape (dh,)."""
    half = dh // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half) / half))   # (dh/2,)
    freqs = pos * inv_freq                                     # (dh/2,)
    # HF stores as cat([freqs, freqs]) so cos/sin span full dh.
    full = np.concatenate([freqs, freqs])
    return np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)


def main():
    print("=== T4.1.2 layer1 golden ===")
    rng = np.random.default_rng(SEED)

    # ── Build inputs ──────────────────────────────────────────────
    x_np    = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)
    cos_np, sin_np = _real_rope(theta=10000.0, dh=SLD_D_HEAD, pos=0)
    cos_np = cos_np.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    sin_np = sin_np.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    attn_mask_np = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    attn_mask_np[..., 0] = 0.0
    wmask_np = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    wmask_np[0, 0, 0, 0] = 1.0
    k_state_np = np.zeros((1, SLD_N_KV, MAX_CTX, SLD_D_HEAD), dtype=np.float16)
    v_state_np = np.zeros((1, SLD_N_KV, MAX_CTX, SLD_D_HEAD), dtype=np.float16)

    # ── PyTorch fp16 reference ────────────────────────────────────
    print("  building PyTorch reference...")
    ref = GemmaLayer1Wrap(MAX_CTX)
    ref.half().eval()
    _load_weights(ref, NPZ)
    with torch.no_grad():
        ref.k_cache_0.zero_()
        ref.v_cache_0.zero_()
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

    # ── CoreML INT4 candidate ─────────────────────────────────────
    print(f"  loading {MLMODELC}...")
    m = ct.models.CompiledMLModel(
        str(MLMODELC), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = m.make_state()
    inputs = {
        "x": x_np, "cos": cos_np, "sin": sin_np,
        "attn_mask": attn_mask_np, "kv_write_mask": wmask_np,
    }
    # Warmup once (first call has compile cost).
    _ = m.predict(inputs, state=state)
    state = m.make_state()  # reset
    t0 = time.perf_counter()
    out = m.predict(inputs, state=state)
    cml_t = time.perf_counter() - t0
    cml_hidden = np.asarray(out["hidden"]).astype(np.float32).reshape(D_MODEL)
    cml_k = np.asarray(out["k_new"]).astype(np.float32)
    cml_v = np.asarray(out["v_new"]).astype(np.float32)
    print(f"  coreml int4 fwd: {cml_t*1e3:.1f} ms, "
          f"hidden ‖x‖={np.linalg.norm(cml_hidden):.3f}")

    # ── Compare ────────────────────────────────────────────────────
    def cos(a, b):
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cos_h = cos(ref_hidden, cml_hidden)
    cos_k = cos(ref_k_np,  cml_k)
    cos_v = cos(ref_v_np,  cml_v)
    rmse_h = float(np.sqrt(((ref_hidden - cml_hidden) ** 2).mean()))

    # Top-k agreement on hidden vector (treat as "which 32 dims have biggest abs")
    K = 32
    ref_top = set(np.argsort(-np.abs(ref_hidden))[:K].tolist())
    cml_top = set(np.argsort(-np.abs(cml_hidden))[:K].tolist())
    overlap = len(ref_top & cml_top) / K

    print()
    print(f"  cos(hidden) = {cos_h:.6f}")
    print(f"  cos(k_new)  = {cos_k:.6f}")
    print(f"  cos(v_new)  = {cos_v:.6f}")
    print(f"  rmse(hidden) = {rmse_h:.4f}")
    print(f"  top-{K} dim overlap = {overlap:.3f}")

    ok = (cos_h >= TOL_COS) and (cos_k >= TOL_COS) and (cos_v >= TOL_COS) \
         and (overlap >= TOL_TOPK_AGREE)
    print()
    if ok:
        print("# T4.1.2 LAYER1 GOLDEN: PASS")
        sys.exit(0)
    else:
        print(f"# T4.1.2 LAYER1 GOLDEN: FAIL (cos≥{TOL_COS}, top-k≥{TOL_TOPK_AGREE})")
        sys.exit(1)


if __name__ == "__main__":
    main()
