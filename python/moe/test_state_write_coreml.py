"""test_state_write_coreml.py — Step B.5 runtime check.

Verifies CoreML state buffers actually persist across predict() calls on the
1-layer artifact. Uses real PyTorch reference for ground truth.

A: predict pos=0 then pos=1 with persistent state (state mutates).
B: predict pos=1 with zeroed state (control: should differ from A).
C: PyTorch reference pos=1 after pos=0 (ground truth).

Pass:
  cos(A_pos1, C_pos1) >= 0.95   (CoreML matches PyTorch with state)
  cos(A_pos1, B_pos1) < 0.999   (state mattered at runtime)
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
    GemmaLayer1Wrap, _load_weights,
    D_MODEL, SLD_D_HEAD,
)
from gemma_mixedN_golden import _real_rope  # noqa: E402

OUT_DIR = Path("python/moe/out")
MAX_CTX = 1024
MLMC = OUT_DIR / "gemma4_layer1_real.mlmodelc"


def _inputs_at(pos):
    cos, sin = _real_rope(theta=10_000.0, dh=SLD_D_HEAD, pos=pos)
    cos = cos.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    sin = sin.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    am = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    am[..., : pos + 1] = 0.0
    wm = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    wm[0, 0, pos, 0] = 1.0
    return cos, sin, am, wm


def _to_t(arr):
    return torch.from_numpy(arr)


def _cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    assert MLMC.exists(), MLMC
    rng = np.random.default_rng(0xC011)
    x0 = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)
    x1 = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)

    # ---- PyTorch reference ----
    ref = GemmaLayer1Wrap(MAX_CTX); ref.half().eval()
    _load_weights(ref, OUT_DIR / "gemma_layer0_packed.npz")
    with torch.no_grad():
        ref.k_cache_0.zero_(); ref.v_cache_0.zero_()
        cos0, sin0, am0, wm0 = _inputs_at(0)
        h0_ref, _, _ = ref(_to_t(x0), _to_t(cos0), _to_t(sin0), _to_t(am0), _to_t(wm0))
        cos1, sin1, am1, wm1 = _inputs_at(1)
        h1_ref, _, _ = ref(_to_t(x1), _to_t(cos1), _to_t(sin1), _to_t(am1), _to_t(wm1))
    h1_ref_np = h1_ref.float().numpy().reshape(D_MODEL)

    # ---- A: persistent state across two predict() calls ----
    print(f"  loading {MLMC.name}...")
    m = ct.models.CompiledMLModel(str(MLMC), compute_units=ct.ComputeUnit.CPU_AND_NE)

    state_A = m.make_state()
    cos0, sin0, am0, wm0 = _inputs_at(0)
    out0 = m.predict(dict(x=x0, cos=cos0, sin=sin0,
                          attn_mask=am0, kv_write_mask=wm0), state=state_A)
    cos1, sin1, am1, wm1 = _inputs_at(1)
    out1_A = m.predict(dict(x=x1, cos=cos1, sin=sin1,
                            attn_mask=am1, kv_write_mask=wm1), state=state_A)
    h1_A = np.asarray(out1_A["hidden"]).astype(np.float32).reshape(D_MODEL)

    # ---- B: fresh state at pos=1 (control) ----
    state_B = m.make_state()
    out1_B = m.predict(dict(x=x1, cos=cos1, sin=sin1,
                            attn_mask=am1, kv_write_mask=wm1), state=state_B)
    h1_B = np.asarray(out1_B["hidden"]).astype(np.float32).reshape(D_MODEL)

    cos_AC = _cos(h1_A, h1_ref_np)
    cos_AB = _cos(h1_A, h1_B)
    print(f"\n  cos(coreml_persistent_pos1, pytorch_ref_pos1) = {cos_AC:.6f}  "
          f"(should be >= 0.95)")
    print(f"  cos(coreml_persistent_pos1, coreml_zeroed_pos1) = {cos_AB:.6f}  "
          f"(should be < 0.999 — state mattered at runtime)")

    ok_match = cos_AC >= 0.95
    ok_diff  = cos_AB < 0.999
    if ok_match and ok_diff:
        print("\n# Step B.5 runtime: PASS — state propagates across predict() calls")
        sys.exit(0)
    print(f"\n# Step B.5 runtime: FAIL — match={ok_match} diff={ok_diff}")
    sys.exit(1)


if __name__ == "__main__":
    main()
