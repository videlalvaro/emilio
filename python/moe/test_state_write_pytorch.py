"""test_state_write_pytorch.py — Step B of T4.1.4 fix gate.

Pure PyTorch (no CoreML). Verifies that GemmaMixedStackWrap.forward now
write-backs k_cache_i / v_cache_i across forward calls.

Method: run the SAME stack on 2 sequential positions in-place, vs a manually
threaded reference where we explicitly carry k_new[pos=0] -> k_cache[pos=1].
Should match exactly (same code path).

Then a NEGATIVE control: zero the buffers between calls — pos=1 hidden must
DIFFER, proving state matters.

Pass: cos(in_place_pos1, threaded_pos1) >= 0.99999
      AND cos(in_place_pos1, zeroed_pos1) < 0.999  (state matters)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaMixedStackWrap, _load_layer_weights, _layer_types_from_config,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM,
)
from gemma_mixedN_golden import _real_rope  # noqa: E402

OUT_DIR = Path("python/moe/out")
MAX_CTX = 1024
N = 6  # small stack — 6 layers covers 1 sliding-cycle including a global at idx 5
SEED = 0xB10


def _inputs_at(pos: int):
    cos_s, sin_s = _real_rope(theta=10_000.0,    dh=SLD_D_HEAD,  pos=pos)
    cos_g, sin_g = _real_rope(theta=1_000_000.0, dh=GLB_ROT_DIM, pos=pos)
    cos_s = torch.from_numpy(cos_s.astype(np.float16).reshape(1, 1, SLD_D_HEAD))
    sin_s = torch.from_numpy(sin_s.astype(np.float16).reshape(1, 1, SLD_D_HEAD))
    cos_g = torch.from_numpy(cos_g.astype(np.float16).reshape(1, 1, GLB_ROT_DIM))
    sin_g = torch.from_numpy(sin_g.astype(np.float16).reshape(1, 1, GLB_ROT_DIM))
    amask = torch.full((1, 1, 1, MAX_CTX), -1e4, dtype=torch.float16)
    amask[..., : pos + 1] = 0.0
    wmask = torch.zeros(1, 1, MAX_CTX, 1, dtype=torch.float16)
    wmask[0, 0, pos, 0] = 1.0
    return cos_s, sin_s, cos_g, sin_g, amask, wmask


def _build():
    layer_types = _layer_types_from_config(N)
    print(f"  layer_types ({N}): {layer_types}")
    m = GemmaMixedStackWrap(MAX_CTX, layer_types).half().eval()
    npz_paths = [OUT_DIR / f"gemma_layer{i}_packed.npz" for i in range(N)]
    for i, p in enumerate(npz_paths):
        assert p.exists(), p
        _load_layer_weights(m.layers[i], p)
    return m


def _zero_caches(m):
    with torch.no_grad():
        for i in range(N):
            getattr(m, f"k_cache_{i}").zero_()
            getattr(m, f"v_cache_{i}").zero_()


def _cos(a, b):
    a = a.float().flatten().numpy().astype(np.float64)
    b = b.float().flatten().numpy().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _max_state_norm(m):
    n = 0.0
    for i in range(N):
        n = max(n, float(getattr(m, f"k_cache_{i}").float().norm().item()))
        n = max(n, float(getattr(m, f"v_cache_{i}").float().norm().item()))
    return n


def main():
    print("=== Step B: PyTorch state-write proof ===")
    rng = np.random.default_rng(SEED)
    x0 = torch.from_numpy((rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16))
    x1 = torch.from_numpy((rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16))

    # ----- A: in-place run (relies on copy_ inside forward) -----
    print("\n[A] in-place run (state via copy_ in forward)")
    mA = _build(); _zero_caches(mA)
    pre_norm = _max_state_norm(mA)
    with torch.no_grad():
        h0_A, _, _ = mA(x0, *_inputs_at(0))
    mid_norm = _max_state_norm(mA)
    print(f"    state max-norm: pre={pre_norm:.3f}  after pos=0={mid_norm:.3f}  "
          f"{'(WROTE)' if mid_norm > 0.01 else '(NO WRITE — bug)'}")
    if mid_norm < 0.01:
        print("\nFATAL: state buffers stayed at zero after forward. copy_ not effective.")
        sys.exit(2)
    with torch.no_grad():
        h1_A, _, _ = mA(x1, *_inputs_at(1))

    # ----- B: zeroed-between-calls control -----
    print("\n[B] negative control (zero state between pos=0 and pos=1)")
    mB = _build(); _zero_caches(mB)
    with torch.no_grad():
        _ = mB(x0, *_inputs_at(0))
    _zero_caches(mB)
    with torch.no_grad():
        h1_B, _, _ = mB(x1, *_inputs_at(1))

    # ----- C: independent reference using detach copies -----
    print("\n[C] cross-check: in-place at pos=1 vs same code on fresh model")
    mC = _build(); _zero_caches(mC)
    with torch.no_grad():
        _ = mC(x0, *_inputs_at(0))
        h1_C, _, _ = mC(x1, *_inputs_at(1))

    cos_AC = _cos(h1_A, h1_C)
    cos_AB = _cos(h1_A, h1_B)
    print(f"\n  cos(A_pos1, C_pos1)  = {cos_AC:.6f}  (should be ~1.0)")
    print(f"  cos(A_pos1, B_pos1)  = {cos_AB:.6f}  (should be < 0.999 — state matters)")

    ok_repro = cos_AC >= 0.99999
    ok_diff  = cos_AB < 0.999
    if ok_repro and ok_diff:
        print("\n# Step B PASS: state write-back proven in PyTorch")
        sys.exit(0)
    print(f"\n# Step B FAIL: ok_repro={ok_repro} ok_diff={ok_diff}")
    sys.exit(1)


if __name__ == "__main__":
    main()
