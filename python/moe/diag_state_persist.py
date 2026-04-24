"""diag_state_persist.py — investigate why T4.1.4 collapses at pos>=1.

Hypothesis A: CoreML state buffers do NOT carry K/V from pos=0 into pos=1.
Hypothesis B: state persists, but kv_write_mask path produces 0 at slot 0
              by the time it's read at pos=1 (write got lost).
Hypothesis C: state persists fine; some other bug (RoPE, scale, mask) makes
              attention output near-zero at pos>=1, collapsing residual.

Method: feed two distinct random embeddings through shard0_15 at pos=0 and
pos=1 with persistent state. Then re-run pos=1 ALONE with fresh state +
synthetic attn_mask that allows ONLY slot 1 (no past). If the persistent-state
pos=1 differs significantly from the no-past pos=1, attention IS reading slot
0 → state works → bug is elsewhere. If they match, attention is NOT reading
slot 0 → state read is broken.

Also computes ‖hidden(pos=0)‖, ‖hidden(pos=1)‖, cos(pos0, pos1) to detect
"layers are no-ops at pos=1" symptom.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))
from gemma_mixedN_golden import _real_rope  # noqa: E402

OUT_DIR = Path("python/moe/out")
SHARD = OUT_DIR / "gemma4_shard0_15_real.mlmodelc"
D_MODEL = 2816
SLD_DH = 256
GLB_ROT = 128
MAX_CTX = 1024


def _rope(theta, dh, pos):
    cs, sn = _real_rope(theta=theta, dh=dh, pos=pos)
    return (cs.astype(np.float16).reshape(1, 1, dh),
            sn.astype(np.float16).reshape(1, 1, dh))


def _amask_upto(pos):
    m = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    m[..., : pos + 1] = 0.0
    return m


def _amask_only(slot):
    """Mask that allows ONLY one slot to be attended to."""
    m = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    m[..., slot] = 0.0
    return m


def _wmask(pos):
    w = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    w[0, 0, pos, 0] = 1.0
    return w


def _cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _norm(a):
    return float(np.linalg.norm(a.astype(np.float64).ravel()))


def main():
    assert SHARD.exists(), SHARD
    rng = np.random.default_rng(0xD1A6)
    x0 = (rng.standard_normal((1, 1, D_MODEL)) * 1.0).astype(np.float16)
    x1 = (rng.standard_normal((1, 1, D_MODEL)) * 1.0).astype(np.float16)

    print(f"  loading {SHARD.name}...")
    m = ct.models.CompiledMLModel(str(SHARD), compute_units=ct.ComputeUnit.CPU_AND_NE)

    cs0, sn0 = _rope(10_000.0, SLD_DH, 0); cg0, sg0 = _rope(1_000_000.0, GLB_ROT, 0)
    cs1, sn1 = _rope(10_000.0, SLD_DH, 1); cg1, sg1 = _rope(1_000_000.0, GLB_ROT, 1)

    base0 = dict(cos_s=cs0, sin_s=sn0, cos_g=cg0, sin_g=sg0,
                 attn_mask=_amask_upto(0), kv_write_mask=_wmask(0))
    base1 = dict(cos_s=cs1, sin_s=sn1, cos_g=cg1, sin_g=sg1,
                 attn_mask=_amask_upto(1), kv_write_mask=_wmask(1))

    # ---- A: persistent state, pos0 then pos1 ----
    sA = m.make_state()
    out_A0 = m.predict(dict(base0, x=x0), state=sA)
    out_A1 = m.predict(dict(base1, x=x1), state=sA)
    h_A0 = np.asarray(out_A0["hidden"]).astype(np.float32).reshape(D_MODEL)
    h_A1 = np.asarray(out_A1["hidden"]).astype(np.float32).reshape(D_MODEL)

    # ---- B: fresh state, pos1 alone (skip pos0) ----
    sB = m.make_state()
    out_B1 = m.predict(dict(base1, x=x1), state=sB)
    h_B1 = np.asarray(out_B1["hidden"]).astype(np.float32).reshape(D_MODEL)

    # ---- C: persistent state, pos1 with mask hiding slot 0 (only slot 1) ----
    sC = m.make_state()
    _ = m.predict(dict(base0, x=x0), state=sC)  # populate slot 0
    base1_only1 = dict(base1, attn_mask=_amask_only(1))
    out_C1 = m.predict(dict(base1_only1, x=x1), state=sC)
    h_C1 = np.asarray(out_C1["hidden"]).astype(np.float32).reshape(D_MODEL)

    # ---- D: same as A but with x1==x0 (echo input twice) ----
    sD = m.make_state()
    _ = m.predict(dict(base0, x=x0), state=sD)
    out_D1 = m.predict(dict(base1, x=x0), state=sD)
    h_D1 = np.asarray(out_D1["hidden"]).astype(np.float32).reshape(D_MODEL)

    print()
    print(f"  ‖h_A0‖     = {_norm(h_A0):.3f}   (pos=0 with state)")
    print(f"  ‖h_A1‖     = {_norm(h_A1):.3f}   (pos=1 with persisted state)")
    print(f"  ‖h_B1‖     = {_norm(h_B1):.3f}   (pos=1 fresh state, no past)")
    print(f"  ‖h_C1‖     = {_norm(h_C1):.3f}   (pos=1 mask hides slot 0)")
    print(f"  ‖h_D1‖     = {_norm(h_D1):.3f}   (pos=1 with state, same x as pos=0)")
    print()
    print(f"  cos(A0, A1)         = {_cos(h_A0, h_A1):+.4f}   "
          "(low = layers transformed pos1 differently from pos0)")
    print(f"  cos(A1, B1)         = {_cos(h_A1, h_B1):+.4f}   "
          "(==1 means past was IGNORED; <1 means attention used past)")
    print(f"  cos(A1, C1)         = {_cos(h_A1, h_C1):+.4f}   "
          "(if ==1 then state slot 0 was unused even when allowed)")
    print(f"  cos(B1, C1)         = {_cos(h_B1, h_C1):+.4f}   "
          "(should be ~=1 — both effectively see only slot 1)")
    print(f"  cos(A0, D1)         = {_cos(h_A0, h_D1):+.4f}   "
          "(if ~=1 then layers truly are no-op when x repeats)")
    print()
    print("  >>> Diagnosis:")
    if _cos(h_A1, h_B1) > 0.999:
        print("      STATE READ IS BROKEN: pos=1 with vs without past = identical.")
    elif _cos(h_A1, h_C1) > 0.999:
        print("      ATTN MASK INEFFECTIVE: persistent state ignores slot 0 even "
              "when mask allows it.")
    elif _cos(h_A1, h_B1) > 0.99:
        print("      State read working but past contributes only weakly "
              "(quantization noise dominating).")
    else:
        print("      State read is functioning. Collapse must be elsewhere "
              "(scale, RoPE, weights, head).")


if __name__ == "__main__":
    main()
