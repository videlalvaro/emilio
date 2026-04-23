"""T2d: validate PF_attn0_T128.mlpackage against pf_layer0_goldens.npz.

For each of the 8 sentences:
  - reshape embedding_out[i] from (T,640) to (1,640,1,T) for channels-second.
  - build pad_add: 0 at valid positions, -inf at padded.
  - call CoreML model.
  - compute cosine vs attn0_out[i] over valid positions only.

Pass/fail gate: per-sentence cos >= 0.97 (project rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = REPO_ROOT / "python" / "privacy" / "out" / "pf_layer0_goldens.npz"
MLPKG = REPO_ROOT / "emilio" / "conv-ane" / "PF_attn0_T128.mlpackage"


def main() -> int:
    import coremltools as ct

    z = np.load(GOLDEN, allow_pickle=False)
    emb = z["embedding_out"].astype(np.float32)        # (8, 128, 640)
    ref = z["attn0_out"].astype(np.float32)            # (8, 128, 640)
    mask = z["attention_mask"].astype(np.int32)        # (8, 128)
    B, T, D = emb.shape
    print(f"[t2d] B={B} T={T} D={D}")

    print(f"[t2d] loading {MLPKG}")
    model = ct.models.MLModel(str(MLPKG), compute_units=ct.ComputeUnit.ALL)

    cosines: list[float] = []
    for i in range(B):
        n_valid = int(mask[i].sum())
        # x_in: (1, 640, 1, 128) channels-second from (128, 640)
        x_in = emb[i].T[None, :, None, :].astype(np.float16)
        # pad_add: (1, 1, 1, 128); -inf where padded
        pad_add = np.where(mask[i] > 0, 0.0, -1e4).astype(np.float16)[None, None, None, :]
        out = model.predict({"x_in": x_in, "pad_add": pad_add})
        y_arr = out["x_out"]  # (1,640,1,128)
        # back to (T, D)
        y = y_arr.reshape(D, T).T.astype(np.float32)
        # cosine on valid positions only, flattened
        a = ref[i, :n_valid, :].reshape(-1)
        b = y[:n_valid, :].reshape(-1)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        max_abs = float(np.abs(a - b).max())
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
        cosines.append(cos)
        print(f"  [{i}] n_valid={n_valid:3d}  cos={cos:.6f}  max|Δ|={max_abs:.4f}  relL2={rel:.4f}")

    worst = min(cosines)
    mean = float(np.mean(cosines))
    print(f"[t2d] worst={worst:.6f}  mean={mean:.6f}")
    if worst < 0.97:
        print("[t2d] FAIL: worst cos < 0.97 gate")
        return 1
    print("[t2d] PASS: cos >= 0.97 on every sentence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
