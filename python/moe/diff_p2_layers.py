"""diff_p2_layers.py — diff HF vs CML hidden state norms + cosines at boundaries.

Reads diag_p2_hf_layers.npz (per-layer) and diag_p2_cml_layers.npz (per-shard).
Compares at boundaries L0(embed), L15, L22, L30 (post-final-decoder, pre-norm).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

HF = Path("python/moe/out/diag_p2_hf_layers.npz")
CM = Path("python/moe/out/diag_p2_cml_layers.npz")

def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def main():
    hf = np.load(HF); cm = np.load(CM)
    assert (hf["input_ids"] == cm["input_ids"]).all(), "prompt mismatch"
    T = hf["input_ids"].shape[0]
    print(f"prompt T={T}")

    bnd = cm["boundary_layers"].tolist()  # [0, 15, 22, 30]
    print(f"boundaries: {bnd}")

    print()
    print("=== ‖h_last_pos‖ at each boundary ===")
    print(f"{'layer':>6} | {'HF':>10} | {'CML':>10} | {'CML/HF':>8} | {'cos':>8}")
    for b_idx, layer in enumerate(bnd):
        hf_h = hf["hidden_last_pos_per_layer"][layer]
        cm_h = cm["hidden_last_pos_per_boundary"][b_idx]
        hf_n = float(np.linalg.norm(hf_h))
        cm_n = float(np.linalg.norm(cm_h))
        c = cos(hf_h, cm_h)
        print(f"{layer:>6} | {hf_n:>10.3f} | {cm_n:>10.3f} | {cm_n/hf_n:>8.4f} | {c:>+8.4f}")

    print()
    print("=== full per-layer ‖h_last_pos‖ from HF (every layer) ===")
    hf_norms = hf["hidden_norms"][-1]  # (L+1,)
    L1 = hf_norms.shape[0]
    for li in range(L1):
        marker = " <-- shard boundary" if li in bnd else ""
        print(f"  L{li:>2}  ‖h‖ = {hf_norms[li]:>10.3f}{marker}")

    print()
    print("=== ‖h‖ growth ratio HF layer-to-layer (peek for runaway) ===")
    for li in range(1, L1):
        r = hf_norms[li] / max(hf_norms[li-1], 1e-9)
        flag = "  *" if r > 1.2 or r < 0.83 else ""
        print(f"  L{li-1:>2}->L{li:<2}  x{r:.3f}{flag}")

if __name__ == "__main__":
    main()
