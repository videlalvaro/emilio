"""diff_p2_layers_v3.py — test if final-norm uses gamma DIRECTLY (no (1+x)) for Gemma-4.

Hypothesis: our final_norm_gamma has mean ~29.22, suggesting it is meant to be
applied directly as the post-RMSNorm scale (not (1+gamma)). Tests both recipes
on HF_L29 and on CML_B3, scoring against HF's published last_pos_logits.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

OUT = Path("python/moe/out")
HEAD = OUT / "gemma_logit_head.npz"

def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(a @ b / (na * nb))

def main():
    head = np.load(HEAD, allow_pickle=False)
    gamma = head["final_norm_gamma"].astype(np.float64)
    eps = float(head["rms_norm_eps"])
    embed = head["embed_weight"]
    softcap = float(head["softcap"])

    hf = np.load(OUT / "diag_p2_hf_layers.npz")
    cm = np.load(OUT / "diag_p2_cml_layers.npz")
    p2 = np.load(OUT / "diag_p2_hf_top50.npz")

    hf_L29 = hf["hidden_last_pos_per_layer"][29].astype(np.float64)
    cm_B3  = cm["hidden_last_pos_per_boundary"][3].astype(np.float64)
    hf_pub = p2["last_pos_logits"].astype(np.float64)

    def head_apply(h, mode):
        rms = np.sqrt((h * h).mean() + eps)
        if mode == "1plus":   scale = 1.0 + gamma
        elif mode == "direct": scale = gamma
        elif mode == "raw":    scale = np.ones_like(gamma)
        h_norm = (h / rms) * scale
        out = embed.astype(np.float64) @ h_norm
        if softcap > 0:
            out = np.tanh(out / softcap) * softcap
        return out

    print(f"gamma stats: min={gamma.min():.3f}  max={gamma.max():.3f}  "
          f"mean={gamma.mean():.3f}  std={gamma.std():.3f}  "
          f"frac_near_zero={(np.abs(gamma)<0.5).mean():.3f}  "
          f"frac_near_30={(np.abs(gamma-30)<5).mean():.3f}")
    print(f"softcap={softcap}  eps={eps}  embed shape={embed.shape}  embed dtype={embed.dtype}")
    print()

    print("=== Apply each recipe to HF_L29; cos vs HF published logits ===")
    for mode in ("1plus", "direct", "raw"):
        L = head_apply(hf_L29, mode)
        c = cos(L, hf_pub)
        print(f"  mode={mode:>7}  max={L.max():>7.3f}  min={L.min():>7.3f}  cos(L, HF_pub)={c:+.4f}  "
              f"top1_id={int(np.argmax(L))}")

    print()
    print("=== Apply each recipe to CML_B3; cos vs HF published logits ===")
    for mode in ("1plus", "direct", "raw"):
        L = head_apply(cm_B3, mode)
        c = cos(L, hf_pub)
        # Find ' George' and ' a' ranks
        order = np.argsort(-L)
        rk_george = int(np.where(order == 9142)[0][0])
        rk_a = int(np.where(order == 496)[0][0])
        print(f"  mode={mode:>7}  max={L.max():>7.3f}  min={L.min():>7.3f}  cos={c:+.4f}  "
              f"top1={int(order[0])}  George_rank={rk_george}  a_rank={rk_a}")

    # Also: try our current CML "head" (1plus) on CML_B3 and compare to the
    # actual published CML logits (sanity)
    cm_pub = np.load(OUT / "diag_p2_cml_top50.npz")["last_pos_logits"].astype(np.float64)
    print()
    print("=== Sanity: head_apply(CML_B3, 1plus) vs published CML logits ===")
    L = head_apply(cm_B3, "1plus")
    print(f"  cos(head(CML_B3,1plus), CML_pub) = {cos(L, cm_pub):+.4f}  "
          f"max diff = {np.abs(L - cm_pub).max():.3f}")

if __name__ == "__main__":
    main()
