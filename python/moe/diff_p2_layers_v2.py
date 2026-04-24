"""diff_p2_layers_v2.py — fair comparison after recognizing HF L30 = post-final-norm.

Compares:
  (a) cos(HF L29 pre-norm, CML B3 pre-norm) — direction drift through 30 INT4 layers
  (b) cos(applyFinalNorm(CML B3), HF L30 post-norm) — direction drift after both normed
  (c) cos(applyFinalNorm(HF L29), HF L30) — sanity check that HF L30 IS post-norm
  (d) magnitude ratio of pre-norm hidden states
  (e) post-norm magnitudes
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

OUT = Path("python/moe/out")
HF = OUT / "diag_p2_hf_layers.npz"
CM = OUT / "diag_p2_cml_layers.npz"
HEAD = OUT / "gemma_logit_head.npz"

def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(a @ b / (na * nb))

def apply_rms_gamma(h, gamma, eps):
    h = h.astype(np.float64)
    rms = np.sqrt((h * h).mean() + eps)
    return (h / rms) * (1.0 + gamma.astype(np.float64))

def main():
    hf = np.load(HF); cm = np.load(CM)
    head = np.load(HEAD, allow_pickle=False)
    gamma = head["final_norm_gamma"]
    eps = float(head["rms_norm_eps"])
    print(f"final_norm_gamma stats: mean={gamma.mean():.4f}  "
          f"mean(1+gamma)={(1+gamma).mean():.4f}  "
          f"||1+gamma||={np.linalg.norm(1+gamma.astype(np.float64)):.4f}")
    print(f"rms_norm_eps = {eps}")

    hf_L29 = hf["hidden_last_pos_per_layer"][29]    # pre-final-norm last layer
    hf_L30 = hf["hidden_last_pos_per_layer"][30]    # POST-final-norm (mostly)
    cm_B3  = cm["hidden_last_pos_per_boundary"][3]  # pre-final-norm shard 2 out

    print()
    print("=== (a) Pre-final-norm direction drift HF L29 vs CML B3 ===")
    print(f"  ‖HF_L29‖ = {np.linalg.norm(hf_L29):.3f}")
    print(f"  ‖CML_B3‖ = {np.linalg.norm(cm_B3):.3f}  (ratio CML/HF = {np.linalg.norm(cm_B3)/np.linalg.norm(hf_L29):.4f})")
    print(f"  cos(HF_L29, CML_B3) = {cos(hf_L29, cm_B3):+.4f}")

    hf_L29_norm = apply_rms_gamma(hf_L29, gamma, eps)
    cm_B3_norm  = apply_rms_gamma(cm_B3,  gamma, eps)
    print()
    print("=== (b) Post-final-norm direction (applied locally) ===")
    print(f"  ‖norm(HF_L29)‖ = {np.linalg.norm(hf_L29_norm):.3f}")
    print(f"  ‖norm(CML_B3)‖ = {np.linalg.norm(cm_B3_norm):.3f}")
    print(f"  cos(norm(HF_L29), norm(CML_B3)) = {cos(hf_L29_norm, cm_B3_norm):+.4f}")

    print()
    print("=== (c) Sanity: cos(norm(HF_L29), HF_L30) — is HF_L30 truly post-norm? ===")
    print(f"  ‖HF_L30‖ = {np.linalg.norm(hf_L30):.3f}  vs ‖norm(HF_L29)‖ = {np.linalg.norm(hf_L29_norm):.3f}")
    print(f"  cos = {cos(hf_L29_norm, hf_L30):+.4f}  (≈1.0 confirms HF_L30 is post-norm)")

    print()
    print("=== (d) cos vs HF_L30 (post-norm) ===")
    print(f"  cos(norm(CML_B3), HF_L30) = {cos(cm_B3_norm, hf_L30):+.4f}")

    # Compute logits both ways and report magnitude scale
    embed = head["embed_weight"]; softcap = float(head["softcap"])
    def logits_from_norm(hn):
        e = embed.astype(np.float64)
        out = e @ hn.astype(np.float64)
        return np.tanh(out / softcap) * softcap
    L_hf_self = logits_from_norm(hf_L29_norm)
    L_cml_self = logits_from_norm(cm_B3_norm)
    print()
    print("=== (e) Logit scale comparison (using OUR final-norm + softcap recipe) ===")
    print(f"  HF_L29→logits   max={L_hf_self.max():.3f}  top1_id={int(np.argmax(L_hf_self))}")
    print(f"  CML_B3→logits   max={L_cml_self.max():.3f}  top1_id={int(np.argmax(L_cml_self))}")
    print(f"  diff in max logit = {L_cml_self.max() - L_hf_self.max():+.3f}  (positive = CML hotter)")

    # Compare to HF's published last_pos_logits (with HF's own final-norm)
    p2 = np.load(OUT / "diag_p2_hf_top50.npz")
    L_hf_pub = p2["last_pos_logits"]
    print()
    print("=== (f) Sanity: our HF→logits recipe vs HF model's published logits ===")
    print(f"  HF_pub max = {L_hf_pub.max():.3f}  top1_id={int(np.argmax(L_hf_pub))}")
    print(f"  cos(L_hf_self, L_hf_pub) = {cos(L_hf_self, L_hf_pub):+.4f}  (≈1.0 confirms recipe matches HF)")

if __name__ == "__main__":
    main()
