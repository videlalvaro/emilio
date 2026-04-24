"""diff_p2_diag.py — analyze HF vs CML top-50 logit dumps for prompt 2."""
from __future__ import annotations
import numpy as np
import tokenizers

tok = tokenizers.Tokenizer.from_file("models/gemma-4-26b-a4b/tokenizer.json")
hf = np.load("python/moe/out/diag_p2_hf_top50.npz", allow_pickle=False)
cm = np.load("python/moe/out/diag_p2_cml_top50.npz", allow_pickle=False)

hf_l = hf["last_pos_logits"]; cm_l = cm["last_pos_logits"]
hf_ids = hf["top50_ids"]; cm_ids = cm["top50_ids"]

print("=== HF top-20 ===")
for r in range(20):
    tid = int(hf_ids[r]); l = float(hf["top50_logits"][r])
    s = tok.decode([tid], skip_special_tokens=False)
    print(f"  [{r:2d}] id={tid:7d} L={l:7.3f}  {s!r}")

print("=== CML top-20 ===")
for r in range(20):
    tid = int(cm_ids[r]); l = float(cm["top50_logits"][r])
    s = tok.decode([tid], skip_special_tokens=False)
    print(f"  [{r:2d}] id={tid:7d} L={l:7.3f}  {s!r}")

GEORGE = 9142; A = 496
hf_top1 = int(hf_ids[0]); cm_top1 = int(cm_ids[0])
print()
print(f"HF  top1: {tok.decode([hf_top1])!r:14s} L={hf_l[hf_top1]:7.3f}   "
      f"' George' L={hf_l[GEORGE]:7.3f}   ' a' L={hf_l[A]:7.3f}")
print(f"CML top1: {tok.decode([cm_top1])!r:14s} L={cm_l[cm_top1]:7.3f}   "
      f"' George' L={cm_l[GEORGE]:7.3f}   ' a' L={cm_l[A]:7.3f}")
print(f"HF  gap (top1 - runner_up) = "
      f"{hf_l[hf_top1] - hf_l[int(hf_ids[1])]:.3f}")
print(f"CML gap (top1 - runner_up) = "
      f"{cm_l[cm_top1] - cm_l[int(cm_ids[1])]:.3f}")

def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

print()
print(f"cos(HF, CML) full vocab = {cos(hf_l, cm_l):.4f}")
hf20 = set(int(x) for x in hf_ids[:20])
cm20 = set(int(x) for x in cm_ids[:20])
print(f"top-20 set overlap = {len(hf20 & cm20)}/20")
hf50 = set(int(x) for x in hf_ids)
cm50 = set(int(x) for x in cm_ids)
print(f"top-50 set overlap = {len(hf50 & cm50)}/50")

# Where does each top-5 land in the other?
print()
print("HF top-5 → CML rank:")
cm_full_order = np.argsort(-cm_l)
cm_rank = {int(t): r for r, t in enumerate(cm_full_order[:1000])}
for r in range(5):
    tid = int(hf_ids[r])
    cr = cm_rank.get(tid, ">1000")
    print(f"  HF[{r}] id={tid:7d} {tok.decode([tid])!r:14s} → CML rank={cr}")
print("CML top-5 → HF rank:")
hf_full_order = np.argsort(-hf_l)
hf_rank = {int(t): r for r, t in enumerate(hf_full_order[:1000])}
for r in range(5):
    tid = int(cm_ids[r])
    hr = hf_rank.get(tid, ">1000")
    print(f"  CML[{r}] id={tid:7d} {tok.decode([tid])!r:14s} → HF rank={hr}")
