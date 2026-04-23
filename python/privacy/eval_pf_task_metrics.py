"""Task-level evaluation: privacy-filter NER span agreement.

Compares Swift/ANE per-token labels (dumped by pf_e2e.swift) against the
reference (PyTorch bf16) per-token labels for the same 8 sentences.

What we report (paper-quality):
  - per-token argmax agreement (= what cos≈1 implies, but unitful)
  - per-entity span exact-match precision/recall/F1 (treat reference as oracle)
  - per-sentence "all spans identical" rate
  - decoded entities side-by-side for any sentence where they differ

This is the honest task-level number. cos floor of 0.9935 is meaningless to a
reader; "ANE matches the reference's entity decisions on 100% of spans" is not.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "python/privacy/out/pf_golden.npz"
SWIFT  = ROOT / "python/privacy/out/swift_dump/swift_argmax_labels.bin"

T_SEQ, N_LABELS = 128, 33


def decode_entities(labels: list[str]) -> list[tuple[int, int, str]]:
    """BIOES -> list of (start_tok, end_tok_inclusive, entity_type)."""
    spans: list[tuple[int, int, str]] = []
    i, n = 0, len(labels)
    while i < n:
        tag = labels[i]
        if tag == "O" or tag == "<pad>":
            i += 1
            continue
        if "-" not in tag:
            i += 1
            continue
        prefix, etype = tag.split("-", 1)
        if prefix == "S":
            spans.append((i, i, etype)); i += 1; continue
        if prefix == "B":
            j = i + 1
            while j < n and labels[j].endswith("-" + etype) and labels[j].split("-", 1)[0] in ("I", "E"):
                if labels[j].startswith("E-"):
                    j += 1; break
                j += 1
            spans.append((i, j - 1, etype)); i = j; continue
        # stray I- or E- — skip (model error against schema)
        i += 1
    return spans


def main() -> None:
    g = np.load(GOLDEN)
    id2label = json.loads(str(g["id2label_json"]))
    id2label = {int(k): v for k, v in id2label.items()}
    ref_labels = g["argmax_labels"]            # [8,128] reference argmax
    mask = g["attention_mask"]                 # [8,128]
    n_sent = ref_labels.shape[0]

    swift = np.fromfile(SWIFT, dtype=np.int32).reshape(n_sent, T_SEQ)

    print(f"{'sent':>4} {'tok_match':>10} {'ref_spans':>10} {'sw_spans':>10} {'span_TP':>8} {'span_F1':>8}")
    print("-" * 60)

    total_tok, ok_tok = 0, 0
    total_tp, total_pred, total_ref = 0, 0, 0
    sent_full_match = 0
    diffs: list[tuple[int, list, list]] = []

    for b in range(n_sent):
        m = mask[b].astype(bool)
        valid = int(m.sum())
        ref_lab = [id2label[int(x)] for x in ref_labels[b][:valid]]
        sw_lab  = [id2label[int(x)] for x in swift[b][:valid]]

        tok_ok = sum(1 for r, s in zip(ref_lab, sw_lab) if r == s)
        total_tok += valid; ok_tok += tok_ok

        ref_sp = set(decode_entities(ref_lab))
        sw_sp  = set(decode_entities(sw_lab))
        tp = len(ref_sp & sw_sp)
        total_tp += tp
        total_pred += len(sw_sp); total_ref += len(ref_sp)

        f1 = (2 * tp / (len(ref_sp) + len(sw_sp))) if (len(ref_sp) + len(sw_sp)) else 1.0
        if ref_sp == sw_sp:
            sent_full_match += 1
        else:
            diffs.append((b, sorted(ref_sp), sorted(sw_sp)))

        print(f"{b:>4} {tok_ok:>4}/{valid:<4}  {len(ref_sp):>10} {len(sw_sp):>10} {tp:>8} {f1:>8.3f}")

    P = total_tp / total_pred if total_pred else 0.0
    R = total_tp / total_ref  if total_ref  else 0.0
    F = 2 * P * R / (P + R)   if (P + R)    else 0.0
    print("-" * 60)
    print(f"per-token agreement : {ok_tok}/{total_tok} = {ok_tok/total_tok*100:.2f}%")
    print(f"span precision      : {total_tp}/{total_pred} = {P*100:.2f}%")
    print(f"span recall         : {total_tp}/{total_ref} = {R*100:.2f}%")
    print(f"span F1             : {F*100:.2f}%")
    print(f"sentences w/ identical span set: {sent_full_match}/{n_sent}")

    if diffs:
        print("\n[diff] per-sentence span differences:")
        for b, r, s in diffs:
            print(f"  s{b}:")
            print(f"    ref:    {r}")
            print(f"    swift:  {s}")


if __name__ == "__main__":
    main()
