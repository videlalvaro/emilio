"""Routing-locality probe: are top-4 expert sets stable across prompts?

Runs the opf model on N independent prompt batches (PII-style, generic,
code, multilingual...) and compares the popularity ranking of top-4 sets
in layer 0.

Decision rule:
  - If the top-30 sets cover >70% of rows on EVERY prompt batch, set-cache wins.
  - If coverage drops below 30% on any new batch, set-cache fails; use shard-dispatch.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights"
SEQ_LEN = 128

# 4 prompt batches representing different domains.
BATCHES = {
    "pii_original": [
        "Alice Smith was born on 1990-01-02 in Springfield.",
        "Email me at alice.smith@example.com if you have questions.",
        "Call Bob at +1-415-555-0142 between 9 AM and 5 PM.",
        "His home address is 742 Evergreen Terrace, Springfield, IL.",
        "Wire the funds to account 0001234567 at First National Bank.",
        "Visit https://example.com/dashboard?token=abc for details.",
        "The API key is sk-proj-FAKE0000000000000000000000000000000 for staging.",
        "Carol's appointment is scheduled for March 15, 2026 at 10:30 AM.",
    ],
    "pii_paraphrase": [
        "Bob Jones, DOB 1985-11-30, residing in Portland.",
        "Reach out to bob.jones@acme.org for inquiries.",
        "Phone Karen at +44 20 7946 0958 during business hours.",
        "Mailing address: 221B Baker Street, London, NW1 6XE.",
        "Transfer funds to IBAN GB29 NWBK 6016 1331 9268 19.",
        "See https://internal.acme.org/admin?session=xyz123 now.",
        "Token: sk-test-AbCdEfGhIj1234567890QwErTyUiOpAsDfGh value.",
        "Meeting on April 22, 2026 at 14:00 with the new client.",
    ],
    "generic_text": [
        "The quick brown fox jumps over the lazy dog repeatedly.",
        "She sells seashells by the seashore in summer.",
        "Climate change affects polar ice caps and global weather.",
        "Reading is fundamental to lifelong learning and growth.",
        "The ancient library held countless rare manuscripts inside.",
        "Mountains rise majestically from the valley floor below.",
        "Coffee shops bustle with creative energy each morning.",
        "Music transcends language and connects people worldwide.",
    ],
    "code_snippets": [
        "def add(a, b): return a + b",
        "for i in range(10): print(i * 2)",
        "import numpy as np; arr = np.zeros((3, 3))",
        "class Foo: def __init__(self): self.x = 0",
        "try: result = json.loads(text) except: pass",
        "if x > 0 and y < 100: result = compute(x, y)",
        "lambda x: x ** 2 + 2 * x + 1",
        "with open('file.txt', 'r') as f: data = f.read()",
    ],
}


def main() -> int:
    torch.manual_seed(0)
    torch.set_num_threads(8)

    from opf._model.model import Transformer
    import tiktoken

    cfg = json.loads((WEIGHTS_DIR / "config.json").read_text())
    enc = tiktoken.get_encoding(cfg["encoding"])
    pad_id = int(enc.eot_token)

    print("[probe] loading model")
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=torch.device("cpu"))
    model.eval()

    mlp = model.block[0].mlp
    captures: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        captures["norm_out"] = out.detach().to(torch.float32).cpu()
    mlp.norm.register_forward_hook(hook)

    gate_w = mlp.gate.weight.detach().to(torch.float32).cpu()
    gate_b = mlp.gate.bias.detach().to(torch.float32).cpu()

    # Per-batch top-4 set counters
    per_batch_sets: dict[str, Counter] = {}
    per_batch_total_rows: dict[str, int] = {}

    for name, sentences in BATCHES.items():
        ids = np.full((len(sentences), SEQ_LEN), pad_id, dtype=np.int64)
        mask = np.zeros((len(sentences), SEQ_LEN), dtype=np.int64)
        for i, s in enumerate(sentences):
            t = enc.encode(s, allowed_special="all")[:SEQ_LEN]
            ids[i, :len(t)] = t
            mask[i, :len(t)] = 1
        with torch.inference_mode():
            _ = model(torch.from_numpy(ids), attention_mask=torch.from_numpy(mask))
        gate = torch.nn.functional.linear(captures["norm_out"], gate_w, gate_b)
        topk = torch.topk(gate, k=4, dim=-1).indices  # [B, T, 4]
        valid = torch.from_numpy(mask.astype(bool))
        rows = topk[valid]  # [N_valid, 4]
        c = Counter(tuple(sorted(r.tolist())) for r in rows)
        per_batch_sets[name] = c
        per_batch_total_rows[name] = int(rows.shape[0])
        print(f"[probe] {name:18s}  rows={rows.shape[0]:4d}  unique sets={len(c)}")

    # Build a "global popularity" ranking from the original PII batch only,
    # then test what fraction of OTHER batches' rows it covers at each cache size.
    base = per_batch_sets["pii_original"]
    top_sets_ranked = [s for s, _ in base.most_common()]

    print()
    print(f"=== Set-cache stability test (cache built from pii_original) ===")
    print(f"{'cache_size':>10s} | " + " | ".join(f"{n:>15s}" for n in BATCHES))
    print("-" * 80)
    for cache_size in (10, 30, 50, 100, 192):
        cache = set(top_sets_ranked[:cache_size])
        line = f"{cache_size:>10d} | "
        cov_per = []
        for name, c in per_batch_sets.items():
            covered = sum(cnt for s, cnt in c.items() if s in cache)
            pct = 100.0 * covered / per_batch_total_rows[name]
            cov_per.append(pct)
            line += f"{pct:>14.1f}% | "
        print(line)

    print()
    print("=== Set-overlap matrix (Jaccard of top-30 sets) ===")
    names = list(BATCHES.keys())
    top30 = {n: set(s for s, _ in c.most_common(30)) for n, c in per_batch_sets.items()}
    print(" " * 18 + " | " + " | ".join(f"{n:>15s}" for n in names))
    for a in names:
        line = f"{a:>18s} | "
        for b in names:
            inter = len(top30[a] & top30[b])
            union = len(top30[a] | top30[b])
            line += f"{100*inter/union:>14.1f}% | "
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
