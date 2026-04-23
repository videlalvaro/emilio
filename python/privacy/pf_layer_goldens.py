"""T2a: per-block goldens for layer 0 of openai/privacy-filter.

Captures the residual stream after embedding, attention block 0, and MLP block 0
for the same 8 sentences as `python/privacy/pf_ref.py`. Used by T2 and T3 cosine
gates against CoreML conversions.

Mirrors golden-npz contract from `python/moe/GEMMA_ANE_RESEARCH.md`.
Pins: opf src 2e8c95b9, HF revision 7ffa9a04 (stamped from pf_ref).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights"
GOLDEN_DIR = REPO_ROOT / "python" / "privacy" / "out"
REF_GOLDEN = GOLDEN_DIR / "pf_golden.npz"
LAYER0_GOLDEN = GOLDEN_DIR / "pf_layer0_goldens.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not REF_GOLDEN.exists():
        raise SystemExit(f"Missing {REF_GOLDEN}. Run T1 (pf_ref.py) first.")
    if LAYER0_GOLDEN.exists() and not args.force:
        raise SystemExit(f"{LAYER0_GOLDEN} exists. Use --force to overwrite.")

    z = np.load(REF_GOLDEN, allow_pickle=False)
    input_ids_np = z["input_ids"]
    attn_np = z["attention_mask"]
    model_sha = str(z["model_sha"])
    opf_src_sha = str(z["opf_src_sha"])

    torch.manual_seed(0)
    torch.set_num_threads(min(8, os.cpu_count() or 8))

    from opf._model.model import Transformer  # vendored, audited

    device = torch.device("cpu")
    print(f"[t2a] loading model from {WEIGHTS_DIR}")
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=device)
    model.eval()

    captures: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(module, inputs, output):
            captures[name] = output.detach().to(torch.float32).cpu()
        return hook

    handles = [
        model.embedding.register_forward_hook(make_hook("embedding_out")),
        model.block[0].attn.register_forward_hook(make_hook("attn0_out")),
        model.block[0].mlp.register_forward_hook(make_hook("mlp0_out")),
    ]

    input_ids = torch.from_numpy(input_ids_np.astype(np.int64))
    attn = torch.from_numpy(attn_np.astype(np.int64))

    print(f"[t2a] forward B={input_ids.shape[0]} T={input_ids.shape[1]}")
    with torch.inference_mode():
        _ = model(input_ids, attention_mask=attn)
    for h in handles:
        h.remove()

    for k, v in captures.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} "
              f"min={v.min().item():.4f} max={v.max().item():.4f} "
              f"mean={v.mean().item():.6f}")

    np.savez_compressed(
        LAYER0_GOLDEN,
        embedding_out=captures["embedding_out"].numpy(),
        attn0_out=captures["attn0_out"].numpy(),
        mlp0_out=captures["mlp0_out"].numpy(),
        input_ids=input_ids_np,
        attention_mask=attn_np,
        model_sha=np.str_(model_sha),
        opf_src_sha=np.str_(opf_src_sha),
    )
    size_mb = LAYER0_GOLDEN.stat().st_size / 1024**2
    print(f"[t2a] wrote {LAYER0_GOLDEN} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
