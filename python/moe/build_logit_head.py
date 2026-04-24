"""build_logit_head.py — extract embed/final-norm/softcap metadata for T4.1.4.

Reads the gemma-4-26b-a4b safetensors and saves to npz:
  embed_weight: (vocab=262144, D=2816) fp16
  final_norm_gamma: (D,) fp16
  rms_norm_eps: float
  softcap: float
  tie_word_embeddings: bool

Hard-aborts if tie_word_embeddings != True (per gatekeeper C2 caveat — we rely
on embed_tokens.weight as lm_head).

Output: python/moe/out/gemma_logit_head.npz  (~1.5 GB fp16)

Runs in any python (CPU only). No model load — uses safetensors index +
selective tensor reads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT_PATH = Path("python/moe/out/gemma_logit_head.npz")

# Gemma 4 weight key conventions (verified via model.safetensors.index.json).
EMBED_KEY = "model.language_model.embed_tokens.weight"
FINAL_NORM_KEY = "model.language_model.norm.weight"


def _resolve_shard(index_path: Path, key: str) -> Path:
    idx = json.loads(index_path.read_text())["weight_map"]
    if key not in idx:
        raise KeyError(f"{key!r} not in safetensors index. "
                       f"Available samples: {list(idx)[:5]}")
    return MODEL_DIR / idx[key]


def main():
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    tie = bool(tc.get("tie_word_embeddings"))
    if not tie:
        print(f"FATAL tie_word_embeddings={tie}; T4.1.4 plan assumes tied. "
              "Aborting per gatekeeper.", file=sys.stderr)
        sys.exit(2)
    vocab = int(tc["vocab_size"])
    d_model = int(tc["hidden_size"])
    eps = float(tc["rms_norm_eps"])
    softcap = float(tc["final_logit_softcapping"])
    print(f"  vocab={vocab}  D={d_model}  eps={eps}  softcap={softcap}  tied={tie}")

    index = MODEL_DIR / "model.safetensors.index.json"
    embed_shard = _resolve_shard(index, EMBED_KEY)
    norm_shard  = _resolve_shard(index, FINAL_NORM_KEY)
    print(f"  embed_shard: {embed_shard.name}")
    print(f"  norm_shard:  {norm_shard.name}")

    import torch
    with safe_open(str(embed_shard), framework="pt", device="cpu") as f:
        embed = f.get_tensor(EMBED_KEY)
    with safe_open(str(norm_shard), framework="pt", device="cpu") as f:
        gamma = f.get_tensor(FINAL_NORM_KEY)

    embed_np = embed.to(dtype=torch.float16).numpy()
    gamma_np = gamma.to(dtype=torch.float16).numpy()

    assert embed_np.shape == (vocab, d_model), \
        f"embed shape {embed_np.shape} != ({vocab},{d_model})"
    assert gamma_np.shape == (d_model,), \
        f"gamma shape {gamma_np.shape} != ({d_model},)"
    assert np.isfinite(embed_np).all(), "non-finite in embed"
    assert np.isfinite(gamma_np).all(), "non-finite in final_norm gamma"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(OUT_PATH),
        embed_weight=embed_np,
        final_norm_gamma=gamma_np,
        rms_norm_eps=np.array(eps, dtype=np.float64),
        softcap=np.array(softcap, dtype=np.float64),
        tie_word_embeddings=np.array(tie, dtype=bool),
    )
    sz_mb = OUT_PATH.stat().st_size / 1e6
    print(f"  saved -> {OUT_PATH}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
