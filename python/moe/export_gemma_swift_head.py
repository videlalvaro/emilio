"""export_gemma_swift_head.py — Swift-runtime export for Gemma logit head.

The Python-side Gemma gate uses `python/moe/out/gemma_logit_head.npz`, which is
convenient for NumPy but awkward for a Swift runtime. This helper converts the
NPZ into raw fp16 binaries plus a compact JSON metadata file so a future
`gemma_ane.swift` can memory-map the final logit head directly.

Outputs (default: `python/moe/out/`):
  - `gemma_embed_fp16.bin`              (vocab, d_model) fp16 row-major
  - `gemma_final_norm_gamma_fp16.bin`   (d_model,) fp16
  - `gemma_swift_head_meta.json`        small runtime metadata blob

Run:
  .venv313/bin/python python/moe/export_gemma_swift_head.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

IN_NPZ = Path("python/moe/out/gemma_logit_head.npz")
OUT_DIR = Path("python/moe/out")

EMBED_BIN = "gemma_embed_fp16.bin"
GAMMA_BIN = "gemma_final_norm_gamma_fp16.bin"
META_JSON = "gemma_swift_head_meta.json"

TOKENIZER_JSON = "models/gemma-4-26b-a4b/tokenizer.json"
SHARDS = [
    {"start": 0, "end": 8, "path": "python/moe/out/gemma4_shard0_8_real.mlmodelc"},
    {"start": 8, "end": 15, "path": "python/moe/out/gemma4_shard8_15_real.mlmodelc"},
    {"start": 15, "end": 22, "path": "python/moe/out/gemma4_shard15_22_real.mlmodelc"},
    {"start": 22, "end": 30, "path": "python/moe/out/gemma4_shard22_30_real.mlmodelc"},
]


def _atomic_write_bytes(path: Path, blob: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_NPZ,
                    help="input NPZ from build_logit_head.py")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="output directory for Swift runtime artifacts")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing outputs")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"FATAL missing input NPZ: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    embed_path = args.out_dir / EMBED_BIN
    gamma_path = args.out_dir / GAMMA_BIN
    meta_path = args.out_dir / META_JSON

    existing = [p for p in (embed_path, gamma_path, meta_path) if p.exists()]
    if existing and not args.force:
        names = ", ".join(str(p) for p in existing)
        raise SystemExit(f"FATAL outputs exist (use --force to overwrite): {names}")

    z = np.load(args.input, allow_pickle=False)
    embed = np.asarray(z["embed_weight"], dtype=np.float16)
    gamma = np.asarray(z["final_norm_gamma"], dtype=np.float16)
    eps = float(z["rms_norm_eps"])
    softcap = float(z["softcap"])
    tie = bool(z["tie_word_embeddings"])

    if embed.ndim != 2:
        raise SystemExit(f"FATAL embed_weight must be 2D, got {embed.shape}")
    if gamma.ndim != 1:
        raise SystemExit(f"FATAL final_norm_gamma must be 1D, got {gamma.shape}")
    if embed.shape[1] != gamma.shape[0]:
        raise SystemExit(
            f"FATAL embed/gamma mismatch: embed {embed.shape}, gamma {gamma.shape}")
    if not np.isfinite(embed).all():
        raise SystemExit("FATAL non-finite embed_weight")
    if not np.isfinite(gamma).all():
        raise SystemExit("FATAL non-finite final_norm_gamma")

    vocab_size, d_model = map(int, embed.shape)
    _atomic_write_bytes(embed_path, embed.tobytes(order="C"))
    _atomic_write_bytes(gamma_path, gamma.tobytes(order="C"))

    meta = {
        "artifacts_version": 1,
        "embed_bin": embed_path.name,
        "final_norm_gamma_bin": gamma_path.name,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "rms_norm_eps": eps,
        "softcap": softcap,
        "tie_word_embeddings": tie,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "max_ctx": 1024,
        "sliding_rope_theta": 10000.0,
        "global_rope_theta": 1000000.0,
        "sliding_d_head": 256,
        "global_rot_dim": 512,
        "tokenizer_json": TOKENIZER_JSON,
        "shards": SHARDS,
    }
    _atomic_write_json(meta_path, meta)

    print(f"wrote {embed_path}  ({embed_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {gamma_path}  ({gamma_path.stat().st_size / 1e6:.3f} MB)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()