"""Regenerate router weight bins as Float32 (from original safetensors).

The original per-expert build saved router proj and per_expert_scale as Float16.
Float16 truncation of router weights can change expert selection (catastrophic
for quality). This script re-extracts from safetensors and saves as Float32.

Run:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/regen_router_fp32.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

D_MODEL = 2816
N_EXPERTS_ALL = 128
MODEL_DIR = Path("models/gemma-4-26b-a4b")
REAP_MASK = Path("python/moe/out/gemma_reap_mask.npz")
OUT_BASE = Path("python/moe/out/per_expert")


def _load_index():
    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _read_tensor(idx, key):
    import torch
    from safetensors import safe_open
    fname = idx[key]
    with safe_open(MODEL_DIR / fname, framework="pt") as f:
        return f.get_tensor(key).to(torch.float32).contiguous().numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-experts", type=int, default=N_EXPERTS_ALL)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=30)
    args = ap.parse_args()

    idx = _load_index()
    use_reap = (args.n_experts < N_EXPERTS_ALL)
    if use_reap:
        mask = np.load(REAP_MASK)

    t0 = time.time()
    for li in range(args.layer_start, args.layer_end):
        base = f"model.language_model.layers.{li}"
        router_proj = _read_tensor(idx, f"{base}.router.proj.weight")   # (128, 2816)
        router_scale = _read_tensor(idx, f"{base}.router.scale")        # (2816,)
        router_per_exp = _read_tensor(idx, f"{base}.router.per_expert_scale")  # (128,)

        if use_reap:
            keep = mask["keep_idx"][li].astype(np.int64)
        else:
            keep = np.arange(N_EXPERTS_ALL, dtype=np.int64)

        router_proj_k = router_proj[keep]          # (n_experts, 2816)
        router_per_exp_k = router_per_exp[keep]    # (n_experts,)

        # Fuse router scale into proj (same as build_gemma_per_expert.py)
        scalar_root = D_MODEL ** -0.5
        r_scale = (router_scale * scalar_root).astype(np.float32)
        router_proj_fused = router_proj_k * r_scale[np.newaxis, :]  # (n_experts, 2816)

        # Save as Float32
        router_dir = OUT_BASE / f"L{li}" / "router"
        router_dir.mkdir(parents=True, exist_ok=True)
        router_proj_fused.astype(np.float32).tofile(router_dir / "proj_fp32.bin")
        router_per_exp_k.astype(np.float32).tofile(router_dir / "per_expert_scale_fp32.bin")
        print(f"  L{li}: proj {router_proj_fused.shape} → proj_fp32.bin, "
              f"scale {router_per_exp_k.shape} → per_expert_scale_fp32.bin")

    elapsed = time.time() - t0
    print(f"\nDone: {args.layer_end - args.layer_start} layers in {elapsed:.1f}s")

    # Update meta JSON to use fp32 router bins
    meta_path = Path("python/moe/out/gemma_swift_head_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        updated = 0
        for entry in meta.get("per_expert_layers", []):
            li = entry["layer"]
            if args.layer_start <= li < args.layer_end:
                old_proj = entry["router_proj_bin"]
                old_scale = entry["router_per_expert_scale_bin"]
                entry["router_proj_bin"] = old_proj.replace("proj_fp16.bin", "proj_fp32.bin")
                entry["router_per_expert_scale_bin"] = old_scale.replace(
                    "per_expert_scale_fp16.bin", "per_expert_scale_fp32.bin")
                updated += 1
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Updated {updated} entries in {meta_path}")


if __name__ == "__main__":
    main()
