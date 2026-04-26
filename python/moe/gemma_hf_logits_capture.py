"""gemma_hf_logits_capture.py — capture HF gemma-4-26b-a4b reference logits.

Run ONCE in .venv313 (HF/transformers) in foreground. Writes:
  python/moe/out/gemma_hf_golden_logits__<promptHash>__<shaShort>.npz
  python/moe/out/gemma_hf_golden_logits.npz   (symlink to latest)
  python/moe/out/.gemma_hf_golden_PASS         (sentinel)

Captures all-position logits per gatekeeper C3.
Pins fp16 + CPU per gatekeeper C2 (avoid MPS numerics drift).

Usage:
  .venv313/bin/python python/moe/gemma_hf_logits_capture.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from hf_full_model_safety import (
    DEFAULT_DISK_FREE_MIN_GIB,
    DEFAULT_MAX_CPU_MEMORY_GIB,
    DEFAULT_MAX_DISK_MEMORY_GIB,
    prepare_model_load_kwargs,
    require_disk_free,
    validate_full_model_load_policy,
)

PROMPT = "The capital of France is"
MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT_DIR = Path("python/moe/out")
SENTINEL = OUT_DIR / ".gemma_hf_golden_PASS"
LATEST_LINK = OUT_DIR / "gemma_hf_golden_logits.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offload", dest="offload", action="store_true", default=True)
    ap.add_argument("--no-offload", dest="offload", action="store_false")
    ap.add_argument("--offload-folder", type=Path,
                    default=OUT_DIR / ".offload_hf_logits_capture")
    ap.add_argument("--max-cpu-memory-gib", type=int, default=DEFAULT_MAX_CPU_MEMORY_GIB)
    ap.add_argument("--max-disk-memory-gib", type=int, default=DEFAULT_MAX_DISK_MEMORY_GIB)
    ap.add_argument("--disk-free-min-gib", type=int, default=DEFAULT_DISK_FREE_MIN_GIB)
    ap.add_argument("--allow-unsafe-cpu-memory", action="store_true")
    ap.add_argument("--allow-no-disk-offload", action="store_true")
    args = ap.parse_args()

    validate_full_model_load_policy(
        "gemma_hf_logits_capture",
        offload_enabled=args.offload,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        max_disk_memory_gib=args.max_disk_memory_gib,
        allow_unsafe_cpu_memory=args.allow_unsafe_cpu_memory,
        allow_no_disk_offload=args.allow_no_disk_offload,
    )

    if SENTINEL.exists():
        print(f"sentinel already exists: {SENTINEL}")
        print("Delete it to recapture.")
        sys.exit(0)

    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    print(f"=== HF gemma-4-26b-a4b logit capture ===")
    print(f"  prompt: {PROMPT!r}")
    print(f"  model_dir: {MODEL_DIR}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tfv
    print(f"  transformers={tfv}  torch={torch.__version__}")
    disk_paths = [MODEL_DIR, OUT_DIR]
    if args.offload:
        disk_paths.append(args.offload_folder)
    require_disk_free(disk_paths, args.disk_free_min_gib)

    print("  loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"]
    print(f"  tokens ({input_ids.shape[1]}): {input_ids[0].tolist()}")
    print(f"  decoded: {[tok.decode([i]) for i in input_ids[0].tolist()]}")

    print("  loading model fp16 with guarded offload policy...")
    print(
        f"  memory policy: offload={args.offload} cpu={args.max_cpu_memory_gib}GiB "
        f"disk={args.max_disk_memory_gib}GiB"
    )
    t0 = time.perf_counter()
    model_kwargs, actual_offload_folder = prepare_model_load_kwargs(
        torch_dtype=torch.float16,
        offload_enabled=args.offload,
        offload_folder=args.offload_folder,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        max_disk_memory_gib=args.max_disk_memory_gib,
        local_files_only=True,
    )
    if actual_offload_folder is not None:
        print(f"  offload folder: {actual_offload_folder}")
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), **model_kwargs).eval()
    print(f"  load wall: {time.perf_counter()-t0:.1f}s")

    print("  forward...")
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=False)
    fwd_t = time.perf_counter() - t0
    print(f"  forward wall: {fwd_t:.1f}s")

    logits = out.logits[0].float().cpu().numpy()  # (T, vocab)
    print(f"  logits shape: {logits.shape}  dtype={logits.dtype}")
    last_top1 = int(np.argmax(logits[-1]))
    print(f"  last-pos top-1: token_id={last_top1}  text={tok.decode([last_top1])!r}")

    # Hash prompt + model.safetensors.index.json for filename versioning.
    prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]
    idx_text = (MODEL_DIR / "model.safetensors.index.json").read_bytes()
    model_sha8 = hashlib.sha256(idx_text).hexdigest()[:8]
    out_path = OUT_DIR / f"gemma_hf_golden_logits__{prompt_hash}__{model_sha8}.npz"

    np.savez(
        str(out_path),
        logits=logits.astype(np.float32),  # save fp32 for downstream cos
        input_ids=input_ids[0].numpy(),
        prompt=np.array(PROMPT),
        tokenizer=np.array(str(MODEL_DIR)),
        model_sha8=np.array(model_sha8),
        prompt_sha8=np.array(prompt_hash),
        transformers_version=np.array(tfv),
        torch_version=np.array(torch.__version__),
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  saved -> {out_path}  ({sz_mb:.1f} MB)")

    if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
        LATEST_LINK.unlink()
    LATEST_LINK.symlink_to(out_path.name)
    print(f"  symlink: {LATEST_LINK} -> {out_path.name}")

    SENTINEL.write_text(
        f"prompt={PROMPT!r} prompt_sha8={prompt_hash} "
        f"model_sha8={model_sha8} last_top1={last_top1} "
        f"transformers={tfv} torch={torch.__version__}\n"
    )
    print(f"  sentinel: {SENTINEL}")
    print("\n# T4.1.4 step 2: HF golden capture PASS")


if __name__ == "__main__":
    main()
