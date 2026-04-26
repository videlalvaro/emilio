#!/usr/bin/env python3
"""Build 4 ANE LM head shards for Gemma-4-26B-A4B.

Reads python/moe/out/gemma_logit_head.npz (from build_logit_head.py) and splits
the vocab projection into N shards, each containing:
  - RMSNorm (duplicated, tiny — avoids host-side norm entirely)
  - Conv2d(d_model → vocab_chunk, 1) with INT8 quantization

Each shard takes raw hidden state (1, d_model, 1, 1) and outputs logits for its
vocab slice.  Host concatenates the slices + applies tanh·softcap.

Shard math (INT8):
  262144 × 2816 × 1 byte = 703 MB total
  4 shards of 65536 × 2816 = 176 MB each → well under 250 MB ANE limit

Must run with Xcode python3 (has coremltools 9):
  /usr/bin/python3 python/moe/build_gemma_lm_head_shards.py

Probe mode (1 shard, no compile — fast check):
  /usr/bin/python3 python/moe/build_gemma_lm_head_shards.py --probe
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

NPZ_PATH = Path("python/moe/out/gemma_logit_head.npz")
OUT_DIR = Path("python/moe/out/lm_head_shards")


def build_shard_model(norm_weight_f32, embed_slice_f32, d_model, vocab_start,
                      vocab_end, rms_eps, shard_idx, num_shards):
    """Build a single LM head shard: RMSNorm + Conv2d projection.

    Returns (traced_model, example_input).
    """
    import torch
    import torch.nn as nn

    vocab_chunk = vocab_end - vocab_start

    class RMSNormConv(nn.Module):
        """ANE-friendly RMSNorm: (1, D, 1, 1) → (1, D, 1, 1)."""
        def __init__(self, weight, eps):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(
                torch.tensor(weight, dtype=torch.float16).reshape(-1, 1, 1),
                requires_grad=False)

        def forward(self, x):
            # Safe-norm peephole (Dragon Book §8.7): scale before squaring
            K = x.shape[1] ** 0.5
            x_scaled = x * (1.0 / K)
            variance = x_scaled.pow(2).mean(dim=1, keepdim=True)
            x_normed = x_scaled * torch.rsqrt(variance + self.eps / (K * K))
            return (x_normed * self.weight).to(x.dtype)

    class LMHeadShard(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNormConv(norm_weight_f32, rms_eps)
            self.proj = nn.Conv2d(d_model, vocab_chunk, 1, bias=False)
            self.proj.weight = nn.Parameter(
                torch.tensor(embed_slice_f32, dtype=torch.float16).reshape(
                    vocab_chunk, d_model, 1, 1),
                requires_grad=False)

        def forward(self, x):
            """x: (1, d_model, 1, 1) fp16 → logits: (1, vocab_chunk, 1, 1) fp16"""
            normed = self.norm(x)
            return self.proj(normed)

    model = LMHeadShard()
    model.half()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Shard {shard_idx}/{num_shards}: vocab [{vocab_start},{vocab_end}) "
          f"= {vocab_chunk} tokens, {n_params:,} params")

    example_input = torch.randn(1, d_model, 1, 1, dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    return traced, example_input


def convert_and_quantize(traced, example_input, d_model, shard_name, quant_bits=8):
    """Convert traced model to CoreML + INT8 quantize."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    ct_inputs = [ct.TensorType(name="hidden", shape=(1, d_model, 1, 1),
                                dtype=np.float16)]
    ct_outputs = [ct.TensorType(name="logits", dtype=np.float16)]

    print(f"  Converting {shard_name} to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if quant_bits == 8:
        print(f"  Quantizing to INT8...")
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        opt_config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=opt_config)
    elif quant_bits == 0:
        print(f"  Skipping quantization (fp16)")
    else:
        raise ValueError(f"Only INT8 or fp16 (0) supported, got {quant_bits}")

    return mlmodel


def compile_mlpackage(pkg_path: str) -> str | None:
    """Compile .mlpackage → .mlmodelc."""
    abs_path = str(Path(pkg_path).resolve())
    out_dir = str(Path(pkg_path).parent.resolve())
    print(f"  Compiling {Path(pkg_path).name}...")
    t0 = time.time()
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", abs_path, out_dir],
        capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAIL compile: {result.stderr}", file=sys.stderr)
        return None
    stem = Path(pkg_path).stem
    mlmodelc = Path(out_dir) / f"{stem}.mlmodelc"
    if mlmodelc.exists():
        size_mb = sum(f.stat().st_size for f in mlmodelc.rglob('*')
                      if f.is_file()) / 1e6
        print(f"  OK {mlmodelc.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")
        return str(mlmodelc)
    print(f"  FAIL: expected {mlmodelc} not found", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build Gemma-4 ANE LM head shards from logit_head.npz")
    parser.add_argument("--npz", default=str(NPZ_PATH),
                        help=f"Input NPZ (default: {NPZ_PATH})")
    parser.add_argument("--output-dir", default=str(OUT_DIR),
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--num-shards", type=int, default=4,
                        help="Number of vocab shards (default: 4)")
    parser.add_argument("--quant-bits", type=int, default=8, choices=[0, 8],
                        help="Quantization: 0=fp16, 8=INT8 (default: 8)")
    parser.add_argument("--probe", action="store_true",
                        help="Build 1 shard only, no compile — fast check")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip coremlcompiler step")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"FATAL: {npz_path} not found. Run build_logit_head.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {npz_path}...")
    data = np.load(str(npz_path), allow_pickle=False)
    embed_weight = data["embed_weight"]    # (vocab, d_model) fp16
    norm_gamma = data["final_norm_gamma"]  # (d_model,) fp16
    rms_eps = float(data["rms_norm_eps"])
    softcap = float(data["softcap"])
    tie = bool(data["tie_word_embeddings"])

    vocab, d_model = embed_weight.shape
    print(f"  vocab={vocab}  d_model={d_model}  eps={rms_eps}  softcap={softcap}  tied={tie}")

    if not tie:
        print("FATAL: tie_word_embeddings=False — this script uses embed as LM head.",
              file=sys.stderr)
        sys.exit(2)

    # Convert to float32 for torch tracing
    embed_f32 = embed_weight.astype(np.float32)
    norm_f32 = norm_gamma.astype(np.float32)

    num_shards = args.num_shards
    chunk_size = math.ceil(vocab / num_shards)
    shard_ranges = []
    for i in range(num_shards):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, vocab)
        shard_ranges.append((start, end))

    # Size estimates
    for i, (s, e) in enumerate(shard_ranges):
        chunk = e - s
        int8_mb = chunk * d_model / 1e6  # 1 byte per weight
        print(f"  Shard {i}: vocab [{s},{e}) = {chunk} tokens, ~{int8_mb:.0f} MB INT8")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_to_build = 1 if args.probe else num_shards
    if args.probe:
        print(f"\n── PROBE MODE: building shard 0 only ──")

    import torch  # Late import so arg parsing + NPZ load is fast

    built = []
    total_t0 = time.time()
    for i in range(num_to_build):
        start, end = shard_ranges[i]
        weight_slice = embed_f32[start:end]

        q_tag = "q8" if args.quant_bits == 8 else "fp16"
        shard_name = f"GemmaLMHead_s{i}_{q_tag}"
        print(f"\n{'='*60}")
        print(f"Building {shard_name}  (vocab [{start},{end}))...")

        traced, example_input = build_shard_model(
            norm_f32, weight_slice, d_model, start, end,
            rms_eps, i, num_shards)

        mlmodel = convert_and_quantize(
            traced, example_input, d_model, shard_name,
            quant_bits=args.quant_bits)

        pkg_path = out_dir / f"{shard_name}.mlpackage"
        mlmodel.save(str(pkg_path))
        pkg_mb = sum(f.stat().st_size for f in pkg_path.rglob('*')
                     if f.is_file()) / 1e6
        print(f"  Saved {pkg_path.name} ({pkg_mb:.1f} MB)")

        mlmodelc_path = None
        if not args.no_compile and not args.probe:
            mlmodelc_path = compile_mlpackage(str(pkg_path))

        built.append({
            "shard_idx": i,
            "vocab_start": start,
            "vocab_end": end,
            "mlpackage": str(pkg_path),
            "mlmodelc": mlmodelc_path,
            "pkg_size_mb": pkg_mb,
        })

    total_elapsed = time.time() - total_t0

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({total_elapsed:.0f}s total)")
    for s in built:
        c = f"  compiled={Path(s['mlmodelc']).name}" if s['mlmodelc'] else ""
        print(f"  shard {s['shard_idx']}: vocab [{s['vocab_start']},{s['vocab_end']}) "
              f"pkg={s['pkg_size_mb']:.1f}MB{c}")

    if args.probe:
        s = built[0]
        est = s['pkg_size_mb'] * 0.9
        print(f"\n  Probe estimate: ~{est:.0f} MB compiled")
        if est > 250:
            print(f"  WARNING: over 250 MB ANE limit — increase --num-shards")
        else:
            print(f"  OK: under 250 MB ANE limit")

    # Write a manifest for the Swift driver to read
    if not args.probe:
        manifest = {
            "num_shards": num_shards,
            "vocab_size": vocab,
            "d_model": d_model,
            "rms_norm_eps": rms_eps,
            "softcap": softcap,
            "quant": "int8" if args.quant_bits == 8 else "fp16",
            "shards": [{
                "shard_idx": s["shard_idx"],
                "vocab_start": s["vocab_start"],
                "vocab_end": s["vocab_end"],
                "mlmodelc": str(Path(s["mlmodelc"]).name) if s["mlmodelc"] else None,
                "mlpackage": str(Path(s["mlpackage"]).name),
            } for s in built],
        }
        manifest_path = out_dir / "lm_head_manifest.json"
        import json
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
