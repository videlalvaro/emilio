#!/usr/bin/env python3
"""Factory-level Qwen2.5 → ANE CoreML conversion pipeline.

Automates the full GGUF → multi-shard ANE model pipeline:
  1. Read GGUF metadata to determine architecture
  2. Calculate optimal shard boundaries (respecting 96 MB ANE cliff)
  3. Convert each shard via gguf_to_ane.py --stateful --layer-start/--layer-end
  4. Export shared artifacts (embeddings, tokenizer, final norm, RoPE)
  5. Compile each shard with xcrun coremlcompiler
  6. Generate master meta JSON for the Swift runtime

Usage:
  python3 convert_qwen.py <model.gguf> [--seq-len 2048] [--quant-bits 4]
                                        [--output-dir ./out] [--max-shard-mb 80]

Requires: .venv (torch) for step 3, Xcode python3 for coremltools.
The script detects which Python has coremltools and uses it automatically.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Import the GGUF parser from our converter
sys.path.insert(0, str(Path(__file__).parent))
from gguf_to_ane import GGUFModel


# ─── ANE Constraints ────────────────────────────────────────────────────────

MAX_COMPILED_SHARD_MB = 80  # Conservative limit (cliff is ~96 MB)


def estimate_layer_size_int4(cfg):
    """Estimate per-layer compiled size in MB at INT4 quantization.

    Components per layer:
      - qkv_conv: (d + 2*kv_dim) × d × 0.5 bytes (INT4)
      - out_conv: d × d × 0.5 bytes
      - gate_up_conv: 2*d_ff × d × 0.5 bytes
      - down_conv: d × d_ff × 0.5 bytes
      - norms: 2 × d × 2 bytes (fp16, negligible)
      - biases: qkv bias = (d + 2*kv_dim) × 2 bytes (negligible)
      - per-group scales: ~6% overhead at group_size=32
    """
    d = cfg["d_model"]
    d_ff = cfg["d_ff"]
    n_kv = cfg["n_kv_heads"]
    d_head = cfg["d_head"]
    kv_dim = n_kv * d_head
    qkv_dim = d + 2 * kv_dim

    # Weight elements per layer
    qkv = qkv_dim * d
    out = d * d
    gate_up = 2 * d_ff * d
    down = d * d_ff
    total_elements = qkv + out + gate_up + down

    # INT4: 0.5 bytes per element + ~6% group scale overhead
    bytes_int4 = total_elements * 0.5 * 1.06

    # KV state overhead per layer (in compiled model): nkv × seq × dh × 2 (fp16)
    # This is relatively small compared to weights for typical seq_len
    # but we account for it as it's part of the compiled package
    # (Conservative: assume seq_len=2048 for estimation)
    kv_state = 2 * n_kv * 2048 * d_head * 2  # 2 caches × nkv × seq × dh × fp16

    return (bytes_int4 + kv_state) / (1024 * 1024)


def compute_shard_boundaries(n_layers, per_layer_mb, max_shard_mb):
    """Compute optimal shard boundaries for the given constraints.

    Returns a list of (start, end) tuples covering [0, n_layers).
    Tries to maximize layers per shard while staying under max_shard_mb.
    """
    layers_per_shard = max(1, int(max_shard_mb / per_layer_mb))
    # Clamp: at least 1, at most all layers
    layers_per_shard = min(layers_per_shard, n_layers)

    boundaries = []
    start = 0
    while start < n_layers:
        end = min(start + layers_per_shard, n_layers)
        boundaries.append((start, end))
        start = end

    return boundaries


def export_shared_artifacts(gguf_path, output_dir, cfg, max_seq_len):
    """Export artifacts shared across all shards: embeddings, tokenizer,
    final norm weights, and RoPE tables.

    These live on the host (Swift) side — not in any CoreML shard.
    """
    gguf = GGUFModel(gguf_path)
    d = cfg["d_model"]
    d_head = cfg["d_head"]

    # ── Embeddings (fp16) ──
    print("Exporting embeddings (fp16)...")
    token_embd = gguf.get_tensor("token_embd.weight", dtype=np.float16)
    embd_path = output_dir / "embed.bin"
    token_embd.tofile(str(embd_path))
    print(f"  {embd_path} ({embd_path.stat().st_size / 1e6:.1f} MB)")

    # ── Final norm weights (fp16) ──
    print("Exporting final norm weights...")
    final_norm = gguf.get_tensor("output_norm.weight", dtype=np.float16)
    norm_path = output_dir / "final_norm.bin"
    final_norm.tofile(str(norm_path))
    print(f"  {norm_path} ({norm_path.stat().st_size / 1e3:.1f} KB)")

    # ── LM head weights (fp16) — for host-side projection ──
    # For tied embeddings, this is the same as token_embd
    print("Exporting LM head weights...")
    if "output.weight" in gguf.tensors:
        lm_head = gguf.get_tensor("output.weight", dtype=np.float16)
        lm_head_path = output_dir / "lm_head.bin"
        lm_head.tofile(str(lm_head_path))
        print(f"  {lm_head_path} ({lm_head_path.stat().st_size / 1e6:.1f} MB)")
        tie_embeddings = False
    else:
        # Tied: LM head = embedding matrix (no separate file needed)
        print("  Tied to embeddings (no separate file)")
        lm_head_path = embd_path
        tie_embeddings = True

    # ── BPE Tokenizer ──
    print("Exporting BPE tokenizer...")
    tok_data = gguf.extract_tokenizer()
    tok_path = output_dir / "tokenizer.json"
    with open(tok_path, "w") as f:
        json.dump(tok_data, f)
    print(f"  {tok_path} ({len(tok_data['tokens'])} tokens, {len(tok_data['merges'])} merges)")

    # ── RoPE tables ──
    d_half = d_head // 2
    freqs = 1.0 / (cfg["rope_freq_base"] ** (
        np.arange(0, d_half, dtype=np.float32) / d_half))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    rope_cos = np.cos(angles).astype(np.float32)
    rope_sin = np.sin(angles).astype(np.float32)

    return {
        "embed_bin": "embed.bin",
        "final_norm_bin": "final_norm.bin",
        "lm_head_bin": str(lm_head_path.relative_to(output_dir)) if not tie_embeddings else "embed.bin",
        "tokenizer_json": "tokenizer.json",
        "tie_word_embeddings": tie_embeddings,
        "rope_cos": rope_cos.tolist(),
        "rope_sin": rope_sin.tolist(),
    }


def generate_master_meta(cfg, shards, shared_artifacts, max_seq_len,
                         quant_bits, output_dir):
    """Generate the master meta JSON that the Swift runtime reads.

    Follows the proven Gemma multi-shard pattern:
      - shards array with (start, end, path)
      - model config for runtime
      - paths to shared artifacts (embed, norm, tokenizer)
    """
    meta = {
        "artifacts_version": 1,
        "model_family": "qwen2.5",
        **cfg,
        "max_seq_len": max_seq_len,
        "quant_bits": quant_bits,
        **shared_artifacts,
        "shards": [
            {"start": s, "end": e, "path": path}
            for s, e, path in shards
        ],
    }
    # Remove large arrays from cfg that are in shared_artifacts
    meta.pop("rms_norm_eps", None)
    meta["rms_norm_eps"] = cfg["rms_norm_eps"]

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✓ Master meta: {meta_path}")
    return meta_path


def find_coremltools_python():
    """Find the Python that has coremltools installed.

    Priority: /usr/bin/python3 (Xcode, has coremltools 9) > .venv313 > .venv
    """
    candidates = [
        "/usr/bin/python3",
        str(Path(__file__).parent.parent.parent / ".venv313" / "bin" / "python3"),
        str(Path(__file__).parent.parent.parent / ".venv" / "bin" / "python3"),
    ]
    for py in candidates:
        if not Path(py).exists():
            continue
        try:
            result = subprocess.run(
                [py, "-c", "import coremltools; print(coremltools.__version__)"],
                capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                ver = result.stdout.strip()
                print(f"  Using {py} (coremltools {ver})")
                return py
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    raise RuntimeError("No Python with coremltools found. Need Xcode python3 or .venv313.")


def convert_shard(gguf_path, layer_start, layer_end, output_dir, seq_len,
                  quant_bits, group_size, strategy, python_path):
    """Convert a single shard by invoking gguf_to_ane.py."""
    converter = Path(__file__).parent / "gguf_to_ane.py"
    n_layers_total = layer_end  # Not strictly needed — GGUF has it

    cmd = [
        python_path, str(converter), str(gguf_path),
        "--stateful",
        "--layer-start", str(layer_start),
        "--layer-end", str(layer_end),
        "--seq-len", str(seq_len),
        "--quant-bits", str(quant_bits),
        "--group-size", str(group_size),
        "--quant-strategy", strategy,
    ]

    print(f"\n{'─'*60}")
    print(f"Converting shard [{layer_start}, {layer_end})...")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'─'*60}")

    # Run in the output directory so artifacts land there
    result = subprocess.run(cmd, cwd=str(output_dir))
    if result.returncode != 0:
        raise RuntimeError(f"Shard [{layer_start},{layer_end}) conversion failed (exit {result.returncode})")

    # Find the generated .mlpackage
    # Naming convention: QwenANE_{total}L_s{start}-{end}[_q4].mlpackage
    # But we need to discover the actual name since it includes total layers from GGUF
    candidates = list(output_dir.glob(f"*_s{layer_start}-{layer_end}*.mlpackage"))
    if not candidates:
        raise RuntimeError(f"No .mlpackage found for shard [{layer_start},{layer_end})")
    pkg_path = candidates[0]
    print(f"  ✓ {pkg_path.name}")
    return pkg_path


def compile_shard(pkg_path, output_dir):
    """Compile .mlpackage → .mlmodelc using xcrun coremlcompiler."""
    print(f"  Compiling {pkg_path.name}...")
    cmd = ["xcrun", "coremlcompiler", "compile", str(pkg_path), str(output_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Compilation failed for {pkg_path.name}")

    # Compiled output: same name but .mlmodelc
    compiled = output_dir / pkg_path.name.replace(".mlpackage", ".mlmodelc")
    if not compiled.exists():
        # Sometimes coremlcompiler creates with slightly different name
        candidates = list(output_dir.glob(pkg_path.stem + "*.mlmodelc"))
        if candidates:
            compiled = candidates[0]
        else:
            raise RuntimeError(f"Compiled model not found for {pkg_path.name}")
    print(f"  ✓ {compiled.name}")
    return compiled


def main():
    parser = argparse.ArgumentParser(
        description="Factory-level Qwen2.5 → ANE CoreML conversion pipeline")
    parser.add_argument("gguf", help="Path to GGUF model file")
    parser.add_argument("--seq-len", type=int, default=2048, dest="seq_len",
                        help="Max sequence length (default: 2048)")
    parser.add_argument("--quant-bits", type=int, choices=[0, 4, 8], default=4,
                        dest="quant_bits",
                        help="Weight quantization (default: 4 = INT4)")
    parser.add_argument("--group-size", type=int, default=32, dest="group_size",
                        help="Quantization group size (default: 32)")
    parser.add_argument("--quant-strategy", default="uniform", dest="quant_strategy",
                        choices=["uniform", "mixed", "gptq", "gptq-real"],
                        help="Quantization strategy (default: uniform)")
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir",
                        help="Output directory (default: auto from model name)")
    parser.add_argument("--max-shard-mb", type=float, default=MAX_COMPILED_SHARD_MB,
                        dest="max_shard_mb",
                        help=f"Max compiled shard size in MB (default: {MAX_COMPILED_SHARD_MB})")
    parser.add_argument("--skip-compile", action="store_true", dest="skip_compile",
                        help="Skip xcrun compilation (produce .mlpackage only)")
    parser.add_argument("--layers", type=int, default=None,
                        help="Override number of layers (default: all from GGUF)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print shard plan without converting")
    args = parser.parse_args()

    gguf_path = Path(args.gguf).resolve()
    if not gguf_path.exists():
        print(f"ERROR: {gguf_path} not found")
        sys.exit(1)

    # ── Read model config from GGUF ──
    print(f"Reading GGUF metadata: {gguf_path.name}")
    gguf = GGUFModel(str(gguf_path))
    cfg = gguf.config()

    n_layers = args.layers or cfg["n_layers"]
    d = cfg["d_model"]
    d_ff = cfg["d_ff"]
    nh = cfg["n_heads"]
    nkv = cfg["n_kv_heads"]
    dh = cfg["d_head"]
    vocab = cfg["vocab_size"]

    print(f"\n{'='*60}")
    print(f"  Model:   Qwen2.5 ({n_layers}L, d={d}, d_ff={d_ff})")
    print(f"  Heads:   {nh} attn, {nkv} KV, d_head={dh}")
    print(f"  Vocab:   {vocab}")
    print(f"  SeqLen:  {args.seq_len}")
    print(f"  Quant:   INT{args.quant_bits} (group_size={args.group_size})")
    print(f"{'='*60}")

    # ── Calculate shard boundaries ──
    per_layer_mb = estimate_layer_size_int4(cfg)
    boundaries = compute_shard_boundaries(n_layers, per_layer_mb, args.max_shard_mb)

    print(f"\n  Per-layer estimated size: {per_layer_mb:.1f} MB (INT4)")
    print(f"  Shard plan ({len(boundaries)} shards, max {args.max_shard_mb} MB each):")
    for i, (s, e) in enumerate(boundaries):
        est_mb = (e - s) * per_layer_mb
        print(f"    shard {i}: layers [{s}, {e}) = {e-s} layers ≈ {est_mb:.0f} MB")

    if len(boundaries) == 1:
        print(f"\n  ✓ Model fits in a single shard — no multi-shard needed")

    if args.dry_run:
        print("\n  [dry-run] Exiting without conversion.")
        return

    # ── Setup output directory ──
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        stem = gguf_path.stem.replace("-instruct", "").replace("_", "-")
        output_dir = Path(__file__).parent / f"qwen_{stem}_ane"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output: {output_dir}")

    # ── Find Python with coremltools ──
    print("\nLocating coremltools Python...")
    python_path = find_coremltools_python()

    # ── Export shared artifacts ──
    print("\n" + "="*60)
    print("Exporting shared artifacts...")
    print("="*60)
    shared = export_shared_artifacts(gguf_path, output_dir, cfg, args.seq_len)

    # ── Convert each shard ──
    shard_info = []
    for i, (layer_start, layer_end) in enumerate(boundaries):
        pkg_path = convert_shard(
            gguf_path, layer_start, layer_end, output_dir,
            args.seq_len, args.quant_bits, args.group_size,
            args.quant_strategy, python_path)

        if not args.skip_compile:
            compiled = compile_shard(pkg_path, output_dir)
            shard_path = compiled.name
        else:
            shard_path = pkg_path.name

        shard_info.append((layer_start, layer_end, shard_path))

    # ── Generate master meta ──
    print("\n" + "="*60)
    print("Generating master meta JSON...")
    print("="*60)
    meta_path = generate_master_meta(
        cfg, shard_info, shared, args.seq_len, args.quant_bits, output_dir)

    # ── Summary ──
    total_shards = len(shard_info)
    print(f"\n{'='*60}")
    print(f"  ✓ CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:      Qwen2.5 ({n_layers}L, d={d})")
    print(f"  Shards:     {total_shards}")
    print(f"  Quant:      INT{args.quant_bits}")
    print(f"  SeqLen:     {args.seq_len}")
    print(f"  Output:     {output_dir}")
    print(f"  Meta:       {meta_path}")
    print(f"\n  Swift runtime usage:")
    print(f"    ./qwen_ane --meta {meta_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
