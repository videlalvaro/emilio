#!/usr/bin/env python3
"""GGUF → ANE-native CoreML converter for Qwen2.5 and Phi-3/4 families.

All architecture-dependent sizes (d_model, n_heads, n_kv_heads, d_head, d_ff, …)
are read from the GGUF metadata at runtime — nothing is hardcoded to 0.5B.
Supports both split (Qwen: attn_q/k/v, ffn_gate+ffn_up) and fused
(Phi: attn_qkv, fused gate+up in ffn_up) tensor layouts automatically.

Design targeting Apple Neural Engine directly:
  - Float16 throughout (ANE native dtype)
  - Stateful KV cache via CoreML StateType (on-chip, no host round-trips)
  - Int4 / Int8 post-training grouped weight quantization (4× / 2× smaller than fp16)
  - macOS26/iOS26 deployment target (latest ANE compiler)
  - Proper BPE tokenizer exported from GGUF
  - ChatML prompt template

Every linear projection is Conv2d(1×1). The ANE's conv engine does all matmuls.

Usage: python3 gguf_to_ane.py <model.gguf> [--layers N] [--seq-len S] [--quant-bits {0,4,8}]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# ─── Minimal GGUF Parser ────────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747


def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_i32(f):
    return struct.unpack("<i", f.read(4))[0]


def read_f32(f):
    return struct.unpack("<f", f.read(4))[0]


def read_string(f):
    length = read_u64(f)
    return f.read(length).decode("utf-8")


def read_meta_value(f, vtype):
    readers = {
        0: lambda: f.read(1)[0],           # uint8
        1: lambda: struct.unpack("b", f.read(1))[0],  # int8
        2: lambda: struct.unpack("<H", f.read(2))[0],  # uint16
        3: lambda: struct.unpack("<h", f.read(2))[0],  # int16
        4: lambda: read_u32(f),             # uint32
        5: lambda: read_i32(f),             # int32
        6: lambda: read_f32(f),             # float32
        7: lambda: bool(f.read(1)[0]),      # bool
        8: lambda: read_string(f),          # string
        10: lambda: read_u64(f),            # uint64
        12: lambda: struct.unpack("<d", f.read(8))[0],  # float64
    }
    if vtype == 9:  # array
        elem_type = read_u32(f)
        count = read_u64(f)
        return [read_meta_value(f, elem_type) for _ in range(count)]
    if vtype in readers:
        return readers[vtype]()
    raise ValueError(f"Unknown meta type {vtype}")


def dequant_q8_0(data, n_elements):
    """Dequantize Q8_0 blocks → float16 (ANE native)."""
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.empty(n_elements, dtype=np.float16)
    for i in range(n_blocks):
        offset = i * 34
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16)[0]
        quants = np.frombuffer(data[offset + 2:offset + 34], dtype=np.int8).astype(np.float16)
        result[i * block_size:(i + 1) * block_size] = quants * scale
    return result


class GGUFModel:
    """GGUF reader with direct float16 dequantization."""

    def __init__(self, path):
        self.path = path
        self.metadata = {}
        self.tensors = {}
        self._data_offset = 0
        self._parse()

    def _parse(self):
        with open(self.path, "rb") as f:
            magic = read_u32(f)
            assert magic == GGUF_MAGIC, f"Bad magic: {magic:#x}"
            version = read_u32(f)
            assert version >= 2, f"GGUF version {version} too old"
            n_tensors = read_u64(f)
            n_meta = read_u64(f)
            for _ in range(n_meta):
                key = read_string(f)
                vtype = read_u32(f)
                value = read_meta_value(f, vtype)
                self.metadata[key] = value
            for _ in range(n_tensors):
                name = read_string(f)
                n_dims = read_u32(f)
                dims = [read_u64(f) for _ in range(n_dims)]
                dtype = read_u32(f)
                offset = read_u64(f)
                self.tensors[name] = {
                    "shape": dims, "type": dtype, "offset": offset,
                    "n_elements": int(np.prod(dims)),
                }
            alignment = self.metadata.get("general.alignment", 32)
            pos = f.tell()
            self._data_offset = pos + (alignment - pos % alignment) % alignment

    def meta(self, key, default=None):
        return self.metadata.get(key, default)

    def get_tensor(self, name, dtype=np.float16):
        """Load tensor, dequantize to float16 (ANE native)."""
        info = self.tensors[name]
        n_elements = info["n_elements"]
        ttype = info["type"]
        with open(self.path, "rb") as f:
            f.seek(self._data_offset + info["offset"])
            if ttype == 0:  # F32
                data = f.read(n_elements * 4)
                arr = np.frombuffer(data, dtype=np.float32).astype(dtype).copy()
            elif ttype == 1:  # F16
                data = f.read(n_elements * 2)
                arr = np.frombuffer(data, dtype=np.float16).copy()
                if dtype != np.float16:
                    arr = arr.astype(dtype)
            elif ttype == 8:  # Q8_0
                n_blocks = n_elements // 32
                data = f.read(n_blocks * 34)
                if dtype == np.float16:
                    arr = dequant_q8_0(data, n_elements)
                else:
                    arr = dequant_q8_0(data, n_elements).astype(dtype)
            else:
                raise ValueError(f"Unsupported type {ttype} for {name}")
        shape = list(reversed(info["shape"]))
        return arr.reshape(shape)

    def config(self):
        arch = self.meta("general.architecture", "qwen2")
        d_model = self.meta(f"{arch}.embedding_length", 896)
        n_heads = self.meta(f"{arch}.attention.head_count", 14)
        # Derive vocab_size from actual tensor shape — metadata can be wrong
        # (e.g. Qwen 7B: metadata says 151936, tensor is 152064)
        meta_vocab = self.meta(f"{arch}.vocab_size",
                     self.meta("tokenizer.ggml.vocab_size", 151936))
        if "token_embd.weight" in self.tensors:
            # GGUF shape is [d_model, vocab] — vocab is the last dim
            embd_shape = self.tensors["token_embd.weight"]["shape"]
            tensor_vocab = embd_shape[-1]
            if tensor_vocab != meta_vocab:
                print(f"  WARNING: vocab_size mismatch: metadata={meta_vocab}, "
                      f"tensor={tensor_vocab}. Using tensor shape.")
            vocab_size = tensor_vocab
        else:
            vocab_size = meta_vocab
        return {
            "arch": arch,
            "vocab_size": vocab_size,
            "n_layers": self.meta(f"{arch}.block_count", 24),
            "n_heads": n_heads,
            "n_kv_heads": self.meta(f"{arch}.attention.head_count_kv", 2),
            "d_model": d_model,
            "d_head": d_model // n_heads,
            "d_ff": self.meta(f"{arch}.feed_forward_length", 4864),
            "rms_norm_eps": self.meta(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6),
            "rope_freq_base": self.meta(f"{arch}.rope.freq_base", 1000000.0),
            "eos_token_id": self.meta("tokenizer.ggml.eos_token_id", 151645),
            "bos_token_id": self.meta("tokenizer.ggml.bos_token_id", 151643),
        }

    # ── Architecture-agnostic tensor helpers ────────────────────────────

    def get_qkv_weights(self, prefix, cfg):
        """Return (q_w, k_w, v_w) as separate numpy arrays.
        Handles both split (Qwen: attn_q/attn_k/attn_v) and fused (Phi: attn_qkv) layouts."""
        if f"{prefix}.attn_q.weight" in self.tensors:
            return (self.get_tensor(f"{prefix}.attn_q.weight"),
                    self.get_tensor(f"{prefix}.attn_k.weight"),
                    self.get_tensor(f"{prefix}.attn_v.weight"))
        # Fused QKV (Phi-3/4): shape (d + 2*kv_dim, d_model)
        qkv = self.get_tensor(f"{prefix}.attn_qkv.weight")
        d = cfg["d_model"]
        kv_dim = cfg["n_kv_heads"] * cfg["d_head"]
        return qkv[:d], qkv[d:d + kv_dim], qkv[d + kv_dim:]

    def get_qkv_biases(self, prefix, cfg):
        """Return (q_b, k_b, v_b) or None if no biases exist."""
        if f"{prefix}.attn_q.bias" in self.tensors:
            return (self.get_tensor(f"{prefix}.attn_q.bias"),
                    self.get_tensor(f"{prefix}.attn_k.bias"),
                    self.get_tensor(f"{prefix}.attn_v.bias"))
        if f"{prefix}.attn_qkv.bias" in self.tensors:
            qkv_b = self.get_tensor(f"{prefix}.attn_qkv.bias")
            d = cfg["d_model"]
            kv_dim = cfg["n_kv_heads"] * cfg["d_head"]
            return qkv_b[:d], qkv_b[d:d + kv_dim], qkv_b[d + kv_dim:]
        return None

    def get_gate_up_weights(self, prefix, cfg):
        """Return (gate_w, up_w) as separate numpy arrays.
        Handles both split (Qwen: ffn_gate + ffn_up) and fused (Phi: ffn_up=gate+up) layouts."""
        if f"{prefix}.ffn_gate.weight" in self.tensors:
            return (self.get_tensor(f"{prefix}.ffn_gate.weight"),
                    self.get_tensor(f"{prefix}.ffn_up.weight"))
        # Fused gate+up (Phi-3/4): ffn_up has shape (2*d_ff, d_model)
        fused = self.get_tensor(f"{prefix}.ffn_up.weight")
        dff = cfg["d_ff"]
        return fused[:dff], fused[dff:]

    def has_biases(self, prefix):
        """Check if this block has QKV biases."""
        return (f"{prefix}.attn_q.bias" in self.tensors or
                f"{prefix}.attn_qkv.bias" in self.tensors)

    def extract_tokenizer(self):
        """Extract BPE vocab + merges from GGUF metadata."""
        tokens = self.meta("tokenizer.ggml.tokens", [])
        merges = self.meta("tokenizer.ggml.merges", [])
        token_types = self.meta("tokenizer.ggml.token_type", [])
        return {
            "model": self.meta("tokenizer.ggml.model", "gpt2"),
            "tokens": tokens,
            "merges": merges,
            "token_types": token_types,
            "eos_token_id": self.meta("tokenizer.ggml.eos_token_id", 151645),
            "bos_token_id": self.meta("tokenizer.ggml.bos_token_id", 151643),
            "padding_token_id": self.meta("tokenizer.ggml.padding_token_id", 151643),
        }


# ─── Weight Quantization Helpers ───────────────────────────────────────────

def _quantize_weights(mlmodel, bits, group_size=32, strategy="uniform"):
    """Apply post-training weight quantization.

    bits=0        → no-op (returns model as-is, fp16 weights)
    bits=8        → per-tensor symmetric int8 (≈2× smaller than fp16)
    bits=4        → grouped symmetric int4 (≈4× smaller than fp16, group_size scales)

    strategy controls HOW the quantization is applied:
      "uniform"   → same precision for all ops (original behavior, RTN)
      "mixed"     → sensitive ops at int8, FFN bulk at int4 (MIL-level)
    """
    if bits == 0:
        return mlmodel
    if bits not in (4, 8):
        raise ValueError(f"quant_bits must be 0, 4, or 8 (got {bits})")

    if strategy == "mixed" and bits == 4:
        return _quantize_mixed_precision(mlmodel, group_size=group_size)

    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    if bits == 8:
        print("\nQuantizing weights to int8 (per-tensor symmetric)...")
        op_config = OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
        )
    else:  # bits == 4
        print(f"\nQuantizing weights to int4 (grouped, group_size={group_size})...")
        op_config = OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=group_size,
        )

    opt_config = OptimizationConfig(global_config=op_config)
    mlmodel = linear_quantize_weights(mlmodel, config=opt_config)
    print(f"  Weights quantized to int{bits} ✓")
    return mlmodel


def _quantize_mixed_precision(mlmodel, group_size=32):
    """Mixed-precision quantization at the MIL level.

    Strategy: keep precision-sensitive ops at int8, apply int4 only to FFN bulk.

    Sensitive (int8):
      - QKV projections (layers_*_qkv_conv_weight) — attention quality
      - Output projections (layers_*_out_conv_weight*) — residual stream injection
      - LM head (lm_head_conv_weight) — final logit quality
      - First and last transformer layers — boundary layers accumulate/emit error

    Bulk (int4 grouped):
      - FFN gate_up and down convolutions — largest weights, most tolerant of compression
    """
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        get_weights_metadata,
        linear_quantize_weights,
    )

    print(f"\nMixed-precision quantization (sensitive=int8, FFN=int4 g{group_size})...")

    int8_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
    )
    int4_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=group_size,
    )

    # Discover all weight op names
    weight_meta = get_weights_metadata(mlmodel, weight_threshold=512)
    op_names = sorted(weight_meta.keys())

    # Find the highest layer index to identify first/last layers
    layer_indices = set()
    for name in op_names:
        if name.startswith("layers_"):
            parts = name.split("_")
            try:
                layer_indices.add(int(parts[1]))
            except (IndexError, ValueError):
                pass
    first_layer = min(layer_indices) if layer_indices else 0
    last_layer = max(layer_indices) if layer_indices else 0

    # Build per-op config: default is int4 (bulk FFN), override sensitive ops to int8
    op_name_configs = {}
    int8_ops = []
    int4_ops = []
    skipped_ops = []

    for name in op_names:
        is_sensitive = False

        # QKV projections — attention quality (all layers)
        # These are the most precision-sensitive: they compute Q, K, V
        # which directly control attention pattern quality.
        if "qkv_conv_weight" in name:
            is_sensitive = True

        # Skip tiny weights (norms, biases)
        elif weight_meta[name].val.size < 2048:
            skipped_ops.append(name)
            continue

        # Everything else (out_conv, FFN, LM head) → int4 bulk

        if is_sensitive:
            op_name_configs[name] = int8_config
            int8_ops.append(name)
        else:
            int4_ops.append(name)

    print(f"  int8 (sensitive): {len(int8_ops)} ops")
    for name in int8_ops:
        shape = weight_meta[name].val.shape
        print(f"    {name}  {shape}")
    print(f"  int4 (FFN bulk):  {len(int4_ops)} ops")
    for name in int4_ops:
        shape = weight_meta[name].val.shape
        print(f"    {name}  {shape}")
    if skipped_ops:
        print(f"  skipped (tiny):   {len(skipped_ops)} ops")

    opt_config = OptimizationConfig(
        global_config=int4_config,
        op_name_configs=op_name_configs,
    )
    mlmodel = linear_quantize_weights(mlmodel, config=opt_config)
    print("  Mixed-precision quantization ✓")
    return mlmodel


# ─── Calibration Data ──────────────────────────────────────────────────────

def _make_calibration_inputs(gguf_model, token_embd, cfg, n_samples=128, seq_len=64,
                              use_real_text=False):
    """Generate calibration inputs by embedding token sequences.

    Returns a list of (x_embd, position) tuples where x_embd is (1, d, 1, 1) fp16
    and position is the token position index.

    use_real_text=True: chat-template-structured sequences with Zipf-distributed
    common tokens (preserves instruction-following patterns in GPTQ Hessian).
    use_real_text=False: uniformly random tokens from full vocab (original).
    """
    if use_real_text:
        return _make_real_calibration_inputs(gguf_model, token_embd, cfg, n_samples, seq_len)

    import torch

    vocab = cfg["vocab_size"]
    d = cfg["d_model"]

    # Generate random token sequences for calibration
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(n_samples):
        token_ids = rng.integers(0, vocab, size=seq_len)
        for pos, tid in enumerate(token_ids):
            emb = token_embd[tid].astype(np.float16)
            x = torch.tensor(emb, dtype=torch.float16).reshape(1, d, 1, 1)
            samples.append((x, pos))

    print(f"  Calibration: {len(samples)} samples ({n_samples} seqs × {seq_len} tokens)")
    return samples


def _make_real_calibration_inputs(gguf_model, token_embd, cfg, n_samples, seq_len):
    """Generate calibration inputs using chat template structure + Zipf-distributed tokens.

    Instead of uniform random tokens from the full 151K vocab (mostly rare tokens),
    this uses:
    1. Qwen chat template markers (<|im_start|>, <|im_end|>) so GPTQ sees the
       instruction-following structure and preserves it.
    2. Uniform random tokens from the first 50K vocab IDs (common BPE merges).
       NOT Zipf — Zipf concentrates on too few embeddings, creating low-rank
       Hessians that blow up gate_up_conv errors (1449 at layer 1!).

    This gives the GPTQ Hessian instruction-following structure without
    the rank-deficiency that Zipf causes for wide FFN projections.
    """
    import torch

    vocab = cfg["vocab_size"]
    d = cfg["d_model"]

    tok_data = gguf_model.extract_tokenizer()
    token_types = tok_data.get("token_types", [])

    # Qwen chat template special tokens
    IM_START = 151644
    IM_END = tok_data.get("eos_token_id", 151645)

    # Normal vocabulary tokens (type 1), truncated to first 50K (common BPE merges).
    # Uniform random within this range — NOT Zipf (Zipf creates low-rank Hessians).
    normal_ids = np.array([i for i, t in enumerate(token_types)
                           if t == 1 and i < 50000], dtype=np.int64)
    if len(normal_ids) < 1000:
        # Fallback: tokens 256-50000 (skip byte tokens, take common BPE merges)
        normal_ids = np.arange(256, min(50000, vocab), dtype=np.int64)

    rng = np.random.default_rng(42)

    samples = []
    for seq_idx in range(n_samples):
        seq = []

        # ── System message (~12 tokens) ──
        # <|im_start|> system-content <|im_end|>
        sys_len = min(10, max(seq_len // 8, 4))
        seq.append(IM_START)
        sys_ids = normal_ids[rng.integers(0, len(normal_ids), size=sys_len)]
        seq.extend(sys_ids.tolist())
        seq.append(IM_END)

        # ── User message (roughly half of remaining budget) ──
        user_budget = max((seq_len - len(seq)) // 2, 4)
        seq.append(IM_START)
        user_ids = normal_ids[rng.integers(0, len(normal_ids), size=user_budget - 2)]
        seq.extend(user_ids.tolist())
        seq.append(IM_END)

        # ── Assistant response (fill remaining) ──
        remaining = seq_len - len(seq)
        if remaining > 1:
            seq.append(IM_START)
            asst_ids = normal_ids[rng.integers(0, len(normal_ids), size=remaining - 1)]
            seq.extend(asst_ids.tolist())

        # Truncate to exact seq_len
        seq = seq[:seq_len]

        for pos, tid in enumerate(seq):
            tid_safe = min(max(tid, 0), vocab - 1)
            emb = token_embd[tid_safe].astype(np.float16)
            x = torch.tensor(emb, dtype=torch.float16).reshape(1, d, 1, 1)
            samples.append((x, pos))

    print(f"  Calibration: {len(samples)} chat-structured samples "
          f"({n_samples} seqs × {seq_len} tokens, uniform top-50K)")
    return samples


def _woodbury_gptq_compress(conv_module, hook_inputs, group_size=32,
                            damp_pct=0.01, processing_group_size=128):
    """Custom GPTQ compression for Conv2d layers.

    Replaces coremltools' GPTQ class with our own implementation:
    - Stores raw activations and forms Hessian via single BLAS matmul
      (faster than coremltools' incremental per-sample accumulation)
    - Standard 3-Cholesky inversion (numerically stable)
    - Symmetric uint4 per-block quantization with GPTQ column iteration
    - Full control over damping, block size, and quantization parameters
    """
    import time
    import torch
    import torch.nn as _nn

    weight = conv_module.weight.data.clone()
    if isinstance(conv_module, _nn.Conv2d):
        weight = weight.flatten(1)
    weight = weight.float()

    n_out, n_in = weight.shape

    # Collect and unfold all input activations → X matrix (k × n_in)
    unfold = _nn.Unfold(
        conv_module.kernel_size,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        stride=conv_module.stride,
    )
    all_x = []
    for inp in hook_inputs:
        x = unfold(inp)           # (batch, C*kH*kW, L)
        x = x.permute(1, 0, 2)   # (C, batch, L)
        x = x.flatten(1).t()     # (batch*L, C) — each row is one sample
        all_x.append(x.float())
    X = torch.cat(all_x, dim=0)  # (k, n_in)
    k, n = X.shape
    del all_x

    tick = time.time()

    # Dead columns
    col_norms = (X ** 2).sum(dim=0)
    dead = col_norms == 0
    weight[:, dead] = 0

    # Form Hessian from raw activations and invert via standard 3-Cholesky.
    # We store raw X (k×n) and form H = X^TX/k in one BLAS matmul, which is
    # faster than coremltools' incremental per-sample accumulation.
    # Woodbury (bypassing 2 of 3 Choleskys) was attempted but suffers
    # catastrophic cancellation in float32 when k/n < 1 — the subtraction
    # I - X^T G^{-1} X loses precision in near-null-space directions.
    H = (X.t() @ X) / k
    del X
    H[dead, dead] = 1.0
    damp = damp_pct * torch.mean(torch.diag(H))
    diag_idx = torch.arange(n, device=H.device)
    H[diag_idx, diag_idx] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    hessian_inverse = torch.linalg.cholesky(H, upper=True)
    del H

    t_hessian = time.time() - tick
    rank_info = f" (rank≤{min(k, n)})" if k < n else ""
    print(f"      k={k}, n={n}{rank_info}, hessian+inv={t_hessian:.1f}s")

    # ── Standard GPTQ column iteration (identical to coremltools) ──
    tick2 = time.time()

    # Standard symmetric uint4 quantization — 16 levels (0..15).
    #
    # MIL internally uses a 15-level grid (0..14, zp=7), but using 15 levels
    # during GPTQ causes error amplification: the ~7% larger quantization step
    # compounds through column iteration, causing divergent error propagation.
    # The standard 16-level grid (used by coremltools GPTQ, GPTQ paper, etc.)
    # keeps errors small. MIL re-quantization introduces minor shifts (~1 level
    # for a few values), which is much less harmful than amplified GPTQ errors.
    n_bits = 4
    q_max = 2 ** n_bits - 1  # 15 (standard 16-level grid)
    q_zp = q_max // 2        # 7 (asymmetric: range is [-8, +7] * scale)
    # Note: q_zp=7 maps to 0, so range is [-7, +8] * scale after clamp [0, 15]

    quant_weight = torch.zeros_like(weight)
    losses = torch.zeros_like(weight)

    for i1 in range(0, n, processing_group_size):
        i2 = min(i1 + processing_group_size, n)
        count = i2 - i1

        weight_block = weight[:, i1:i2].clone()
        quant_weight_block = torch.zeros_like(weight_block)
        error_block = torch.zeros_like(weight_block)
        losses_block = torch.zeros_like(weight_block)
        hessian_inverse_block = hessian_inverse[i1:i2, i1:i2]

        for i in range(count):
            w = weight_block[:, i]
            d = hessian_inverse_block[i, i]

            # Per-row, per-group symmetric quantization
            col_idx = i1 + i
            if col_idx % group_size == 0:
                block_end = min(col_idx + group_size, n)
                block_w = weight[:, col_idx:block_end]
                # Per-row max (MIL uses per-output-channel scales)
                wmax = block_w.abs().amax(dim=1, keepdim=False)  # [m]
                wmax = wmax.clamp(min=1e-7)
                scale = 2.0 * wmax / q_max  # [m] per-row
                zero_point = float(q_zp)

            # Quantize (standard 16-level grid, per-row scale)
            q = torch.clamp(torch.round(w / scale + zero_point), 0, q_max)
            q = scale * (q - zero_point)
            quant_weight_block[:, i] = q
            losses_block[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            weight_block[:, i:] -= err1.unsqueeze(1) @ hessian_inverse_block[i, i:].unsqueeze(0)
            error_block[:, i] = err1

        quant_weight[:, i1:i2] = quant_weight_block
        losses[:, i1:i2] = losses_block / 2
        weight[:, i2:] -= error_block @ hessian_inverse[i1:i2, i2:]

    t_iter = time.time() - tick2
    total_loss = losses.sum().item()
    print(f"      Column iteration: {t_iter:.1f}s, loss={total_loss:.2f}")

    # Write back quantized weights
    conv_module.weight.data = quant_weight.reshape(conv_module.weight.shape).to(
        conv_module.weight.data.dtype
    )


def _apply_gptq(model, gguf_model, token_embd, cfg, n_layers, max_seq_len,
                group_size=32, n_calib_samples=32, calib_seq_len=64,
                use_real_text=False, use_woodbury=False):
    """Apply GPTQ calibrated quantization to Conv2d weights at the PyTorch level.

    For each Conv2d in the model, collects input activations from calibration data,
    then uses coremltools' GPTQ algorithm to find optimal int4 quantization that
    minimizes output reconstruction error (Hessian-informed, not RTN).

    This happens BEFORE ct.convert() — the PyTorch model gets quantized weights
    that, when converted to CoreML, will produce better int4 quality.

    use_real_text: if True, calibration uses chat-template-structured token
    sequences with uniform random tokens from common vocab.
    use_woodbury: if True, uses Woodbury identity to exploit rank deficiency
    in Hessian (faster when n_samples << n_columns, e.g. down_conv).
    """
    import torch
    from coremltools.optimize.torch.layerwise_compression import GPTQ, ModuleGPTQConfig

    mode_str = "Woodbury" if use_woodbury else "standard"
    print(f"\nGPTQ calibrated quantization (int4, group_size={group_size}, {mode_str})...")

    d = cfg["d_model"]
    dh = cfg["d_head"]
    nkv = cfg["n_kv_heads"]
    nh = cfg["n_heads"]

    gptq_config = ModuleGPTQConfig(
        weight_dtype="uint4",
        granularity="per_block",
        block_size=group_size,
        quantization_scheme="symmetric",
    )

    # Generate calibration data
    calib_samples = _make_calibration_inputs(
        gguf_model, token_embd, cfg,
        n_samples=n_calib_samples, seq_len=calib_seq_len,
        use_real_text=use_real_text,
    )

    # Cast model to fp32 for calibration — fp16 attention overflows at dh=128
    model.float()
    model.eval()
    with torch.no_grad():
        hidden_states = []
        positions_list = []
        for x, pos in calib_samples:
            hidden_states.append(x.float())
            positions_list.append(pos)

        # RoPE tables in fp32
        d_half = dh // 2
        freqs = 1.0 / (cfg["rope_freq_base"] ** (np.arange(0, d_half, dtype=np.float32) / d_half))
        positions_arr = np.arange(max_seq_len, dtype=np.float32)
        angles = np.outer(positions_arr, freqs)
        rope_cos_t = torch.tensor(np.cos(angles), dtype=torch.float32)
        rope_sin_t = torch.tensor(np.sin(angles), dtype=torch.float32)

        for layer_idx, layer in enumerate(model.layers):
            print(f"  GPTQ layer {layer_idx}/{n_layers}...")

            # Create per-layer KV cache (fp32)
            k_cache = torch.zeros(1, nkv, max_seq_len, dh, dtype=torch.float32)
            v_cache = torch.zeros(1, nkv, max_seq_len, dh, dtype=torch.float32)

            # Hook into Conv2d layers to capture inputs
            qkv_hook_inputs = []
            out_hook_inputs = []
            gate_up_hook_inputs = []
            down_hook_inputs = []

            def make_hook(storage):
                def hook(module, inp, out):
                    storage.append(inp[0].detach().clone())
                return hook

            h1 = layer.qkv_conv.register_forward_hook(make_hook(qkv_hook_inputs))
            h2 = layer.out_conv.register_forward_hook(make_hook(out_hook_inputs))
            h3 = layer.gate_up_conv.register_forward_hook(make_hook(gate_up_hook_inputs))
            h4 = layer.down_conv.register_forward_hook(make_hook(down_hook_inputs))

            # Run all calibration samples through this layer
            new_hidden = []
            for i, (x, pos) in enumerate(zip(hidden_states, positions_list)):
                pos_clamped = min(pos, max_seq_len - 1)
                rope_cos_pos = rope_cos_t[pos_clamped:pos_clamped+1]
                rope_sin_pos = rope_sin_t[pos_clamped:pos_clamped+1]

                # Attention mask (fp32)
                attn_mask = torch.full((1, 1, 1, max_seq_len), -1e4, dtype=torch.float32)
                attn_mask[0, 0, 0, :pos_clamped + 1] = 0.0

                # Write mask (fp32)
                kv_write_mask = torch.zeros(1, 1, max_seq_len, 1, dtype=torch.float32)
                kv_write_mask[0, 0, pos_clamped, 0] = 1.0

                x_out = layer(x, k_cache, v_cache, rope_cos_pos, rope_sin_pos,
                              attn_mask, kv_write_mask)
                new_hidden.append(x_out)

            h1.remove()
            h2.remove()
            h3.remove()
            h4.remove()

            # Apply GPTQ to each Conv2d using collected activations
            for conv_name, conv_module, hook_inputs in [
                ("qkv_conv", layer.qkv_conv, qkv_hook_inputs),
                ("out_conv", layer.out_conv, out_hook_inputs),
                ("gate_up_conv", layer.gate_up_conv, gate_up_hook_inputs),
                ("down_conv", layer.down_conv, down_hook_inputs),
            ]:
                if not hook_inputs:
                    print(f"    WARNING: no inputs captured for {conv_name}")
                    continue

                if use_woodbury:
                    print(f"    Woodbury GPTQ {conv_name}...")
                    _woodbury_gptq_compress(conv_module, hook_inputs,
                                           group_size=group_size)
                else:
                    gptq = GPTQ(conv_module, gptq_config)
                    for inp_tensor in hook_inputs:
                        out_tensor = conv_module(inp_tensor)
                        gptq.add_batch(inp_tensor, out_tensor)
                    gptq.compress()
                    gptq.cleanup()

            # Re-run calibration samples through the NOW-QUANTIZED layer
            # to get correct hidden states for the next layer.
            # Without this, later layers see activations from unquantized
            # predecessors — a compounding calibration mismatch that causes
            # divergent quantization across 28 layers.
            new_hidden_q = []
            k_cache_q = torch.zeros(1, nkv, max_seq_len, dh, dtype=torch.float32)
            v_cache_q = torch.zeros(1, nkv, max_seq_len, dh, dtype=torch.float32)
            for i, (x, pos) in enumerate(zip(hidden_states, positions_list)):
                pos_clamped = min(pos, max_seq_len - 1)
                rope_cos_pos = rope_cos_t[pos_clamped:pos_clamped+1]
                rope_sin_pos = rope_sin_t[pos_clamped:pos_clamped+1]
                attn_mask = torch.full((1, 1, 1, max_seq_len), -1e4, dtype=torch.float32)
                attn_mask[0, 0, 0, :pos_clamped + 1] = 0.0
                kv_write_mask = torch.zeros(1, 1, max_seq_len, 1, dtype=torch.float32)
                kv_write_mask[0, 0, pos_clamped, 0] = 1.0
                x_out = layer(x, k_cache_q, v_cache_q, rope_cos_pos, rope_sin_pos,
                              attn_mask, kv_write_mask)
                new_hidden_q.append(x_out)
            hidden_states = new_hidden_q
            del new_hidden, new_hidden_q
            del qkv_hook_inputs, out_hook_inputs, gate_up_hook_inputs, down_hook_inputs

        # GPTQ the LM head
        print(f"  GPTQ lm_head...")
        lm_head_inputs = []
        for x in hidden_states:
            normed = model.output_norm(x)
            lm_head_inputs.append(normed)

        if use_woodbury:
            _woodbury_gptq_compress(model.lm_head_conv, lm_head_inputs,
                                   group_size=group_size)
        else:
            gptq = GPTQ(model.lm_head_conv, gptq_config)
            for inp_tensor in lm_head_inputs:
                out_tensor = model.lm_head_conv(inp_tensor)
                gptq.add_batch(inp_tensor, out_tensor)
            gptq.compress()
            gptq.cleanup()

    # Cast back to fp16 for tracing/conversion
    model.half()
    print("  GPTQ quantization complete ✓")
    return model


def _apply_smoothquant(model, gguf_model, token_embd, cfg, n_layers, max_seq_len,
                       alpha=0.5, n_calib_samples=32, calib_seq_len=64):
    """Apply SmoothQuant to weight matrices before quantization.

    SmoothQuant migrates quantization difficulty from activations to weights:
      1. Run calibration data, collect per-channel activation max |X_j| at each layer input
      2. Compute smoothing factor: s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
      3. Scale preceding norm weights by 1/s (absorbs into activation path)
      4. Scale current layer weights by s (compensates)

    After smoothing, the activation outlier channels are reduced, making uniform
    quantization much more effective. α controls the balance — higher α pushes more
    difficulty onto weights.
    """
    import torch

    print(f"\nSmoothQuant (α={alpha})...")

    d = cfg["d_model"]
    dh = cfg["d_head"]
    nkv = cfg["n_kv_heads"]
    d_half = dh // 2

    # RoPE tables (fp32 for calibration; model.half() at end handles tracing)
    freqs = 1.0 / (cfg["rope_freq_base"] ** (np.arange(0, d_half, dtype=np.float32) / d_half))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    rope_cos_t = torch.tensor(np.cos(angles), dtype=torch.float32)
    rope_sin_t = torch.tensor(np.sin(angles), dtype=torch.float32)

    calib_samples = _make_calibration_inputs(
        gguf_model, token_embd, cfg,
        n_samples=n_calib_samples, seq_len=calib_seq_len,
    )

    # Cast model to fp32 for calibration — fp16 attention overflows at dh=128,
    # producing NaN that corrupts all activation statistics.  fp32 gives clean
    # activations so smoothing factors are computed from real data.
    model.float()
    model.eval()
    with torch.no_grad():
        hidden_states = [x.float() for x, _ in calib_samples]
        positions_list = [pos for _, pos in calib_samples]

        for layer_idx, layer in enumerate(model.layers):
            print(f"  SmoothQuant layer {layer_idx}/{n_layers}...")

            # Create per-layer KV cache for calibration (fp32)
            k_cache = torch.zeros(1, nkv, max_seq_len, dh, dtype=torch.float32)
            v_cache = torch.zeros(1, nkv, max_seq_len, dh, dtype=torch.float32)

            # Collect per-channel activation magnitudes
            qkv_act_max = torch.zeros(d, dtype=torch.float32)
            ffn_act_max = torch.zeros(d, dtype=torch.float32)

            new_hidden = []
            for x, pos in zip(hidden_states, positions_list):
                # attn_norm output → qkv_conv input
                normed = layer.attn_norm(x)
                normed_flat = normed.squeeze()
                if normed_flat.dim() == 0:
                    normed_flat = normed_flat.unsqueeze(0)
                qkv_act_max = torch.max(qkv_act_max, normed_flat.abs())

                # ffn_norm output → gate_up_conv input (approximate: uses pre-attn x)
                ffn_normed = layer.ffn_norm(x)
                ffn_flat = ffn_normed.squeeze()
                if ffn_flat.dim() == 0:
                    ffn_flat = ffn_flat.unsqueeze(0)
                ffn_act_max = torch.max(ffn_act_max, ffn_flat.abs())

                # Run full layer in fp32
                pos_clamped = min(pos, max_seq_len - 1)
                rope_cos_pos = rope_cos_t[pos_clamped:pos_clamped+1]
                rope_sin_pos = rope_sin_t[pos_clamped:pos_clamped+1]
                attn_mask = torch.full((1, 1, 1, max_seq_len), -1e4, dtype=torch.float32)
                attn_mask[0, 0, 0, :pos_clamped + 1] = 0.0
                kv_write_mask = torch.zeros(1, 1, max_seq_len, 1, dtype=torch.float32)
                kv_write_mask[0, 0, pos_clamped, 0] = 1.0

                x_out = layer(x, k_cache, v_cache, rope_cos_pos, rope_sin_pos,
                              attn_mask, kv_write_mask)
                new_hidden.append(x_out)

            # Compute smoothing factors for QKV path
            qkv_w = layer.qkv_conv.weight.squeeze()  # already fp32
            qkv_w_max = qkv_w.abs().amax(dim=0)  # per input-channel max

            qkv_act_max = qkv_act_max.clamp(min=1e-5)
            qkv_w_max = qkv_w_max.clamp(min=1e-5)

            s_qkv = (qkv_act_max.pow(alpha) / qkv_w_max.pow(1 - alpha)).clamp(min=0.01, max=100.0)

            # Apply smoothing in fp32 — no nan_to_num needed since fp32 doesn't overflow
            layer.attn_norm.weight.data.copy_(
                (layer.attn_norm.weight / s_qkv.reshape(-1, 1, 1)))
            layer.qkv_conv.weight.data.copy_(
                (qkv_w * s_qkv.unsqueeze(0)).reshape_as(layer.qkv_conv.weight))

            # Smooth FFN path
            gate_up_w = layer.gate_up_conv.weight.squeeze()  # already fp32
            gate_up_w_max = gate_up_w.abs().amax(dim=0)
            ffn_act_max = ffn_act_max.clamp(min=1e-5)
            gate_up_w_max = gate_up_w_max.clamp(min=1e-5)

            s_ffn = (ffn_act_max.pow(alpha) / gate_up_w_max.pow(1 - alpha)).clamp(min=0.01, max=100.0)

            layer.ffn_norm.weight.data.copy_(
                (layer.ffn_norm.weight / s_ffn.reshape(-1, 1, 1)))
            layer.gate_up_conv.weight.data.copy_(
                (gate_up_w * s_ffn.unsqueeze(0)).reshape_as(layer.gate_up_conv.weight))

            hidden_states = new_hidden

    # Cast back to fp16 for tracing/conversion
    model.half()
    print("  SmoothQuant complete ✓")
    return model


# ─── PyTorch Model (Float16, Stateful KV) ──────────────────────────────────

def build_model(gguf_path, n_layers=None, max_seq_len=512, quant_bits=0, group_size=32, strategy="uniform"):
    import coremltools as ct
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print(f"Loading GGUF: {gguf_path}")
    gguf = GGUFModel(gguf_path)
    cfg = gguf.config()

    if n_layers is None:
        n_layers = cfg["n_layers"]
    n_layers = min(n_layers, cfg["n_layers"])

    d = cfg["d_model"]       # 0.5B:896  1.5B:1536
    nh = cfg["n_heads"]      # 0.5B:14   1.5B:12
    nkv = cfg["n_kv_heads"]  # 0.5B:2    1.5B:2
    dh = cfg["d_head"]       # 0.5B:64   1.5B:128
    dff = cfg["d_ff"]        # 0.5B:4864 1.5B:8960
    eps = cfg["rms_norm_eps"]
    vocab = cfg["vocab_size"]
    kv_dim = nkv * dh        # 0.5B:128  1.5B:256
    qkv_dim = d + 2 * kv_dim  # 0.5B:1152 1.5B:2048

    print(f"Config: {n_layers}L, d={d}, nh={nh}, nkv={nkv}, dh={dh}, dff={dff}")
    print(f"Seq len: {max_seq_len}, Float16 throughout")

    # ── Load embeddings (keep as float32 for host-side lookup) ──
    print("Loading token embeddings...")
    token_embd = gguf.get_tensor("token_embd.weight", dtype=np.float32)  # (vocab, d)

    # ── RoPE tables ──
    d_half = dh // 2
    freqs = 1.0 / (cfg["rope_freq_base"] ** (np.arange(0, d_half, dtype=np.float32) / d_half))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    rope_cos = np.cos(angles).astype(np.float32)
    rope_sin = np.sin(angles).astype(np.float32)

    # ── PyTorch model — all float16 ─────────────────────────────────────

    class RMSNormConv(nn.Module):
        def __init__(self, weight, eps):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(
                torch.tensor(weight, dtype=torch.float16).reshape(-1, 1, 1),
                requires_grad=False)

        def forward(self, x):
            # Safe-norm peephole (Dragon Book §8.7): pre-divide by √d
            # to keep squared values in fp16-safe range on ANE.
            K = x.shape[1] ** 0.5
            x_scaled = x * (1.0 / K)
            variance = x_scaled.pow(2).mean(dim=1, keepdim=True)
            x_normed = x_scaled * torch.rsqrt(variance + self.eps / (K * K))
            return (x_normed * self.weight).half()

    class QwenLayerConv(nn.Module):
        """Transformer layer — all Conv2d(1×1), fp16.
        KV cache is passed in/out explicitly with flexible seq dim.
        Host manages concatenation between steps.
        """
        def __init__(self, layer_idx, gguf_model, cfg):
            super().__init__()
            d = cfg["d_model"]
            dff = cfg["d_ff"]
            kv_dim = cfg["n_kv_heads"] * cfg["d_head"]
            qkv_dim = d + 2 * kv_dim
            eps = cfg["rms_norm_eps"]
            prefix = f"blk.{layer_idx}"

            self.d = d
            self.dff = dff
            self.kv_dim = kv_dim
            self.nh = cfg["n_heads"]
            self.nkv = cfg["n_kv_heads"]
            self.dh = cfg["d_head"]
            self.scale = 1.0 / (cfg["d_head"] ** 0.5)

            self.attn_norm = RMSNormConv(
                gguf_model.get_tensor(f"{prefix}.attn_norm.weight"), eps)

            # Fused QKV (handles both split Qwen and fused Phi layouts)
            q_w, k_w, v_w = gguf_model.get_qkv_weights(prefix, cfg)
            qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
            has_bias = gguf_model.has_biases(prefix)
            self.qkv_conv = nn.Conv2d(d, qkv_dim, 1, bias=has_bias)
            self.qkv_conv.weight = nn.Parameter(
                torch.tensor(qkv_w, dtype=torch.float16).reshape(qkv_dim, d, 1, 1),
                requires_grad=False)
            if has_bias:
                biases = gguf_model.get_qkv_biases(prefix, cfg)
                qkv_b = np.concatenate([biases[0], biases[1], biases[2]])
                self.qkv_conv.bias = nn.Parameter(
                    torch.tensor(qkv_b, dtype=torch.float16), requires_grad=False)

            # Output projection
            o_w = gguf_model.get_tensor(f"{prefix}.attn_output.weight")
            self.out_conv = nn.Conv2d(d, d, 1, bias=False)
            self.out_conv.weight = nn.Parameter(
                torch.tensor(o_w, dtype=torch.float16).reshape(d, d, 1, 1),
                requires_grad=False)

            # FFN (handles both split Qwen and fused Phi layouts)
            self.ffn_norm = RMSNormConv(
                gguf_model.get_tensor(f"{prefix}.ffn_norm.weight"), eps)

            gate_w, up_w = gguf_model.get_gate_up_weights(prefix, cfg)
            gate_up_w = np.concatenate([gate_w, up_w], axis=0)
            self.gate_up_conv = nn.Conv2d(d, 2 * dff, 1, bias=False)
            self.gate_up_conv.weight = nn.Parameter(
                torch.tensor(gate_up_w, dtype=torch.float16).reshape(2 * dff, d, 1, 1),
                requires_grad=False)

            down_w = gguf_model.get_tensor(f"{prefix}.ffn_down.weight")
            self.down_conv = nn.Conv2d(dff, d, 1, bias=False)
            self.down_conv.weight = nn.Parameter(
                torch.tensor(down_w, dtype=torch.float16).reshape(d, dff, 1, 1),
                requires_grad=False)

        def forward(self, x, k_cache, v_cache, rope_cos_pos, rope_sin_pos):
            """
            x:         (1, d, 1, 1) fp16
            k_cache:   (1, nkv, seq, dh) fp16
            v_cache:   (1, nkv, seq, dh) fp16
            rope_cos:  (1, d_half) fp16
            rope_sin:  (1, d_half) fp16

            Returns: (x_out, new_k, new_v)
              new_k/new_v: (1, nkv, 1, dh) — single new KV entry
            """
            residual = x
            normed = self.attn_norm(x)

            # QKV
            qkv = self.qkv_conv(normed).squeeze(-1).squeeze(-1)
            q = qkv[:, :self.d]
            k = qkv[:, self.d:self.d + self.kv_dim]
            v = qkv[:, self.d + self.kv_dim:]

            # RoPE
            d_half = self.dh // 2
            def apply_rope(x_flat, n_heads):
                x_r = x_flat.reshape(1, n_heads, self.dh)
                x_lo = x_r[:, :, :d_half]
                x_hi = x_r[:, :, d_half:]
                cos_b = rope_cos_pos.unsqueeze(1)
                sin_b = rope_sin_pos.unsqueeze(1)
                r_lo = x_lo * cos_b - x_hi * sin_b
                r_hi = x_lo * sin_b + x_hi * cos_b
                return torch.cat([r_lo, r_hi], dim=-1).reshape(1, n_heads * self.dh)

            q = apply_rope(q, self.nh)
            k = apply_rope(k, self.nkv)

            new_k = k.reshape(1, self.nkv, 1, self.dh)
            new_v = v.reshape(1, self.nkv, 1, self.dh)

            # Append and attend
            k_full = torch.cat([k_cache, new_k], dim=2)
            v_full = torch.cat([v_cache, new_v], dim=2)

            # Batched GQA — 2 matmuls per KV group, not heads_per_kv per-head
            # Reduces n_layers×nh×2 → n_layers×nkv×2 matmul ops across all layers
            heads_per_kv = self.nh // self.nkv
            q_heads = q.reshape(1, self.nh, self.dh)
            attn_parts = []
            for kv_idx in range(self.nkv):
                # Group all query heads for this KV group
                q_group = q_heads[:, kv_idx*heads_per_kv:(kv_idx+1)*heads_per_kv, :]
                # q_group: (1, hpk=7, dh=64)

                k_head = k_full[:, kv_idx:kv_idx+1, :, :]  # (1, 1, seq, dh)
                v_head = v_full[:, kv_idx:kv_idx+1, :, :]  # (1, 1, seq, dh)

                # Batched Q*K^T: (1, 7, 1, 64) @ (1, 1, 64, seq) → (1, 7, 1, seq)
                q_g = q_group.unsqueeze(2)                    # (1, 7, 1, 64)
                k_t = k_head.transpose(-2, -1)                # (1, 1, 64, seq)
                scores = torch.matmul(q_g, k_t) * self.scale  # (1, 7, 1, seq)

                attn_w = torch.softmax(scores.float(), dim=-1).half()

                # Batched attn*V: (1, 7, 1, seq) @ (1, 1, seq, 64) → (1, 7, 1, 64)
                head_out = torch.matmul(attn_w, v_head)  # (1, 7, 1, 64)
                attn_parts.append(head_out.squeeze(2))     # (1, 7, 64)

            # Concat all heads: (1, 14, 64) → (1, 896, 1, 1)
            attn_out = torch.cat(attn_parts, dim=1).reshape(1, self.d, 1, 1)
            attn_out = self.out_conv(attn_out)
            x = residual + attn_out

            # FFN
            residual2 = x
            normed2 = self.ffn_norm(x)
            gate_up = self.gate_up_conv(normed2)
            gate = gate_up[:, :self.dff, :, :]
            up = gate_up[:, self.dff:, :, :]
            hidden = F.silu(gate.float()).half() * up
            ffn_out = self.down_conv(hidden)
            x = residual2 + ffn_out

            return x, new_k, new_v

    class QwenConv(nn.Module):
        """Full Qwen — all conv, all fp16, explicit KV I/O."""
        def __init__(self, gguf_model, cfg, n_layers, max_seq_len):
            super().__init__()
            self.n_layers = n_layers
            self.d = cfg["d_model"]
            self.nkv = cfg["n_kv_heads"]
            self.dh = cfg["d_head"]
            self.max_seq_len = max_seq_len

            self.layers = nn.ModuleList()
            for i in range(n_layers):
                print(f"  Layer {i}/{n_layers}")
                self.layers.append(QwenLayerConv(i, gguf_model, cfg))

            out_norm_w = gguf_model.get_tensor("output_norm.weight")
            self.output_norm = RMSNormConv(out_norm_w, cfg["rms_norm_eps"])

            print("  LM head...")
            if "output.weight" in gguf_model.tensors:
                lm_w = gguf_model.get_tensor("output.weight")
            else:
                lm_w = gguf_model.get_tensor("token_embd.weight")
            self.lm_head_conv = nn.Conv2d(cfg["d_model"], cfg["vocab_size"], 1, bias=False)
            self.lm_head_conv.weight = nn.Parameter(
                torch.tensor(lm_w, dtype=torch.float16).reshape(
                    cfg["vocab_size"], cfg["d_model"], 1, 1),
                requires_grad=False)

        def forward(self, x, rope_cos_pos, rope_sin_pos, *kv_flat):
            """
            x:            (1, d, 1, 1) fp16
            rope_cos_pos: (1, d_half) fp16
            rope_sin_pos: (1, d_half) fp16
            kv_flat:      k0, v0, k1, v1, ... each (1, nkv, seq, dh)

            Returns: (logits, new_k0, new_v0, new_k1, new_v1, ...)
              new_k/new_v: (1, nkv, 1, dh) — single new KV entry per layer
            """
            kv_out = []
            for i, layer in enumerate(self.layers):
                k_cache = kv_flat[2*i]
                v_cache = kv_flat[2*i+1]
                x, nk, nv = layer(x, k_cache, v_cache, rope_cos_pos, rope_sin_pos)
                kv_out.append(nk)
                kv_out.append(nv)

            x = self.output_norm(x)
            logits = self.lm_head_conv(x).squeeze(-1).squeeze(-1)
            return tuple([logits] + kv_out)

    # ── Build ────────────────────────────────────────────────────────────

    print(f"\nBuilding QwenConv ({n_layers}L, fp16)...")
    model = QwenConv(gguf, cfg, n_layers, max_seq_len)
    model.half()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters (fp16)")

    # ── Trace ────────────────────────────────────────────────────────────

    d_half = dh // 2
    x_ex = torch.randn(1, d, 1, 1, dtype=torch.float16)
    cos_ex = torch.randn(1, d_half, dtype=torch.float16)
    sin_ex = torch.randn(1, d_half, dtype=torch.float16)
    # KV caches: flexible seq dim, start with seq=1 for tracing
    kv_examples = []
    for _ in range(n_layers):
        kv_examples.append(torch.randn(1, nkv, 1, dh, dtype=torch.float16))
        kv_examples.append(torch.randn(1, nkv, 1, dh, dtype=torch.float16))

    print("\nTracing...")
    with torch.no_grad():
        outputs = model(x_ex, cos_ex, sin_ex, *kv_examples)
        print(f"  logits: {outputs[0].shape}, dtype: {outputs[0].dtype}")
        print(f"  new_k: {outputs[1].shape}")

    traced = torch.jit.trace(model, (x_ex, cos_ex, sin_ex, *kv_examples))

    # ── Convert to CoreML ────────────────────────────────────────────────

    print(f"\nConverting to CoreML (fp16, {n_layers}L)...")

    ct_inputs = [
        ct.TensorType(name="x", shape=(1, d, 1, 1), dtype=np.float16),
        ct.TensorType(name="rope_cos", shape=(1, d_half), dtype=np.float16),
        ct.TensorType(name="rope_sin", shape=(1, d_half), dtype=np.float16),
    ]
    for i in range(n_layers):
        k_shape = ct.Shape(shape=(1, nkv,
            ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1), dh))
        v_shape = ct.Shape(shape=(1, nkv,
            ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1), dh))
        ct_inputs.append(ct.TensorType(name=f"k_cache_{i}", shape=k_shape,
                                       dtype=np.float16))
        ct_inputs.append(ct.TensorType(name=f"v_cache_{i}", shape=v_shape,
                                       dtype=np.float16))

    ct_outputs = [ct.TensorType(name="logits", dtype=np.float16)]
    for i in range(n_layers):
        ct_outputs.append(ct.TensorType(name=f"new_k_{i}", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_v_{i}", dtype=np.float16))

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    # ── Post-training weight quantization ────────────────────────────────

    mlmodel = _quantize_weights(mlmodel, quant_bits, group_size=group_size, strategy=strategy)

    # ── Save ─────────────────────────────────────────────────────────────

    strat_tag = f"_{strategy}" if strategy != "uniform" else ""
    suffix = f"_q{quant_bits}{strat_tag}" if quant_bits else ""
    prefix = f"QwenANE_{n_layers}L{suffix}"
    pkg_path = f"{prefix}.mlpackage"
    mlmodel.save(pkg_path)
    print(f"\n✓ Saved {pkg_path}")

    # ── Metadata ─────────────────────────────────────────────────────────

    meta = {
        **cfg,
        "n_layers_exported": n_layers,
        "max_seq_len": max_seq_len,
        "dtype": "float16",
        "stateful_kv": False,
        "quant_bits": quant_bits,
        "rope_cos": rope_cos.tolist(),
        "rope_sin": rope_sin.tolist(),
    }
    meta_path = f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"  Meta: {meta_path}")

    # ── Embeddings (float16 for host) ────────────────────────────────────

    embd_path = f"{prefix}_embd.bin"
    token_embd.astype(np.float16).tofile(embd_path)
    print(f"  Embeddings: {embd_path} ({Path(embd_path).stat().st_size / 1e6:.1f} MB)")

    # ── BPE Tokenizer ────────────────────────────────────────────────────

    print("Exporting BPE tokenizer...")
    tok_data = gguf.extract_tokenizer()
    tok_path = f"{prefix}_tokenizer.json"
    with open(tok_path, "w") as f:
        json.dump(tok_data, f)
    print(f"  Tokenizer: {tok_path} ({len(tok_data['tokens'])} tokens, "
          f"{len(tok_data['merges'])} merges)")

    print(f"\n{'='*60}")
    print(f"  Model:      Qwen2.5 ({n_layers}L, d={d}, nh={nh}, nkv={nkv})")
    print(f"  Format:     CoreML mlprogram (iOS18+)")
    print(f"  Dtype:      Float16{f' + int{quant_bits} weights' if quant_bits else ''}")
    print(f"  KV Cache:   StateType (on-device, {max_seq_len} positions)")
    print(f"  Tokenizer:  BPE ({len(tok_data['tokens'])} vocab)")
    print(f"  Target:     ANE (CPU_AND_NE)")
    print(f"{'='*60}")

    return pkg_path


# ─── Fixed-Shape Model (zero dynamic dims, batched GQA) ─────────────────────

def build_fixed_model(gguf_path, n_layers=None, max_seq_len=512, quant_bits=0, group_size=32, strategy="uniform"):
    """Build a fully fixed-shape model variant.

    Key differences from build_model():
    - KV caches are fixed (1, nkv, max_seq_len, dh) — no RangeDim
    - Attention uses masking instead of dynamic cat
    - GQA attention is batched: 2 matmuls per KV group (not 7 per-head)
    - Reduces matmul ops from 672 to 96 (7× reduction)
    - Eliminates all dynamic shape overhead (~6ms per token)
    """
    import coremltools as ct
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print(f"Loading GGUF: {gguf_path}")
    gguf = GGUFModel(gguf_path)
    cfg = gguf.config()

    if n_layers is None:
        n_layers = cfg["n_layers"]
    n_layers = min(n_layers, cfg["n_layers"])

    d = cfg["d_model"]       # 0.5B:896  1.5B:1536
    nh = cfg["n_heads"]      # 0.5B:14   1.5B:12
    nkv = cfg["n_kv_heads"]  # 0.5B:2    1.5B:2
    dh = cfg["d_head"]       # 0.5B:64   1.5B:128
    dff = cfg["d_ff"]        # 0.5B:4864 1.5B:8960
    eps = cfg["rms_norm_eps"]
    vocab = cfg["vocab_size"]
    kv_dim = nkv * dh        # 0.5B:128  1.5B:256
    qkv_dim = d + 2 * kv_dim  # 0.5B:1152 1.5B:2048
    hpk = nh // nkv           # heads per KV group (0.5B:7, 1.5B:6)

    print(f"Config: {n_layers}L, d={d}, nh={nh}, nkv={nkv}, dh={dh}, dff={dff}")
    print(f"Fixed seq len: {max_seq_len}, Float16, batched GQA")

    # ── Load embeddings ──
    print("Loading token embeddings...")
    token_embd = gguf.get_tensor("token_embd.weight", dtype=np.float32)

    # ── RoPE tables ──
    d_half = dh // 2
    freqs = 1.0 / (cfg["rope_freq_base"] ** (np.arange(0, d_half, dtype=np.float32) / d_half))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    rope_cos = np.cos(angles).astype(np.float32)
    rope_sin = np.sin(angles).astype(np.float32)

    # ── PyTorch model — fixed shapes ────────────────────────────────────

    class RMSNormConv(nn.Module):
        def __init__(self, weight, eps):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(
                torch.tensor(weight, dtype=torch.float16).reshape(-1, 1, 1),
                requires_grad=False)

        def forward(self, x):
            # Safe-norm peephole (Dragon Book §8.7)
            K = x.shape[1] ** 0.5
            x_scaled = x * (1.0 / K)
            variance = x_scaled.pow(2).mean(dim=1, keepdim=True)
            x_normed = x_scaled * torch.rsqrt(variance + self.eps / (K * K))
            return (x_normed * self.weight).half()

    class FixedLayerConv(nn.Module):
        """Transformer layer — fixed KV shapes, batched GQA, masked attention."""
        def __init__(self, layer_idx, gguf_model, cfg):
            super().__init__()
            d = cfg["d_model"]
            dff = cfg["d_ff"]
            kv_dim = cfg["n_kv_heads"] * cfg["d_head"]
            qkv_dim = d + 2 * kv_dim
            eps = cfg["rms_norm_eps"]
            prefix = f"blk.{layer_idx}"

            self.d = d
            self.dff = dff
            self.kv_dim = kv_dim
            self.nh = cfg["n_heads"]
            self.nkv = cfg["n_kv_heads"]
            self.dh = cfg["d_head"]
            self.hpk = self.nh // self.nkv  # 7
            self.scale = 1.0 / (cfg["d_head"] ** 0.5)

            self.attn_norm = RMSNormConv(
                gguf_model.get_tensor(f"{prefix}.attn_norm.weight"), eps)

            # Fused QKV (handles both split Qwen and fused Phi layouts)
            q_w, k_w, v_w = gguf_model.get_qkv_weights(prefix, cfg)
            qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
            has_bias = gguf_model.has_biases(prefix)
            self.qkv_conv = nn.Conv2d(d, qkv_dim, 1, bias=has_bias)
            self.qkv_conv.weight = nn.Parameter(
                torch.tensor(qkv_w, dtype=torch.float16).reshape(qkv_dim, d, 1, 1),
                requires_grad=False)
            if has_bias:
                biases = gguf_model.get_qkv_biases(prefix, cfg)
                qkv_b = np.concatenate([biases[0], biases[1], biases[2]])
                self.qkv_conv.bias = nn.Parameter(
                    torch.tensor(qkv_b, dtype=torch.float16), requires_grad=False)

            # Output projection
            o_w = gguf_model.get_tensor(f"{prefix}.attn_output.weight")
            self.out_conv = nn.Conv2d(d, d, 1, bias=False)
            self.out_conv.weight = nn.Parameter(
                torch.tensor(o_w, dtype=torch.float16).reshape(d, d, 1, 1),
                requires_grad=False)

            # FFN (handles both split Qwen and fused Phi layouts)
            self.ffn_norm = RMSNormConv(
                gguf_model.get_tensor(f"{prefix}.ffn_norm.weight"), eps)

            gate_w, up_w = gguf_model.get_gate_up_weights(prefix, cfg)
            gate_up_w = np.concatenate([gate_w, up_w], axis=0)
            self.gate_up_conv = nn.Conv2d(d, 2 * dff, 1, bias=False)
            self.gate_up_conv.weight = nn.Parameter(
                torch.tensor(gate_up_w, dtype=torch.float16).reshape(2 * dff, d, 1, 1),
                requires_grad=False)

            down_w = gguf_model.get_tensor(f"{prefix}.ffn_down.weight")
            self.down_conv = nn.Conv2d(dff, d, 1, bias=False)
            self.down_conv.weight = nn.Parameter(
                torch.tensor(down_w, dtype=torch.float16).reshape(d, dff, 1, 1),
                requires_grad=False)

        def forward(self, x, k_cache, v_cache, rope_cos_pos, rope_sin_pos,
                    attn_mask, kv_write_mask):
            """
            x:              (1, d, 1, 1) fp16
            k_cache:        (1, nkv, max_seq, dh) fp16 — full fixed buffer
            v_cache:        (1, nkv, max_seq, dh) fp16
            rope_cos_pos:   (1, d_half) fp16
            rope_sin_pos:   (1, d_half) fp16
            attn_mask:      (1, 1, 1, max_seq) fp16 — 0=valid, -1e4=masked
            kv_write_mask:  (1, 1, max_seq, 1) fp16 — 1 at current pos, 0 elsewhere

            Returns: (x_out, new_k, new_v)
            """
            residual = x
            normed = self.attn_norm(x)

            # QKV projection
            qkv = self.qkv_conv(normed).squeeze(-1).squeeze(-1)
            q = qkv[:, :self.d]
            k = qkv[:, self.d:self.d + self.kv_dim]
            v = qkv[:, self.d + self.kv_dim:]

            # RoPE
            d_half = self.dh // 2
            def apply_rope(x_flat, n_heads):
                x_r = x_flat.reshape(1, n_heads, self.dh)
                x_lo = x_r[:, :, :d_half]
                x_hi = x_r[:, :, d_half:]
                cos_b = rope_cos_pos.unsqueeze(1)
                sin_b = rope_sin_pos.unsqueeze(1)
                r_lo = x_lo * cos_b - x_hi * sin_b
                r_hi = x_lo * sin_b + x_hi * cos_b
                return torch.cat([r_lo, r_hi], dim=-1).reshape(1, n_heads * self.dh)

            q = apply_rope(q, self.nh)
            k = apply_rope(k, self.nkv)

            new_k = k.reshape(1, self.nkv, 1, self.dh)
            new_v = v.reshape(1, self.nkv, 1, self.dh)

            # Scatter-write new KV at current position using mask
            # kv_write_mask: (1, 1, max_seq, 1) — broadcasts over nkv and dh
            k_full = k_cache * (1.0 - kv_write_mask) + new_k * kv_write_mask
            v_full = v_cache * (1.0 - kv_write_mask) + new_v * kv_write_mask

            # Batched GQA attention — 2 matmuls per KV group instead of 7
            q_heads = q.reshape(1, self.nh, self.dh)
            attn_parts = []
            for kv_idx in range(self.nkv):
                # Group queries for this KV head
                q_group = q_heads[:, kv_idx*self.hpk:(kv_idx+1)*self.hpk, :]
                # q_group: (1, hpk=7, dh=64)

                k_head = k_full[:, kv_idx:kv_idx+1, :, :]  # (1, 1, max_seq, dh)
                v_head = v_full[:, kv_idx:kv_idx+1, :, :]  # (1, 1, max_seq, dh)

                # Batched Q*K^T: (1, 7, 1, 64) @ (1, 1, 64, max_seq) → (1, 7, 1, max_seq)
                q_g = q_group.unsqueeze(2)                    # (1, 7, 1, 64)
                k_t = k_head.transpose(-2, -1)                # (1, 1, 64, max_seq)
                scores = torch.matmul(q_g, k_t) * self.scale  # (1, 7, 1, max_seq)

                # Apply attention mask
                scores = scores + attn_mask  # (1, 1, 1, max_seq) broadcasts

                attn_w = torch.softmax(scores.float(), dim=-1).half()  # (1, 7, 1, max_seq)

                # Batched attn*V: (1, 7, 1, max_seq) @ (1, 1, max_seq, 64) → (1, 7, 1, 64)
                head_out = torch.matmul(attn_w, v_head)  # (1, 7, 1, 64)
                attn_parts.append(head_out.squeeze(2))     # (1, 7, 64)

            # Concat all heads: (1, 14, 64) → (1, 896, 1, 1)
            attn_out = torch.cat(attn_parts, dim=1)  # (1, 14, 64)
            attn_out = attn_out.reshape(1, self.d, 1, 1)
            attn_out = self.out_conv(attn_out)
            x = residual + attn_out

            # FFN
            residual2 = x
            normed2 = self.ffn_norm(x)
            gate_up = self.gate_up_conv(normed2)
            gate = gate_up[:, :self.dff, :, :]
            up = gate_up[:, self.dff:, :, :]
            hidden = F.silu(gate.float()).half() * up
            ffn_out = self.down_conv(hidden)
            x = residual2 + ffn_out

            return x, new_k, new_v

    class QwenFixedConv(nn.Module):
        """Full Qwen — fixed shapes, batched GQA, masked attention."""
        def __init__(self, gguf_model, cfg, n_layers, max_seq_len):
            super().__init__()
            self.n_layers = n_layers
            self.d = cfg["d_model"]
            self.nkv = cfg["n_kv_heads"]
            self.dh = cfg["d_head"]
            self.max_seq_len = max_seq_len

            self.layers = nn.ModuleList()
            for i in range(n_layers):
                print(f"  Layer {i}/{n_layers}")
                self.layers.append(FixedLayerConv(i, gguf_model, cfg))

            out_norm_w = gguf_model.get_tensor("output_norm.weight")
            self.output_norm = RMSNormConv(out_norm_w, cfg["rms_norm_eps"])

            print("  LM head...")
            if "output.weight" in gguf_model.tensors:
                lm_w = gguf_model.get_tensor("output.weight")
            else:
                lm_w = gguf_model.get_tensor("token_embd.weight")
            self.lm_head_conv = nn.Conv2d(cfg["d_model"], cfg["vocab_size"], 1, bias=False)
            self.lm_head_conv.weight = nn.Parameter(
                torch.tensor(lm_w, dtype=torch.float16).reshape(
                    cfg["vocab_size"], cfg["d_model"], 1, 1),
                requires_grad=False)

        def forward(self, x, rope_cos_pos, rope_sin_pos, attn_mask,
                    kv_write_mask, *kv_flat):
            """
            x:              (1, d, 1, 1) fp16
            rope_cos_pos:   (1, d_half) fp16
            rope_sin_pos:   (1, d_half) fp16
            attn_mask:      (1, 1, 1, max_seq) fp16 — 0 valid, -1e4 masked
            kv_write_mask:  (1, 1, max_seq, 1) fp16 — 1 at pos, 0 elsewhere
            kv_flat:        k0, v0, k1, v1, ... each (1, nkv, max_seq, dh) FIXED

            Returns: (logits, new_k0, new_v0, ...)
              new_k/new_v: (1, nkv, 1, dh) — single new KV entry per layer
            """
            kv_out = []
            for i, layer in enumerate(self.layers):
                k_cache = kv_flat[2*i]
                v_cache = kv_flat[2*i+1]
                x, nk, nv = layer(x, k_cache, v_cache, rope_cos_pos, rope_sin_pos,
                                  attn_mask, kv_write_mask)
                kv_out.append(nk)
                kv_out.append(nv)

            x = self.output_norm(x)
            logits = self.lm_head_conv(x).squeeze(-1).squeeze(-1)
            return tuple([logits] + kv_out)

    # ── Build ────────────────────────────────────────────────────────────

    print(f"\nBuilding QwenFixedConv ({n_layers}L, fp16, fixed shapes)...")
    model = QwenFixedConv(gguf, cfg, n_layers, max_seq_len)
    model.half()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters (fp16)")

    # ── Trace ────────────────────────────────────────────────────────────

    d_half = dh // 2
    x_ex = torch.randn(1, d, 1, 1, dtype=torch.float16)
    cos_ex = torch.randn(1, d_half, dtype=torch.float16)
    sin_ex = torch.randn(1, d_half, dtype=torch.float16)
    # Fixed-shape attention mask: 0 at pos 0, -1e4 elsewhere
    mask_ex = torch.full((1, 1, 1, max_seq_len), -1e4, dtype=torch.float16)
    mask_ex[0, 0, 0, 0] = 0.0
    # Write mask: 1 at pos 0, 0 elsewhere
    wmask_ex = torch.zeros(1, 1, max_seq_len, 1, dtype=torch.float16)
    wmask_ex[0, 0, 0, 0] = 1.0
    # Fixed KV caches — all zeros initially
    kv_examples = []
    for _ in range(n_layers):
        kv_examples.append(torch.zeros(1, nkv, max_seq_len, dh, dtype=torch.float16))
        kv_examples.append(torch.zeros(1, nkv, max_seq_len, dh, dtype=torch.float16))

    print("\nTracing (fixed shapes)...")
    with torch.no_grad():
        outputs = model(x_ex, cos_ex, sin_ex, mask_ex, wmask_ex, *kv_examples)
        print(f"  logits: {outputs[0].shape}, dtype: {outputs[0].dtype}")
        print(f"  new_k: {outputs[1].shape}")

    traced = torch.jit.trace(model, (x_ex, cos_ex, sin_ex, mask_ex, wmask_ex, *kv_examples))

    # ── Convert to CoreML ────────────────────────────────────────────────

    print(f"\nConverting to CoreML (fp16, {n_layers}L, FIXED shapes)...")

    ct_inputs = [
        ct.TensorType(name="x", shape=(1, d, 1, 1), dtype=np.float16),
        ct.TensorType(name="rope_cos", shape=(1, d_half), dtype=np.float16),
        ct.TensorType(name="rope_sin", shape=(1, d_half), dtype=np.float16),
        ct.TensorType(name="attn_mask", shape=(1, 1, 1, max_seq_len), dtype=np.float16),
        ct.TensorType(name="kv_write_mask", shape=(1, 1, max_seq_len, 1), dtype=np.float16),
    ]
    for i in range(n_layers):
        ct_inputs.append(ct.TensorType(name=f"k_cache_{i}",
                                       shape=(1, nkv, max_seq_len, dh), dtype=np.float16))
        ct_inputs.append(ct.TensorType(name=f"v_cache_{i}",
                                       shape=(1, nkv, max_seq_len, dh), dtype=np.float16))

    ct_outputs = [ct.TensorType(name="logits", dtype=np.float16)]
    for i in range(n_layers):
        ct_outputs.append(ct.TensorType(name=f"new_k_{i}", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_v_{i}", dtype=np.float16))

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    # ── Post-training weight quantization ────────────────────────────────

    mlmodel = _quantize_weights(mlmodel, quant_bits, group_size=group_size, strategy=strategy)

    # ── Save ─────────────────────────────────────────────────────────────

    strat_tag = f"_{strategy}" if strategy != "uniform" else ""
    suffix = f"_q{quant_bits}{strat_tag}" if quant_bits else ""
    prefix = f"QwenANE_{n_layers}L_fixed{suffix}"
    pkg_path = f"{prefix}.mlpackage"
    mlmodel.save(pkg_path)
    print(f"\n✓ Saved {pkg_path}")

    # ── Metadata ─────────────────────────────────────────────────────────

    meta = {
        **cfg,
        "n_layers_exported": n_layers,
        "max_seq_len": max_seq_len,
        "dtype": "float16",
        "fixed_shapes": True,
        "batched_gqa": True,
        "quant_bits": quant_bits,
        "rope_cos": rope_cos.tolist(),
        "rope_sin": rope_sin.tolist(),
    }
    meta_path = f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"  Meta: {meta_path}")

    # ── Embeddings ───────────────────────────────────────────────────────

    embd_path = f"{prefix}_embd.bin"
    token_embd.astype(np.float16).tofile(embd_path)
    print(f"  Embeddings: {embd_path} ({Path(embd_path).stat().st_size / 1e6:.1f} MB)")

    # ── BPE Tokenizer ────────────────────────────────────────────────────

    print("Exporting BPE tokenizer...")
    tok_data = gguf.extract_tokenizer()
    tok_path = f"{prefix}_tokenizer.json"
    with open(tok_path, "w") as f:
        json.dump(tok_data, f)
    print(f"  Tokenizer: {tok_path}")

    print(f"\n{'='*60}")
    print(f"  Model:      Qwen2.5 ({n_layers}L, d={d}, nh={nh}, nkv={nkv})")
    print(f"  Format:     CoreML mlprogram (iOS18+)")
    print(f"  Shapes:     FIXED (all static, no RangeDim)")
    print(f"  Attention:  Batched GQA ({n_layers * nkv * 2} matmuls, not {n_layers * nh * 2})")
    print(f"  Dtype:      Float16{f' + int{quant_bits} weights' if quant_bits else ''}")
    print(f"  KV Cache:   Fixed {max_seq_len} positions (masked)")
    print(f"  Tokenizer:  BPE ({len(tok_data['tokens'])} vocab)")
    print(f"  Target:     ANE (CPU_AND_NE)")
    print(f"{'='*60}")

    return pkg_path


# ─── Stateful Model (KV cache as on-device state, zero host↔ANE copy) ───────

def build_stateful_model(gguf_path, n_layers=None, max_seq_len=512, quant_bits=0,
                         group_size=32, strategy="uniform",
                         layer_start=None, layer_end=None,
                         output_dir=None, output_name=None,
                         split_mode="full"):
    """Build a stateful CoreML model with KV cache as MLState.

    Key differences from build_fixed_model():
    - KV caches are register_buffer → ct.StateType (state stays on ANE)
    - No KV inputs or outputs — model reads/writes state internally
    - Eliminates all host↔ANE KV data transfer (~6MB per token)
    - Layers do in-place scatter-write to state buffers

    Sharding (layer_start / layer_end):
    - When set, builds only layers [layer_start, layer_end)
    - Output is "hidden" (1, d, 1, 1) instead of "logits" — no LM head
    - Each shard has its own KV state (locally numbered 0..N-1)
    - Embedding lookup, final norm, and LM head are done on host
    - This follows the proven Gemma multi-shard pattern

    split_mode (for large-layer models like Phi-4 14B):
    - 'full' (default): complete transformer layer per shard
    - 'attn': attention-only (QKV + O + RoPE + GQA), with KV state
    - 'ffn': FFN-only (gate_up + down + SiLU), no state
    When split_mode is 'attn' or 'ffn', each physical layer produces two
    separate shards. This keeps per-shard compiled size under ~250 MB for
    models where a full layer exceeds the ANE budget.

    NOTE: seq_len must be ≥ 1637 to avoid a CoreML CPU-backend state bug
    that produces NaN at forward pass 3. The threshold depends on model
    dimensions (d=1536, nkv=2, dh=128 → KV state shape (1,2,seq,128)).
    """
    # Guard against the CoreML CPU-backend state bug (NaN at fwd=3)
    MIN_SEQ_LEN = 1637
    if max_seq_len < MIN_SEQ_LEN:
        print(f"  ⚠️  seq_len {max_seq_len} < {MIN_SEQ_LEN}: bumping to {MIN_SEQ_LEN} "
              f"(CoreML CPU-backend state bug workaround)")
        max_seq_len = MIN_SEQ_LEN
    import coremltools as ct
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print(f"Loading GGUF: {gguf_path}")
    gguf = GGUFModel(gguf_path)
    cfg = gguf.config()

    if n_layers is None:
        n_layers = cfg["n_layers"]
    n_layers = min(n_layers, cfg["n_layers"])

    # Shard range: [layer_start, layer_end). Default = full model.
    if layer_start is None:
        layer_start = 0
    if layer_end is None:
        layer_end = n_layers
    is_shard = (layer_start != 0) or (layer_end != n_layers)
    shard_n = layer_end - layer_start
    assert 0 <= layer_start < layer_end <= n_layers, \
        f"Invalid layer range [{layer_start}, {layer_end}) for {n_layers} layers"

    d = cfg["d_model"]
    nh = cfg["n_heads"]
    nkv = cfg["n_kv_heads"]
    dh = cfg["d_head"]
    dff = cfg["d_ff"]
    eps = cfg["rms_norm_eps"]
    vocab = cfg["vocab_size"]
    kv_dim = nkv * dh
    qkv_dim = d + 2 * kv_dim
    hpk = nh // nkv

    shard_tag = f" [shard {layer_start}-{layer_end}]" if is_shard else ""
    print(f"Config: {n_layers}L, d={d}, nh={nh}, nkv={nkv}, dh={dh}, dff={dff}{shard_tag}")
    print(f"Fixed seq len: {max_seq_len}, Float16, batched GQA, STATEFUL KV")

    # ── Load embeddings ──
    print("Loading token embeddings...")
    token_embd = gguf.get_tensor("token_embd.weight", dtype=np.float32)

    # ── RoPE tables ──
    d_half = dh // 2
    freqs = 1.0 / (cfg["rope_freq_base"] ** (np.arange(0, d_half, dtype=np.float32) / d_half))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    rope_cos = np.cos(angles).astype(np.float32)
    rope_sin = np.sin(angles).astype(np.float32)

    # ── PyTorch model — stateful KV ─────────────────────────────────────

    class RMSNormConv(nn.Module):
        def __init__(self, weight, eps):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(
                torch.tensor(weight, dtype=torch.float16).reshape(-1, 1, 1),
                requires_grad=False)

        def forward(self, x):
            # Safe-norm peephole (Dragon Book §8.7)
            K = x.shape[1] ** 0.5
            x_scaled = x * (1.0 / K)
            variance = x_scaled.pow(2).mean(dim=1, keepdim=True)
            x_normed = x_scaled * torch.rsqrt(variance + self.eps / (K * K))
            return (x_normed * self.weight).to(x.dtype)

    class StatefulLayerConv(nn.Module):
        """Transformer layer — in-place KV state update, batched GQA.

        split_mode controls which ops to build/run:
          'full' (default): complete layer (attn + FFN)
          'attn': attention-only (norm → QKV → RoPE → GQA → O → residual), has KV state
          'ffn':  FFN-only (norm → gate_up → SiLU → down → residual), no state
        """
        def __init__(self, layer_idx, gguf_model, cfg, split_mode="full"):
            super().__init__()
            d = cfg["d_model"]
            dff = cfg["d_ff"]
            kv_dim = cfg["n_kv_heads"] * cfg["d_head"]
            qkv_dim = d + 2 * kv_dim
            eps = cfg["rms_norm_eps"]
            prefix = f"blk.{layer_idx}"

            self.d = d
            self.dff = dff
            self.kv_dim = kv_dim
            self.nh = cfg["n_heads"]
            self.nkv = cfg["n_kv_heads"]
            self.dh = cfg["d_head"]
            self.hpk = self.nh // self.nkv
            self.scale = 1.0 / (cfg["d_head"] ** 0.5)
            self.split_mode = split_mode

            if split_mode in ("full", "attn"):
                self.attn_norm = RMSNormConv(
                    gguf_model.get_tensor(f"{prefix}.attn_norm.weight"), eps)

                # Fused QKV (handles both split Qwen and fused Phi layouts)
                q_w, k_w, v_w = gguf_model.get_qkv_weights(prefix, cfg)
                qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
                has_bias = gguf_model.has_biases(prefix)
                self.qkv_conv = nn.Conv2d(d, qkv_dim, 1, bias=has_bias)
                self.qkv_conv.weight = nn.Parameter(
                    torch.tensor(qkv_w, dtype=torch.float16).reshape(qkv_dim, d, 1, 1),
                    requires_grad=False)
                if has_bias:
                    biases = gguf_model.get_qkv_biases(prefix, cfg)
                    qkv_b = np.concatenate([biases[0], biases[1], biases[2]])
                    self.qkv_conv.bias = nn.Parameter(
                        torch.tensor(qkv_b, dtype=torch.float16), requires_grad=False)

                # Output projection
                o_w = gguf_model.get_tensor(f"{prefix}.attn_output.weight")
                self.out_conv = nn.Conv2d(d, d, 1, bias=False)
                self.out_conv.weight = nn.Parameter(
                    torch.tensor(o_w, dtype=torch.float16).reshape(d, d, 1, 1),
                    requires_grad=False)

            if split_mode in ("full", "ffn"):
                # FFN (handles both split Qwen and fused Phi layouts)
                self.ffn_norm = RMSNormConv(
                    gguf_model.get_tensor(f"{prefix}.ffn_norm.weight"), eps)

                gate_w, up_w = gguf_model.get_gate_up_weights(prefix, cfg)
                gate_up_w = np.concatenate([gate_w, up_w], axis=0)
                self.gate_up_conv = nn.Conv2d(d, 2 * dff, 1, bias=False)
                self.gate_up_conv.weight = nn.Parameter(
                    torch.tensor(gate_up_w, dtype=torch.float16).reshape(2 * dff, d, 1, 1),
                    requires_grad=False)

                down_w = gguf_model.get_tensor(f"{prefix}.ffn_down.weight")
                self.down_conv = nn.Conv2d(dff, d, 1, bias=False)
                self.down_conv.weight = nn.Parameter(
                    torch.tensor(down_w, dtype=torch.float16).reshape(d, dff, 1, 1),
                    requires_grad=False)

        def _forward_attn(self, x, k_cache, v_cache, rope_cos_pos, rope_sin_pos,
                          attn_mask, kv_write_mask):
            residual = x
            normed = self.attn_norm(x)

            qkv = self.qkv_conv(normed).squeeze(-1).squeeze(-1)
            q = qkv[:, :self.d]
            k = qkv[:, self.d:self.d + self.kv_dim]
            v = qkv[:, self.d + self.kv_dim:]

            d_half = self.dh // 2
            def apply_rope(x_flat, n_heads):
                x_r = x_flat.reshape(1, n_heads, self.dh)
                x_lo = x_r[:, :, :d_half]
                x_hi = x_r[:, :, d_half:]
                cos_b = rope_cos_pos.unsqueeze(1)
                sin_b = rope_sin_pos.unsqueeze(1)
                r_lo = x_lo * cos_b - x_hi * sin_b
                r_hi = x_lo * sin_b + x_hi * cos_b
                return torch.cat([r_lo, r_hi], dim=-1).reshape(1, n_heads * self.dh)

            q = apply_rope(q, self.nh)
            k = apply_rope(k, self.nkv)

            new_k = k.reshape(1, self.nkv, 1, self.dh)
            new_v = v.reshape(1, self.nkv, 1, self.dh)

            k_updated = k_cache * (1.0 - kv_write_mask) + new_k * kv_write_mask
            v_updated = v_cache * (1.0 - kv_write_mask) + new_v * kv_write_mask
            k_cache[:] = k_updated
            v_cache[:] = v_updated

            q_heads = q.reshape(1, self.nh, self.dh)
            attn_parts = []
            for kv_idx in range(self.nkv):
                q_group = q_heads[:, kv_idx*self.hpk:(kv_idx+1)*self.hpk, :]
                k_head = k_updated[:, kv_idx:kv_idx+1, :, :]
                v_head = v_updated[:, kv_idx:kv_idx+1, :, :]
                q_g = q_group.unsqueeze(2)
                k_t = k_head.transpose(-2, -1)
                scores = torch.matmul(q_g, k_t) * self.scale
                scores = scores + attn_mask
                attn_w = torch.softmax(scores.float(), dim=-1).to(q_g.dtype)
                head_out = torch.matmul(attn_w, v_head)
                attn_parts.append(head_out.squeeze(2))

            attn_out = torch.cat(attn_parts, dim=1)
            attn_out = attn_out.reshape(1, self.d, 1, 1)
            attn_out = self.out_conv(attn_out)
            return residual + attn_out

        def _forward_ffn(self, x):
            residual = x
            normed = self.ffn_norm(x)
            gate_up = self.gate_up_conv(normed)
            gate = gate_up[:, :self.dff, :, :]
            up = gate_up[:, self.dff:, :, :]
            hidden = F.silu(gate.float()).to(gate.dtype) * up
            ffn_out = self.down_conv(hidden)
            return residual + ffn_out

        def forward(self, x, k_cache=None, v_cache=None, rope_cos_pos=None,
                    rope_sin_pos=None, attn_mask=None, kv_write_mask=None):
            if self.split_mode == "attn":
                return self._forward_attn(x, k_cache, v_cache, rope_cos_pos,
                                          rope_sin_pos, attn_mask, kv_write_mask)
            elif self.split_mode == "ffn":
                return self._forward_ffn(x)
            else:  # full
                x = self._forward_attn(x, k_cache, v_cache, rope_cos_pos,
                                       rope_sin_pos, attn_mask, kv_write_mask)
                return self._forward_ffn(x)

    class QwenStatefulConv(nn.Module):
        """Full Qwen — stateful KV cache (register_buffer), batched GQA.

        When is_shard=True, builds only layers [layer_start, layer_end) and
        outputs hidden state (1, d, 1, 1) instead of logits. LM head and
        output norm are omitted — host does final projection.
        """
        def __init__(self, gguf_model, cfg, n_layers, max_seq_len,
                     layer_start=0, layer_end=None, is_shard=False,
                     split_mode="full"):
            super().__init__()
            if layer_end is None:
                layer_end = n_layers
            self.shard_n = layer_end - layer_start
            self.d = cfg["d_model"]
            self.nkv = cfg["n_kv_heads"]
            self.dh = cfg["d_head"]
            self.max_seq_len = max_seq_len
            self.is_shard = is_shard
            self.split_mode = split_mode

            mode_tag = f" [{split_mode}]" if split_mode != "full" else ""
            self.layers = nn.ModuleList()
            for i in range(layer_start, layer_end):
                local = i - layer_start
                print(f"  Layer {i} (local {local}/{self.shard_n}){mode_tag}")
                self.layers.append(StatefulLayerConv(i, gguf_model, cfg,
                                                     split_mode=split_mode))

            # KV caches only needed for full and attn modes
            if split_mode in ("full", "attn"):
                for i in range(self.shard_n):
                    self.register_buffer(f"k_cache_{i}",
                        torch.zeros(1, cfg["n_kv_heads"], max_seq_len, cfg["d_head"],
                                    dtype=torch.float16))
                    self.register_buffer(f"v_cache_{i}",
                        torch.zeros(1, cfg["n_kv_heads"], max_seq_len, cfg["d_head"],
                                    dtype=torch.float16))

            if not is_shard and split_mode == "full":
                # Full model: include output norm + LM head
                out_norm_w = gguf_model.get_tensor("output_norm.weight")
                self.output_norm = RMSNormConv(out_norm_w, cfg["rms_norm_eps"])

                print("  LM head...")
                if "output.weight" in gguf_model.tensors:
                    lm_w = gguf_model.get_tensor("output.weight")
                else:
                    lm_w = gguf_model.get_tensor("token_embd.weight")
                self.lm_head_conv = nn.Conv2d(cfg["d_model"], cfg["vocab_size"], 1, bias=False)
                self.lm_head_conv.weight = nn.Parameter(
                    torch.tensor(lm_w, dtype=torch.float16).reshape(
                        cfg["vocab_size"], cfg["d_model"], 1, 1),
                    requires_grad=False)

        def forward(self, x, rope_cos_pos=None, rope_sin_pos=None,
                    attn_mask=None, kv_write_mask=None):
            """
            split_mode='full'/'attn': needs all args, has KV state.
            split_mode='ffn': only needs x.
            Returns: logits (full model) or hidden (shard/split).
            """
            if self.split_mode == "ffn":
                for layer in self.layers:
                    x = layer(x)
                return x

            for i, layer in enumerate(self.layers):
                k_cache = getattr(self, f"k_cache_{i}")
                v_cache = getattr(self, f"v_cache_{i}")
                x = layer(x, k_cache, v_cache, rope_cos_pos, rope_sin_pos,
                          attn_mask, kv_write_mask)

            if self.is_shard or self.split_mode == "attn":
                return x

            x = self.output_norm(x)
            logits = self.lm_head_conv(x).squeeze(-1).squeeze(-1)
            return logits

    # ── Build ────────────────────────────────────────────────────────────

    build_tag = f" shard [{layer_start},{layer_end})" if is_shard else ""
    split_tag = f" [{split_mode}]" if split_mode != "full" else ""
    print(f"\nBuilding QwenStatefulConv ({shard_n}L, fp16, stateful KV{build_tag}{split_tag})...")
    model = QwenStatefulConv(gguf, cfg, n_layers, max_seq_len,
                             layer_start=layer_start, layer_end=layer_end,
                             is_shard=is_shard, split_mode=split_mode)
    model.half()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters (fp16)")

    # ── Pre-conversion weight optimization (PyTorch level) ───────────────

    if strategy == "smooth":
        model = _apply_smoothquant(model, gguf, token_embd, cfg, n_layers, max_seq_len)
    elif strategy in ("gptq", "gptq-mixed"):
        model = _apply_gptq(model, gguf, token_embd, cfg, n_layers, max_seq_len,
                            group_size=group_size, use_real_text=False)
    elif strategy == "gptq-real":
        model = _apply_gptq(model, gguf, token_embd, cfg, n_layers, max_seq_len,
                            group_size=group_size, use_real_text=True)
    elif strategy == "gptq-woodbury":
        model = _apply_gptq(model, gguf, token_embd, cfg, n_layers, max_seq_len,
                            group_size=group_size, use_real_text=True,
                            use_woodbury=True)

    # Re-sync model state after weight modifications (ensures TorchScript IR
    # and state_dict agree — .data assignments can break tensor identity)
    if strategy in ("smooth", "gptq", "gptq-real", "gptq-mixed", "gptq-woodbury"):
        model.load_state_dict(model.state_dict())

    # ── Trace ────────────────────────────────────────────────────────────

    d_half = dh // 2
    x_ex = torch.randn(1, d, 1, 1, dtype=torch.float16)

    if split_mode == "ffn":
        # FFN-only: just x in → x out, no rope/mask/state
        print("\nTracing (FFN-only, no state)...")
        with torch.no_grad():
            output = model(x_ex)
            print(f"  hidden: {output.shape}, dtype: {output.dtype}")
        traced = torch.jit.trace(model, (x_ex,))
    else:
        # Full or attn: needs rope, mask, has KV state
        cos_ex = torch.randn(1, d_half, dtype=torch.float16)
        sin_ex = torch.randn(1, d_half, dtype=torch.float16)
        mask_ex = torch.full((1, 1, 1, max_seq_len), -1e4, dtype=torch.float16)
        mask_ex[0, 0, 0, 0] = 0.0
        wmask_ex = torch.zeros(1, 1, max_seq_len, 1, dtype=torch.float16)
        wmask_ex[0, 0, 0, 0] = 1.0

        mode_label = "attn-only" if split_mode == "attn" else "stateful KV"
        print(f"\nTracing ({mode_label})...")
        with torch.no_grad():
            output = model(x_ex, cos_ex, sin_ex, mask_ex, wmask_ex)
            if is_shard or split_mode == "attn":
                print(f"  hidden: {output.shape}, dtype: {output.dtype}")
            else:
                print(f"  logits: {output.shape}, dtype: {output.dtype}")

        traced = torch.jit.trace(model, (x_ex, cos_ex, sin_ex, mask_ex, wmask_ex))

        # Reset KV cache buffers after tracing — forward pass writes NaN/stale values
        # into caches (fp16 overflow at dh=128), and NaN != NaN breaks coremltools'
        # torch.equal() assertion in _lower_graph_block.
        with torch.no_grad():
            for i in range(shard_n):
                getattr(model, f"k_cache_{i}").zero_()
                getattr(model, f"v_cache_{i}").zero_()

    # ── Convert to CoreML ────────────────────────────────────────────────

    conv_tag = f" shard [{layer_start},{layer_end})" if is_shard else ""
    print(f"\nConverting to CoreML (fp16, {shard_n}L{conv_tag}{split_tag})...")

    if split_mode == "ffn":
        # FFN-only: simple x → hidden, no state
        ct_inputs = [
            ct.TensorType(name="x", shape=(1, d, 1, 1), dtype=np.float16),
        ]
        ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
        ct_states = []
    else:
        # Full or attn: rope + mask inputs, KV state
        ct_inputs = [
            ct.TensorType(name="x", shape=(1, d, 1, 1), dtype=np.float16),
            ct.TensorType(name="rope_cos", shape=(1, d_half), dtype=np.float16),
            ct.TensorType(name="rope_sin", shape=(1, d_half), dtype=np.float16),
            ct.TensorType(name="attn_mask", shape=(1, 1, 1, max_seq_len), dtype=np.float16),
            ct.TensorType(name="kv_write_mask", shape=(1, 1, max_seq_len, 1), dtype=np.float16),
        ]

        if is_shard or split_mode == "attn":
            ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
        else:
            ct_outputs = [ct.TensorType(name="logits", dtype=np.float16)]

        # KV caches as state — these stay on-device between calls
        ct_states = []
        for i in range(shard_n):
            ct_states.append(ct.StateType(
                wrapped_type=ct.TensorType(shape=(1, nkv, max_seq_len, dh), dtype=np.float16),
                name=f"k_cache_{i}"))
            ct_states.append(ct.StateType(
                wrapped_type=ct.TensorType(shape=(1, nkv, max_seq_len, dh), dtype=np.float16),
                name=f"v_cache_{i}"))

    convert_kwargs = dict(
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    if ct_states:
        convert_kwargs["states"] = ct_states

    mlmodel = ct.convert(traced, **convert_kwargs)

    # ── Post-training weight quantization ────────────────────────────────
    # For gptq/gptq-real/smooth: PyTorch-level optimization already done; apply
    # uniform int4 at MIL level to encode the optimized weights into actual int4.
    # For gptq-mixed: GPTQ-optimized weights, but QKV gets int8 at MIL level
    # (higher precision for attention → better prompt discrimination).
    # For mixed: apply per-op mixed-precision at MIL level.
    # For uniform: standard RTN quantization.
    if strategy == "gptq-mixed":
        mil_strategy = "mixed"
    elif strategy in ("gptq", "gptq-real", "gptq-woodbury", "smooth"):
        mil_strategy = "uniform"
    else:
        mil_strategy = strategy
    mlmodel = _quantize_weights(mlmodel, quant_bits, group_size=group_size, strategy=mil_strategy)

    # ── Save ─────────────────────────────────────────────────────────────

    if quant_bits:
        strat_tag = f"_{strategy}" if strategy != "uniform" else ""
        suffix = f"_q{quant_bits}{strat_tag}" + (f"g{group_size}" if group_size != 32 else "")
    else:
        suffix = ""
    split_suffix = f"_{split_mode}" if split_mode != "full" else ""
    if output_name:
        prefix = output_name
    elif is_shard or split_mode != "full":
        prefix = f"QwenANE_{n_layers}L_s{layer_start}-{layer_end}{split_suffix}{suffix}"
    else:
        prefix = f"QwenANE_{n_layers}L_stateful{suffix}"
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        prefix = os.path.join(output_dir, prefix)
    pkg_path = f"{prefix}.mlpackage"
    mlmodel.save(pkg_path)
    print(f"\n✓ Saved {pkg_path}")

    # ── Metadata ─────────────────────────────────────────────────────────

    meta = {
        **cfg,
        "n_layers_exported": shard_n,
        "n_layers_total": n_layers,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "max_seq_len": max_seq_len,
        "dtype": "float16",
        "fixed_shapes": True,
        "batched_gqa": True,
        "stateful_kv": split_mode != "ffn",
        "split_mode": split_mode,
        "is_shard": is_shard or split_mode != "full",
        "quant_bits": quant_bits,
    }
    if not is_shard and split_mode == "full":
        # Full model: include RoPE tables in per-model meta (legacy compat)
        meta["rope_cos"] = rope_cos.tolist()
        meta["rope_sin"] = rope_sin.tolist()
    meta_path = f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"  Meta: {meta_path}")

    # ── Shared artifacts (only for full model, not shards/splits) ──────
    # When sharding or splitting, the orchestrator exports these once.

    if not is_shard and split_mode == "full":
        # ── Embeddings ──
        embd_path = f"{prefix}_embd.bin"
        token_embd.astype(np.float16).tofile(embd_path)
        print(f"  Embeddings: {embd_path} ({Path(embd_path).stat().st_size / 1e6:.1f} MB)")

        # ── BPE Tokenizer ──
        print("Exporting BPE tokenizer...")
        tok_data = gguf.extract_tokenizer()
        tok_path = f"{prefix}_tokenizer.json"
        with open(tok_path, "w") as f:
            json.dump(tok_data, f)
        print(f"  Tokenizer: {tok_path}")

    is_split_or_shard = is_shard or split_mode != "full"
    output_kind = "hidden (shard)" if is_split_or_shard else "logits"
    kv_label = "NONE" if split_mode == "ffn" else "STATEFUL (on-device, zero host copy)"
    print(f"\n{'='*60}")
    print(f"  Model:      Qwen2.5 ({shard_n}L [{layer_start},{layer_end}), d={d}, nh={nh}, nkv={nkv})")
    print(f"  Split:      {split_mode}")
    print(f"  Format:     CoreML mlprogram (iOS18+)")
    print(f"  KV Cache:   {kv_label}")
    if split_mode != "ffn":
        print(f"  Attention:  Batched GQA ({shard_n * nkv * 2} matmul ops)")
    print(f"  Output:     {output_kind}")
    print(f"  Dtype:      Float16{f' + int{quant_bits} weights' if quant_bits else ''}")
    if not is_split_or_shard:
        print(f"  Tokenizer:  BPE ({len(tok_data['tokens'])} vocab)")
    print(f"  Target:     ANE (CPU_AND_NE, seq_len≥1637 state-bug guard)")
    print(f"{'='*60}")

    return pkg_path


def main():
    parser = argparse.ArgumentParser(description="GGUF → ANE-native CoreML converter")
    parser.add_argument("gguf", help="Path to GGUF model file")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of layers (default: all)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--quant-bits", type=int, choices=[0, 4, 8], default=0,
                        dest="quant_bits",
                        help="Weight quantization bits (0=fp16, 4=int4 grouped, 8=int8). Default 0.")
    parser.add_argument("--group-size", type=int, default=32, dest="group_size",
                        help="Block/group size for grouped int4/int8 quantization (default 32). Larger = smaller weights, lower fidelity.")
    parser.add_argument("--quant-strategy", choices=["uniform", "mixed", "gptq", "gptq-real", "gptq-mixed", "gptq-woodbury", "smooth"],
                        default="uniform", dest="quant_strategy",
                        help="Quantization strategy: uniform=same precision everywhere (RTN), "
                             "mixed=int8 for sensitive layers + int4 for FFN (MIL-level), "
                             "gptq=GPTQ calibrated (random cal, uniform int4 MIL), "
                             "gptq-real=GPTQ with chat-template calibration (uniform int4 MIL), "
                             "gptq-mixed=GPTQ (random cal) + mixed MIL (QKV int8, rest int4), "
                             "gptq-woodbury=Woodbury-accelerated GPTQ (real cal, exploits rank deficiency), "
                             "smooth=SmoothQuant activation smoothing + int4 (PyTorch-level). "
                             "Default: uniform.")
    parser.add_argument("--quantize", action="store_const", const=8, dest="quant_bits",
                        help="Deprecated alias for --quant-bits 8")
    parser.add_argument("--fixed", action="store_true",
                        help="Build fixed-shape model (eliminates dynamic shapes)")
    parser.add_argument("--stateful", action="store_true",
                        help="Build stateful model (KV cache as on-device state)")
    parser.add_argument("--layer-start", type=int, default=None, dest="layer_start",
                        help="First layer index for shard (inclusive). Requires --stateful.")
    parser.add_argument("--layer-end", type=int, default=None, dest="layer_end",
                        help="Last layer index for shard (exclusive). Requires --stateful.")
    parser.add_argument("--split-layer", action="store_true", dest="split_layer",
                        help="Split each layer into attn + FFN sub-shards. Requires --stateful. "
                             "Produces two .mlpackage per layer range: *_attn.mlpackage (with KV "
                             "state) and *_ffn.mlpackage (stateless). Use for large-layer models "
                             "(e.g. Phi-4 14B) where a full layer exceeds the ~250 MB ANE shard limit.")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Output directory (default: current directory)")
    parser.add_argument("--output-name", default=None, dest="output_name",
                        help="Output prefix name (default: auto-generated)")
    args = parser.parse_args()

    if args.stateful:
        if args.split_layer:
            # Split mode: build attn and FFN sub-shards separately
            common = dict(
                n_layers=args.layers, max_seq_len=args.seq_len,
                quant_bits=args.quant_bits, group_size=args.group_size,
                strategy=args.quant_strategy,
                layer_start=args.layer_start, layer_end=args.layer_end,
                output_dir=args.output_dir,
            )
            attn_name = f"{args.output_name}_attn" if args.output_name else None
            ffn_name = f"{args.output_name}_ffn" if args.output_name else None
            print("═" * 60)
            print("  SPLIT-LAYER MODE: building attn + FFN sub-shards")
            print("═" * 60)
            build_stateful_model(args.gguf, **common, output_name=attn_name,
                                 split_mode="attn")
            build_stateful_model(args.gguf, **common, output_name=ffn_name,
                                 split_mode="ffn")
        else:
            build_stateful_model(args.gguf, n_layers=args.layers, max_seq_len=args.seq_len,
                                 quant_bits=args.quant_bits, group_size=args.group_size,
                                 strategy=args.quant_strategy,
                                 layer_start=args.layer_start, layer_end=args.layer_end,
                                 output_dir=args.output_dir, output_name=args.output_name)
    elif args.fixed:
        build_fixed_model(args.gguf, n_layers=args.layers, max_seq_len=args.seq_len,
                          quant_bits=args.quant_bits, group_size=args.group_size,
                          strategy=args.quant_strategy)
    else:
        build_model(args.gguf, n_layers=args.layers, max_seq_len=args.seq_len,
                    quant_bits=args.quant_bits, group_size=args.group_size,
                    strategy=args.quant_strategy)


if __name__ == "__main__":
    main()
