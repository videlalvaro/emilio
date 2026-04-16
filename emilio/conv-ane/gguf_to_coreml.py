#!/usr/bin/env python3
"""GGUF → CoreML Conv-Only Converter for Qwen2.5-0.5B.

Every linear projection becomes a Conv2d(1×1). All computation runs through
convolution on the Apple Neural Engine.

Architecture: Qwen2 with GQA
  - 24 layers, d_model=896, n_heads=14, n_kv_heads=2, d_ff=4864
  - All matmuls as 1×1 convolutions
  - RMSNorm as depthwise conv
  - SiLU as ANE-native activation
  - RoPE via precomputed depthwise conv weights
  - Attention via batched matmul (1×1 conv reinterpretation)

Usage: python3 gguf_to_coreml.py <model.gguf> [--layers N] [--seq-len S]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# ─── Minimal GGUF Parser (Q8_0 + F32 + F16) ────────────────────────────────

GGUF_MAGIC = 0x46554747

GGML_TYPES = {
    0: "F32", 1: "F16", 8: "Q8_0",
}

META_TYPES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16",
    4: "uint32", 5: "int32", 6: "float32", 7: "bool",
    8: "string", 9: "array", 10: "uint64", 11: "int64",
    12: "float64",
}


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
    if vtype == 4:  # uint32
        return read_u32(f)
    elif vtype == 5:  # int32
        return read_i32(f)
    elif vtype == 6:  # float32
        return read_f32(f)
    elif vtype == 7:  # bool
        return bool(f.read(1)[0])
    elif vtype == 8:  # string
        return read_string(f)
    elif vtype == 10:  # uint64
        return read_u64(f)
    elif vtype == 9:  # array
        elem_type = read_u32(f)
        count = read_u64(f)
        return [read_meta_value(f, elem_type) for _ in range(count)]
    elif vtype == 0:  # uint8
        return f.read(1)[0]
    elif vtype == 12:  # float64
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown meta type {vtype}")


def dequant_q8_0(data, n_elements):
    """Dequantize Q8_0: blocks of 34 bytes (1 f16 scale + 32 int8 quants)."""
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.empty(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        offset = i * 34  # 2 bytes scale + 32 bytes quants
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16).astype(np.float32)[0]
        quants = np.frombuffer(data[offset + 2:offset + 34], dtype=np.int8).astype(np.float32)
        result[i * block_size:(i + 1) * block_size] = quants * scale

    return result


class GGUFModel:
    """Minimal GGUF reader for Qwen2.5-0.5B weights."""

    def __init__(self, path):
        self.path = path
        self.metadata = {}
        self.tensors = {}  # name -> (shape, type, offset)
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

            # Read metadata
            for _ in range(n_meta):
                key = read_string(f)
                vtype = read_u32(f)
                value = read_meta_value(f, vtype)
                self.metadata[key] = value

            # Read tensor infos
            for _ in range(n_tensors):
                name = read_string(f)
                n_dims = read_u32(f)
                dims = [read_u64(f) for _ in range(n_dims)]
                dtype = read_u32(f)
                offset = read_u64(f)
                self.tensors[name] = {
                    "shape": dims,
                    "type": dtype,
                    "offset": offset,
                    "n_elements": int(np.prod(dims)),
                }

            # Data starts at alignment boundary after header
            alignment = self.metadata.get("general.alignment", 32)
            pos = f.tell()
            self._data_offset = pos + (alignment - pos % alignment) % alignment

    def meta(self, key, default=None):
        return self.metadata.get(key, default)

    def get_tensor(self, name):
        """Load and dequantize a tensor by name.

        GGUF stores dimensions in GGML order: ne[0] is the innermost (fastest
        varying) dimension, ne[1] is the next, etc.  For a 2-D weight matrix
        with shape (out_features, in_features) in PyTorch convention, GGML
        stores ne = [in_features, out_features].  We reverse the stored dims
        so the returned array has standard (rows, cols) = (out, in) layout.
        """
        info = self.tensors[name]
        n_elements = info["n_elements"]
        dtype = info["type"]

        with open(self.path, "rb") as f:
            f.seek(self._data_offset + info["offset"])

            if dtype == 0:  # F32
                data = f.read(n_elements * 4)
                arr = np.frombuffer(data, dtype=np.float32).copy()
            elif dtype == 1:  # F16
                data = f.read(n_elements * 2)
                arr = np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()
            elif dtype == 8:  # Q8_0
                n_blocks = n_elements // 32
                data = f.read(n_blocks * 34)
                arr = dequant_q8_0(data, n_elements)
            else:
                raise ValueError(f"Unsupported type {dtype} for {name}")

        # Reverse GGML dim order → standard (row-major) shape
        shape = list(reversed(info["shape"]))
        return arr.reshape(shape)

    def config(self):
        """Extract Qwen2 config from metadata."""
        arch = self.meta("general.architecture", "qwen2")
        d_model = self.meta(f"{arch}.embedding_length", 896)
        n_heads = self.meta(f"{arch}.attention.head_count", 14)
        return {
            "arch": arch,
            "vocab_size": self.meta(f"{arch}.vocab_size",
                          self.meta("tokenizer.ggml.vocab_size", 151936)),
            "n_layers": self.meta(f"{arch}.block_count", 24),
            "n_heads": n_heads,
            "n_kv_heads": self.meta(f"{arch}.attention.head_count_kv", 2),
            "d_model": d_model,
            "d_head": d_model // n_heads,
            "d_ff": self.meta(f"{arch}.feed_forward_length", 4864),
            "rms_norm_eps": self.meta(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6),
            "rope_freq_base": self.meta(f"{arch}.rope.freq_base", 1000000.0),
        }


# ─── CoreML Model Builder (Conv-Only) ──────────────────────────────────────

def build_coreml_model(gguf_path, n_layers=None, max_seq_len=512):
    """Build a conv-only CoreML model from GGUF weights.

    All linear projections become Conv2d(1×1) operations.
    The model processes one token at a time; the host manages the KV cache
    and autoregressive loop.
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil import register_torch_op  # noqa: F401

    print(f"Loading GGUF: {gguf_path}")
    gguf = GGUFModel(gguf_path)
    cfg = gguf.config()

    if n_layers is None:
        n_layers = cfg["n_layers"]
    n_layers = min(n_layers, cfg["n_layers"])

    d = cfg["d_model"]       # 896
    nh = cfg["n_heads"]      # 14
    nkv = cfg["n_kv_heads"]  # 2
    dh = cfg["d_head"]       # 64
    dff = cfg["d_ff"]        # 4864
    eps = cfg["rms_norm_eps"]
    vocab = cfg["vocab_size"]
    kv_dim = nkv * dh        # 128
    qkv_dim = d + 2 * kv_dim  # 1152

    print(f"Config: {n_layers} layers, d={d}, nh={nh}, nkv={nkv}, dh={dh}, dff={dff}")
    print(f"KV dim: {kv_dim}, QKV fused dim: {qkv_dim}")
    print(f"Max sequence length: {max_seq_len}")

    # Load embedding table (kept as-is — discrete lookup, not a conv)
    print("Loading token embeddings...")
    token_embd = gguf.get_tensor("token_embd.weight")  # (vocab, d)
    print(f"  token_embd: {token_embd.shape}")

    # Precompute RoPE tables
    print("Precomputing RoPE tables...")
    d_half = dh // 2  # 32
    freqs = 1.0 / (cfg["rope_freq_base"] ** (np.arange(0, d_half, dtype=np.float32) / d_half))
    positions = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)  # (max_seq_len, 32)
    rope_cos = np.cos(angles).astype(np.float32)  # (max_seq_len, 32)
    rope_sin = np.sin(angles).astype(np.float32)  # (max_seq_len, 32)

    # ── Build MIL program ───────────────────────────────────────────────
    # Input: token embedding vector (already looked up by the host)
    # The host does: x = token_embd[token_id]
    # We receive x as (1, d, 1, 1) — "one pixel, d channels"
    #
    # KV cache inputs: per-layer K and V tensors from previous steps
    # Shape: (1, kv_dim, 1, seq_len) — channels=kv_dim, spatial=seq_len

    print(f"\nBuilding MIL program ({n_layers} layers, conv-only)...")

    # We'll use coremltools' torch frontend for convenience:
    # build the model in PyTorch with Conv2d layers, then convert.
    import torch
    import torch.nn as nn

    class RMSNormConv(nn.Module):
        """RMSNorm implemented via depthwise operations (ANE-friendly)."""
        def __init__(self, weight, eps):
            super().__init__()
            self.eps = eps
            # Store weight as a (d,1,1) shaped parameter for channel-wise multiply
            self.weight = nn.Parameter(
                torch.tensor(weight, dtype=torch.float32).reshape(-1, 1, 1),
                requires_grad=False)

        def forward(self, x):
            # x: (1, d, 1, 1)
            # RMSNorm: x * w / sqrt(mean(x²) + eps)
            variance = x.pow(2).mean(dim=1, keepdim=True)  # (1,1,1,1)
            x_normed = x * torch.rsqrt(variance + self.eps)
            return x_normed * self.weight

    class ConvOnlyQwenLayer(nn.Module):
        """One transformer layer — all matmuls are Conv2d(1×1)."""
        def __init__(self, layer_idx, gguf_model, cfg):
            super().__init__()
            d = cfg["d_model"]
            dff = cfg["d_ff"]
            kv_dim = cfg["n_kv_heads"] * cfg["d_head"]
            qkv_dim = d + 2 * kv_dim
            eps = cfg["rms_norm_eps"]

            prefix = f"blk.{layer_idx}"

            # ── Attention RMSNorm ──
            attn_norm_w = gguf_model.get_tensor(f"{prefix}.attn_norm.weight")
            self.attn_norm = RMSNormConv(attn_norm_w, eps)

            # ── Fused QKV projection: Conv2d(d, qkv_dim, 1×1) ──
            # GGUF stores W as (out_features, in_features)
            q_w = gguf_model.get_tensor(f"{prefix}.attn_q.weight")    # (d, d)
            k_w = gguf_model.get_tensor(f"{prefix}.attn_k.weight")    # (kv_dim, d)
            v_w = gguf_model.get_tensor(f"{prefix}.attn_v.weight")    # (kv_dim, d)
            qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)          # (qkv_dim, d)
            # Conv2d weight: (out_ch, in_ch, 1, 1)
            self.qkv_conv = nn.Conv2d(d, qkv_dim, 1, bias=True)
            self.qkv_conv.weight = nn.Parameter(
                torch.tensor(qkv_w, dtype=torch.float32).reshape(qkv_dim, d, 1, 1),
                requires_grad=False)
            # Fused bias
            q_b = gguf_model.get_tensor(f"{prefix}.attn_q.bias")     # (d,)
            k_b = gguf_model.get_tensor(f"{prefix}.attn_k.bias")     # (kv_dim,)
            v_b = gguf_model.get_tensor(f"{prefix}.attn_v.bias")     # (kv_dim,)
            qkv_b = np.concatenate([q_b, k_b, v_b])                  # (qkv_dim,)
            self.qkv_conv.bias = nn.Parameter(
                torch.tensor(qkv_b, dtype=torch.float32),
                requires_grad=False)

            # ── Output projection: Conv2d(d, d, 1×1) ──
            o_w = gguf_model.get_tensor(f"{prefix}.attn_output.weight")  # (d, d)
            self.out_conv = nn.Conv2d(d, d, 1, bias=False)
            self.out_conv.weight = nn.Parameter(
                torch.tensor(o_w, dtype=torch.float32).reshape(d, d, 1, 1),
                requires_grad=False)

            # ── FFN RMSNorm ──
            ffn_norm_w = gguf_model.get_tensor(f"{prefix}.ffn_norm.weight")
            self.ffn_norm = RMSNormConv(ffn_norm_w, eps)

            # ── Fused Gate+Up: Conv2d(d, 2*dff, 1×1) ──
            gate_w = gguf_model.get_tensor(f"{prefix}.ffn_gate.weight")  # (dff, d)
            up_w = gguf_model.get_tensor(f"{prefix}.ffn_up.weight")      # (dff, d)
            gate_up_w = np.concatenate([gate_w, up_w], axis=0)           # (2*dff, d)
            self.gate_up_conv = nn.Conv2d(d, 2 * dff, 1, bias=False)
            self.gate_up_conv.weight = nn.Parameter(
                torch.tensor(gate_up_w, dtype=torch.float32).reshape(2 * dff, d, 1, 1),
                requires_grad=False)

            # ── Down projection: Conv2d(dff, d, 1×1) ──
            down_w = gguf_model.get_tensor(f"{prefix}.ffn_down.weight")  # (d, dff)
            self.down_conv = nn.Conv2d(dff, d, 1, bias=False)
            self.down_conv.weight = nn.Parameter(
                torch.tensor(down_w, dtype=torch.float32).reshape(d, dff, 1, 1),
                requires_grad=False)

            self.d = d
            self.dff = dff
            self.kv_dim = kv_dim
            self.nh = cfg["n_heads"]
            self.nkv = cfg["n_kv_heads"]
            self.dh = cfg["d_head"]
            self.scale = 1.0 / (cfg["d_head"] ** 0.5)

        def forward(self, x, k_cache, v_cache, rope_cos_pos, rope_sin_pos):
            """
            x:         (1, d, 1, 1) — current token embedding as conv tensor
            k_cache:   (1, nkv, seq, dh) — K cache up to this point
            v_cache:   (1, nkv, seq, dh) — V cache up to this point
            rope_cos_pos: (1, d_half) — cos(θ) for current position
            rope_sin_pos: (1, d_half) — sin(θ) for current position

            Returns: (x_out, new_k, new_v) where new_k/new_v are the single
            new KV entries to append to the cache (host manages append).
            """
            residual = x

            # ── Pre-attention RMSNorm ──
            normed = self.attn_norm(x)  # (1, d, 1, 1)

            # ── Fused QKV projection (1×1 conv = matmul) ──
            qkv = self.qkv_conv(normed)  # (1, qkv_dim, 1, 1)
            qkv = qkv.squeeze(-1).squeeze(-1)  # (1, qkv_dim)

            q = qkv[:, :self.d]                          # (1, d)
            k = qkv[:, self.d:self.d + self.kv_dim]      # (1, kv_dim)
            v = qkv[:, self.d + self.kv_dim:]             # (1, kv_dim)

            # ── RoPE (applied per head, using precomputed cos/sin) ──
            d_half = self.dh // 2
            cos_t = rope_cos_pos  # (1, d_half)
            sin_t = rope_sin_pos  # (1, d_half)

            def apply_rope(x_flat, n_heads):
                # x_flat: (1, n_heads * dh)
                x_r = x_flat.reshape(1, n_heads, self.dh)
                x_lo = x_r[:, :, :d_half]   # first half
                x_hi = x_r[:, :, d_half:]   # second half
                cos_b = cos_t.unsqueeze(1)  # (1, 1, d_half)
                sin_b = sin_t.unsqueeze(1)  # (1, 1, d_half)
                r_lo = x_lo * cos_b - x_hi * sin_b
                r_hi = x_lo * sin_b + x_hi * cos_b
                return torch.cat([r_lo, r_hi], dim=-1).reshape(1, n_heads * self.dh)

            q = apply_rope(q, self.nh)    # (1, d)
            k = apply_rope(k, self.nkv)   # (1, kv_dim)

            # ── Prepare new K,V for cache return ──
            new_k = k.reshape(1, self.nkv, 1, self.dh)  # (1, nkv, 1, dh)
            new_v = v.reshape(1, self.nkv, 1, self.dh)  # (1, nkv, 1, dh)

            # ── Append to cache for this step's attention ──
            # k_cache: (1, nkv, seq, dh), new_k: (1, nkv, 1, dh)
            k_full = torch.cat([k_cache, new_k], dim=2)  # (1, nkv, seq+1, dh)
            v_full = torch.cat([v_cache, new_v], dim=2)  # (1, nkv, seq+1, dh)
            seq_len = k_full.shape[2]

            # ── GQA Attention ──
            heads_per_kv = self.nh // self.nkv  # 7

            q_heads = q.reshape(1, self.nh, self.dh)  # (1, 14, 64)
            attn_out_parts = []

            for kv_idx in range(self.nkv):
                k_head = k_full[:, kv_idx, :, :]  # (1, seq, dh)
                v_head = v_full[:, kv_idx, :, :]  # (1, seq, dh)

                for h_offset in range(heads_per_kv):
                    h = kv_idx * heads_per_kv + h_offset
                    q_h = q_heads[:, h:h+1, :]     # (1, 1, dh)

                    # Score: q_h @ k_head^T → (1, 1, seq)
                    # This IS a 1×1 conv if you squint: dot product across channel dim
                    scores = torch.matmul(q_h, k_head.transpose(-2, -1))  # (1, 1, seq)
                    scores = scores * self.scale

                    # Softmax over sequence dim
                    attn_w = torch.softmax(scores, dim=-1)  # (1, 1, seq)

                    # Weighted sum of V
                    head_out = torch.matmul(attn_w, v_head)  # (1, 1, dh)
                    attn_out_parts.append(head_out)

            attn_out = torch.cat(attn_out_parts, dim=-1)  # (1, 1, d)
            attn_out = attn_out.reshape(1, self.d, 1, 1)  # back to conv shape

            # ── Output projection (1×1 conv) ──
            attn_out = self.out_conv(attn_out)  # (1, d, 1, 1)

            # ── Residual ──
            x = residual + attn_out

            # ── FFN ──
            residual2 = x
            normed2 = self.ffn_norm(x)  # (1, d, 1, 1)

            # Gate+Up projection (1×1 conv)
            gate_up = self.gate_up_conv(normed2)  # (1, 2*dff, 1, 1)
            gate = gate_up[:, :self.dff, :, :]
            up = gate_up[:, self.dff:, :, :]

            # SiLU activation (ANE-native via sigmoid)
            hidden = torch.nn.functional.silu(gate) * up  # (1, dff, 1, 1)

            # Down projection (1×1 conv)
            ffn_out = self.down_conv(hidden)  # (1, d, 1, 1)

            # Residual
            x = residual2 + ffn_out

            return x, new_k, new_v

    class ConvOnlyQwen(nn.Module):
        """Full Qwen model with all matmuls as 1×1 convolutions."""
        def __init__(self, gguf_model, cfg, n_layers, max_seq_len):
            super().__init__()
            self.cfg = cfg
            self.n_layers = n_layers
            self.d = cfg["d_model"]
            self.nkv = cfg["n_kv_heads"]
            self.dh = cfg["d_head"]
            self.kv_dim = self.nkv * self.dh

            # Layers
            self.layers = nn.ModuleList()
            for i in range(n_layers):
                print(f"  Loading layer {i}/{n_layers}...")
                self.layers.append(ConvOnlyQwenLayer(i, gguf_model, cfg))

            # Final RMSNorm
            out_norm_w = gguf_model.get_tensor("output_norm.weight")
            self.output_norm = RMSNormConv(out_norm_w, cfg["rms_norm_eps"])

            # LM head: Conv2d(d, vocab, 1×1) — the biggest single conv
            print("  Loading LM head...")
            # Check if output.weight exists or if it shares token_embd
            if "output.weight" in gguf_model.tensors:
                lm_w = gguf_model.get_tensor("output.weight")  # (vocab, d)
            else:
                lm_w = gguf_model.get_tensor("token_embd.weight")  # tied
            self.lm_head_conv = nn.Conv2d(cfg["d_model"], cfg["vocab_size"], 1, bias=False)
            self.lm_head_conv.weight = nn.Parameter(
                torch.tensor(lm_w, dtype=torch.float32).reshape(cfg["vocab_size"], cfg["d_model"], 1, 1),
                requires_grad=False)

        def forward(self, x, k_caches, v_caches, rope_cos_pos, rope_sin_pos):
            """
            x:            (1, d, 1, 1) — token embedding
            k_caches:     list of (1, nkv, seq, dh) — one per layer
            v_caches:     list of (1, nkv, seq, dh) — one per layer
            rope_cos_pos: (1, d_half)
            rope_sin_pos: (1, d_half)

            Returns: (logits, new_ks, new_vs)
              logits:  (1, vocab)
              new_ks:  list of (1, nkv, 1, dh)
              new_vs:  list of (1, nkv, 1, dh)
            """
            new_ks = []
            new_vs = []

            for i, layer in enumerate(self.layers):
                x, nk, nv = layer(x, k_caches[i], v_caches[i], rope_cos_pos, rope_sin_pos)
                new_ks.append(nk)
                new_vs.append(nv)

            # Final norm + LM head (1×1 conv = matmul → logits)
            x = self.output_norm(x)       # (1, d, 1, 1)
            logits = self.lm_head_conv(x)  # (1, vocab, 1, 1)
            logits = logits.squeeze(-1).squeeze(-1)  # (1, vocab)

            return logits, new_ks, new_vs

    # ── Build the model ─────────────────────────────────────────────────

    print("\nConstructing ConvOnlyQwen model...")
    model = ConvOnlyQwen(gguf, cfg, n_layers, max_seq_len)
    model.eval()
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Trace with example inputs ────────────────────────────────────────

    d_half = dh // 2

    # Example inputs for tracing
    x_example = torch.randn(1, d, 1, 1)
    # Start with seq_len=1 for tracing (the host manages growing cache)
    # Use flexible shapes via coremltools
    init_seq = 1
    k_examples = [torch.randn(1, nkv, init_seq, dh) for _ in range(n_layers)]
    v_examples = [torch.randn(1, nkv, init_seq, dh) for _ in range(n_layers)]
    cos_example = torch.randn(1, d_half)
    sin_example = torch.randn(1, d_half)

    print("\nTracing model...")
    with torch.no_grad():
        # Test forward pass
        logits, nks, nvs = model(x_example, k_examples, v_examples, cos_example, sin_example)
        print(f"  Output logits shape: {logits.shape}")
        print(f"  New K shape: {nks[0].shape}")

    # ── Convert to CoreML ────────────────────────────────────────────────

    print("\nConverting to CoreML...")

    # Flatten inputs for tracing
    class FlatWrapper(nn.Module):
        """Flatten the list inputs for torch.jit.trace compatibility."""
        def __init__(self, model, n_layers):
            super().__init__()
            self.model = model
            self.n_layers = n_layers

        def forward(self, x, rope_cos, rope_sin, *kv_flat):
            # kv_flat: k0, v0, k1, v1, ..., k_{n-1}, v_{n-1}
            k_caches = [kv_flat[2*i] for i in range(self.n_layers)]
            v_caches = [kv_flat[2*i+1] for i in range(self.n_layers)]
            logits, new_ks, new_vs = self.model(x, k_caches, v_caches, rope_cos, rope_sin)
            # Flatten outputs too
            flat_out = [logits]
            for i in range(self.n_layers):
                flat_out.append(new_ks[i])
                flat_out.append(new_vs[i])
            return tuple(flat_out)

    wrapper = FlatWrapper(model, n_layers)
    wrapper.eval()

    trace_inputs = [x_example, cos_example, sin_example]
    for i in range(n_layers):
        trace_inputs.append(k_examples[i])
        trace_inputs.append(v_examples[i])

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, trace_inputs)

    # Build CoreML input specs with flexible sequence length
    ct_inputs = [
        ct.TensorType(name="x", shape=(1, d, 1, 1)),
        ct.TensorType(name="rope_cos", shape=(1, d_half)),
        ct.TensorType(name="rope_sin", shape=(1, d_half)),
    ]
    for i in range(n_layers):
        # Flexible seq dim for KV cache
        k_shape = ct.Shape(shape=(1, nkv, ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1), dh))
        v_shape = ct.Shape(shape=(1, nkv, ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1), dh))
        ct_inputs.append(ct.TensorType(name=f"k_cache_{i}", shape=k_shape))
        ct_inputs.append(ct.TensorType(name=f"v_cache_{i}", shape=v_shape))

    # Output specs
    ct_outputs = [ct.TensorType(name="logits")]
    for i in range(n_layers):
        ct_outputs.append(ct.TensorType(name=f"new_k_{i}"))
        ct_outputs.append(ct.TensorType(name=f"new_v_{i}"))

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    out_path = f"QwenConvOnly_{n_layers}L.mlpackage"
    mlmodel.save(out_path)
    print(f"\n✓ Saved {out_path}")
    print(f"  {n_layers} layers, all matmuls as Conv2d(1×1)")
    print(f"  Flexible KV cache: seq_len 1..{max_seq_len}")

    # Save metadata for the Swift host
    import json
    meta = {
        **cfg,
        "n_layers_exported": n_layers,
        "max_seq_len": max_seq_len,
        "rope_cos": rope_cos.tolist(),
        "rope_sin": rope_sin.tolist(),
        "token_embd_shape": list(token_embd.shape),
    }
    meta_path = f"QwenConvOnly_{n_layers}L_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"  Metadata: {meta_path}")

    # Save token embeddings separately (host does the lookup)
    embd_path = f"QwenConvOnly_{n_layers}L_embd.bin"
    token_embd.astype(np.float32).tofile(embd_path)
    print(f"  Embeddings: {embd_path} ({token_embd.nbytes / 1e6:.1f} MB)")

    return out_path


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GGUF → CoreML conv-only converter")
    parser.add_argument("gguf", help="Path to GGUF model file")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of layers to export (default: all)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    args = parser.parse_args()

    build_coreml_model(args.gguf, n_layers=args.layers, max_seq_len=args.seq_len)


if __name__ == "__main__":
    main()
