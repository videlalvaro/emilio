"""Run our Python Gemma implementation (fp16, unquantized) on BOS token.

Compares per-layer hidden states against HF golden to isolate:
- Our implementation vs HF: any structural bugs?
- bf16→fp16 precision loss (without INT8)

Usage:
    .venv/bin/python python/moe/gemma_ours_layer_trace.py
"""
from __future__ import annotations

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (
    GemmaSlidingLayer, GemmaGlobalLayer,
    _load_layer_weights, D_MODEL, N_PACKS, PACK_G,
    SLD_D_HEAD, SLD_N_KV, GLB_D_HEAD, GLB_N_KV, GLB_ROT_DIM,
)

MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT_DIR = Path("python/moe/out")
HF_HIDDEN = OUT_DIR / "gemma_hf_layer_hidden_bos.npz"

LAYER_TYPES = [
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
    "sliding", "sliding", "sliding", "sliding", "sliding", "global",
]

MAX_CTX = 1024


def _make_layer(layer_type: str) -> torch.nn.Module:
    if layer_type == "global":
        return GemmaGlobalLayer(MAX_CTX)
    return GemmaSlidingLayer(MAX_CTX)


def cos_sim(a, b):
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12))


def main():
    # Load HF reference
    hf = np.load(str(HF_HIDDEN))
    print(f"Loaded HF hidden states: {sorted(hf.files)[:5]}... ({len(hf.files)} total)")

    # Load embedding
    embed_path = OUT_DIR / "gemma_embed_fp16.bin"
    embed_data = np.fromfile(str(embed_path), dtype=np.float16)
    vocab_size = len(embed_data) // D_MODEL
    embed_table = embed_data.reshape(vocab_size, D_MODEL)
    print(f"Embedding: {vocab_size} × {D_MODEL}")

    # BOS token
    token_id = 2
    embed_scale = float(D_MODEL ** 0.5)
    x = torch.from_numpy(embed_table[token_id].astype(np.float32) * embed_scale).half()
    x = x.view(1, 1, D_MODEL)
    print(f"  embed x[0:4]={x[0,0,:4].tolist()} L2={torch.norm(x.float()).item():.5f}")

    hf_embed = hf["layer_0"]
    print(f"  HF embed x[0:4]={hf_embed[:4].tolist()} L2={np.linalg.norm(hf_embed):.5f}")
    print(f"  embed cos={cos_sim(x[0,0].float().numpy(), hf_embed):.6f}")

    # RoPE
    def make_rope(theta, dim, pos):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.tensor([pos], dtype=torch.float32)
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
        cos_v = torch.cos(angles).half().view(1, 1, dim // 2).repeat(1, 1, 2)[:, :, :dim]
        sin_v = torch.sin(angles).half().view(1, 1, dim // 2).repeat(1, 1, 2)[:, :, :dim]
        # Actually this interleaving is wrong. Let me use proper half-dim layout
        cos_v = torch.cos(angles).half()
        sin_v = torch.sin(angles).half()
        # Expand to full dim: [cos0, cos1, ..., cos_{d/2-1}] repeated
        cos_full = cos_v.repeat(1, 1, 2)[:, :, :dim]
        sin_full = sin_v.repeat(1, 1, 2)[:, :, :dim]
        return cos_full, sin_full

    # This is getting complex. Let me just run layer-by-layer with the full model.
    # Actually, let me use forward_attn + forward_ffn directly.

    attn_mask = torch.full((1, 1, 1, MAX_CTX), -10000.0, dtype=torch.float16)
    attn_mask[0, 0, 0, 0] = 0.0
    kv_write_mask = torch.zeros(1, 1, MAX_CTX, 1, dtype=torch.float16)
    kv_write_mask[0, 0, 0, 0] = 1.0

    print(f"\nRunning 30 layers (fp16 unquantized)...")
    for layer_idx in range(30):
        lt = LAYER_TYPES[layer_idx]
        layer = _make_layer(lt)
        layer.half().eval()
        npz_path = OUT_DIR / f"gemma4_layer{layer_idx}_packed.npz"
        if not npz_path.exists():
            print(f"  L{layer_idx}: SKIP (no packed npz)")
            break
        _load_layer_weights(layer, npz_path)
        layer.fuse_norm_scales_for_ane()

        # RoPE for this layer type
        if lt == "global":
            theta = 1000000.0
            rope_dim = GLB_ROT_DIM
        else:
            theta = 10000.0
            rope_dim = SLD_D_HEAD

        freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
        angles = torch.zeros(1, dtype=torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)  # pos=0
        cos_v = torch.cos(angles).half().view(1, 1, rope_dim // 2)
        sin_v = torch.sin(angles).half().view(1, 1, rope_dim // 2)
        # Duplicate for paired rotation: [c0, c1, ..., c_{d/2-1}, c0, c1, ...]
        cos_r = cos_v.repeat(1, 1, 2)[:, :, :rope_dim]
        sin_r = sin_v.repeat(1, 1, 2)[:, :, :rope_dim]

        # Set up KV cache
        if lt == "global":
            kv_shape = (1, GLB_N_KV, MAX_CTX, GLB_D_HEAD)
        else:
            kv_shape = (1, SLD_N_KV, MAX_CTX, SLD_D_HEAD)
        k_cache = torch.zeros(*kv_shape, dtype=torch.float16)
        v_cache = torch.zeros(*kv_shape, dtype=torch.float16)

        with torch.no_grad():
            x_out, k_new, v_new = layer.forward_attn(
                x, cos_r, sin_r, k_cache, v_cache, attn_mask, kv_write_mask)
            x_out = layer.forward_ffn(x_out)

        x = x_out
        h_np = x[0, 0].float().numpy()
        hf_h = hf[f"layer_{layer_idx + 1}"]
        cos = cos_sim(h_np, hf_h)
        l2 = np.linalg.norm(h_np)
        hf_l2 = np.linalg.norm(hf_h)
        print(f"  L{layer_idx:2d} [{lt:7s}] cos={cos:.6f} L2={l2:.2f} (HF={hf_l2:.2f}) "
              f"x[0:4]=[{h_np[0]:.4f},{h_np[1]:.4f},{h_np[2]:.4f},{h_np[3]:.4f}]")

        del layer


if __name__ == "__main__":
    main()
