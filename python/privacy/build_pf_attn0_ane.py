"""T2b: Convert layer-0 AttentionBlock of openai/privacy-filter to CoreML.

Pattern mirrors emilio/conv-ane/gguf_to_ane.py (PyTorch + Conv2d 1×1 + ct.convert),
but adapted for prefill (T=128, no KV cache) and the OPF-specific bits:
  - YaRN RoPE (scaling_factor=32, base=150000, init_ctx=4096, ntk_alpha=1, ntk_beta=32)
  - Bidirectional sliding window (left=128, right=128 — full attention at T=128)
  - Attention sinks (per-head learned bias added as an extra softmax column)
  - GQA: 14 Q heads, 2 KV heads, q_mult=7, head_dim=64

Inputs:
  x_in:        [1, 640, 1, 128] fp16  (residual stream, channels-second)
  pad_add:     [1, 1, 1, 1, 128] fp16  (0 at valid positions, -inf at padded)

Output:
  x_out:       [1, 640, 1, 128] fp16

Run with Xcode's python3 (coremltools 9):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      python/privacy/build_pf_attn0_ane.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
OUT_PKG = REPO_ROOT / "emilio" / "conv-ane" / "PF_attn0_T128.mlpackage"

# OPF model.config values (read by hand from config.json; verified by pf_ref.py guard)
D_MODEL = 640
N_Q_HEADS = 14
N_KV_HEADS = 2
HEAD_DIM = 64
Q_MULT = N_Q_HEADS // N_KV_HEADS  # 7
QKV_DIM = (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM  # 1152
ATTN_OUT_DIM = N_Q_HEADS * HEAD_DIM  # 896
RMS_EPS = 1e-5
ROPE_BASE = 150000.0
ROPE_SCALING = 32.0
ROPE_INIT_CTX = 4096
ROPE_NTK_ALPHA = 1.0
ROPE_NTK_BETA = 32.0
T_SEQ = 128


def _yarn_cos_sin(T: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce opf RotaryEmbedding._compute_cos_sin in numpy."""
    d_half = HEAD_DIM // 2
    freq = ROPE_BASE ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM)
    # YaRN concentration
    concentration = 0.1 * math.log(ROPE_SCALING) + 1.0
    # NTK by parts
    low = (d_half * math.log(ROPE_INIT_CTX / (ROPE_NTK_BETA * 2 * math.pi))
           / math.log(ROPE_BASE))
    high = (d_half * math.log(ROPE_INIT_CTX / (ROPE_NTK_ALPHA * 2 * math.pi))
            / math.log(ROPE_BASE))
    assert 0 < low < high < d_half - 1
    interpolation = 1.0 / (ROPE_SCALING * freq)
    extrapolation = 1.0 / freq
    ramp = (np.arange(d_half, dtype=np.float64) - low) / (high - low)
    mask = 1.0 - np.clip(ramp, 0.0, 1.0)
    inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
    t = np.arange(T, dtype=np.float64)
    freqs = np.outer(t, inv_freq)  # [T, d_half]
    cos = np.cos(freqs) * concentration
    sin = np.sin(freqs) * concentration
    return cos.astype(np.float32), sin.astype(np.float32)


def _load_block0_weights() -> dict[str, np.ndarray]:
    """Load attention block 0 weights from the safetensors file."""
    # Use torch loader because the file is bf16, which numpy can't represent.
    from safetensors.torch import safe_open
    import torch as _torch

    wanted = {
        "norm_scale":  "block.0.attn.norm.scale",
        "qkv_w":       "block.0.attn.qkv.weight",
        "qkv_b":       "block.0.attn.qkv.bias",
        "out_w":       "block.0.attn.out.weight",
        "out_b":       "block.0.attn.out.bias",
        "sinks":       "block.0.attn.sinks",
    }
    out: dict[str, np.ndarray] = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        keys = set(f.keys())
        missing = [v for v in wanted.values() if v not in keys]
        if missing:
            sample = sorted(k for k in keys if "block.0.attn" in k)[:20]
            raise SystemExit(f"Missing keys: {missing}. Sample block.0.attn keys: {sample}")
        for short, full in wanted.items():
            t = f.get_tensor(full).to(_torch.float32).cpu().numpy()
            out[short] = t
    return out


def build_module(weights: dict[str, np.ndarray]):
    import torch
    import torch.nn as nn

    cos_np, sin_np = _yarn_cos_sin(T_SEQ)  # [T, 32] fp32
    print(f"[t2b] cos[0:3,0:3]={cos_np[:3,:3]}")

    class AttnBlock0(nn.Module):
        def __init__(self):
            super().__init__()
            # RMSNorm scale [640]
            self.norm_scale = nn.Parameter(
                torch.from_numpy(weights["norm_scale"]).to(torch.float32),
                requires_grad=False,
            )
            # QKV: 640 -> 1152 conv1×1 with bias
            self.qkv = nn.Conv2d(D_MODEL, QKV_DIM, 1, bias=True)
            # opf Linear weight is [out, in]; reshape to [out, in, 1, 1]
            self.qkv.weight = nn.Parameter(
                torch.from_numpy(weights["qkv_w"]).to(torch.float32)
                     .reshape(QKV_DIM, D_MODEL, 1, 1),
                requires_grad=False,
            )
            self.qkv.bias = nn.Parameter(
                torch.from_numpy(weights["qkv_b"]).to(torch.float32),
                requires_grad=False,
            )
            # Out: 896 -> 640 conv1×1 with bias
            self.out = nn.Conv2d(ATTN_OUT_DIM, D_MODEL, 1, bias=True)
            self.out.weight = nn.Parameter(
                torch.from_numpy(weights["out_w"]).to(torch.float32)
                     .reshape(D_MODEL, ATTN_OUT_DIM, 1, 1),
                requires_grad=False,
            )
            self.out.bias = nn.Parameter(
                torch.from_numpy(weights["out_b"]).to(torch.float32),
                requires_grad=False,
            )
            # Sinks [n_q_heads=14] (in opf: per-head, then reshape to [n_kv,q_mult])
            self.sinks = nn.Parameter(
                torch.from_numpy(weights["sinks"]).to(torch.float32),
                requires_grad=False,
            )
            # YaRN cos/sin tables [T, 32] — registered as buffers so they bake in
            self.register_buffer("cos_t", torch.from_numpy(cos_np).to(torch.float32))
            self.register_buffer("sin_t", torch.from_numpy(sin_np).to(torch.float32))

        def _rmsnorm(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: [B, 640, 1, T], normalize along channel dim 1 in fp32
            xf = x.float()
            var = xf.pow(2).mean(dim=1, keepdim=True)
            xn = xf * torch.rsqrt(var + RMS_EPS)
            scale = self.norm_scale.view(1, -1, 1, 1)
            return xn * scale  # stays fp32 (we will cast back to fp16 outside)

        def _apply_rope_interleaved(self, x_chd: "torch.Tensor") -> "torch.Tensor":
            """x_chd: [B, H, head_dim=64, T]. Apply OPF interleaved RoPE."""
            # opf indexes the last dim of [..., head_dim]; here head_dim is axis 2
            x_e = x_chd[:, :, 0::2, :]   # even -> "cos partner" [B,H,32,T]
            x_o = x_chd[:, :, 1::2, :]   # odd  -> "sin partner" [B,H,32,T]
            # cos_t/sin_t: [T, 32] -> [1,1,32,T]
            cos = self.cos_t.t().unsqueeze(0).unsqueeze(0)  # [1,1,32,T]
            sin = self.sin_t.t().unsqueeze(0).unsqueeze(0)  # [1,1,32,T]
            o_e = x_e * cos - x_o * sin
            o_o = x_o * cos + x_e * sin
            # Re-interleave: stack along new dim then reshape
            stacked = torch.stack([o_e, o_o], dim=3)        # [B,H,32,2,T]
            B, H, _, _, T = stacked.shape
            return stacked.reshape(B, H, HEAD_DIM, T)

        def forward(self, x_in: "torch.Tensor", pad_add: "torch.Tensor") -> "torch.Tensor":
            """
            x_in:    [1, 640, 1, T] fp16
            pad_add: [1, 1, 1, T] fp16 — additive (0 valid, -inf padded)
            return:  [1, 640, 1, T] fp16
            """
            residual = x_in
            B = x_in.shape[0]
            T = x_in.shape[-1]

            # RMSNorm in fp32, then cast for the conv
            x_n = self._rmsnorm(x_in).to(x_in.dtype)

            # QKV  [B, 1152, 1, T]
            qkv = self.qkv(x_n).squeeze(2)  # [B, 1152, T]

            # Slice channels: Q=896, K=128, V=128
            q = qkv[:, :ATTN_OUT_DIM, :]                          # [B, 896, T]
            k = qkv[:, ATTN_OUT_DIM:ATTN_OUT_DIM + N_KV_HEADS*HEAD_DIM, :]   # [B, 128, T]
            v = qkv[:, ATTN_OUT_DIM + N_KV_HEADS*HEAD_DIM:, :]    # [B, 128, T]

            # Reshape to per-head: Q -> [B, n_kv, q_mult, head_dim, T]
            #                      K -> [B, n_kv, head_dim, T]
            #                      V -> [B, n_kv, head_dim, T]
            q = q.reshape(B, N_KV_HEADS, Q_MULT, HEAD_DIM, T)
            k = k.reshape(B, N_KV_HEADS, HEAD_DIM, T)
            v = v.reshape(B, N_KV_HEADS, HEAD_DIM, T)

            # RoPE on Q (treat the q_mult as head dim for shape purposes)
            # Flatten [n_kv, q_mult] into a single H axis for the helper
            q_flat = q.reshape(B, N_KV_HEADS * Q_MULT, HEAD_DIM, T)  # [B,14,64,T]
            q_flat = self._apply_rope_interleaved(q_flat)
            q = q_flat.reshape(B, N_KV_HEADS, Q_MULT, HEAD_DIM, T)
            k = self._apply_rope_interleaved(k)                       # [B,2,64,T]

            # qk_scale = 1/sqrt(sqrt(head_dim)) — applied to BOTH q and k (opf does this)
            qk_scale = 1.0 / math.sqrt(math.sqrt(HEAD_DIM))
            q = q * qk_scale
            k = k * qk_scale

            # Scores: q [B,2,7,T_q,64] · k [B,2,1,64,T_k] -> [B,2,7,T_q,T_k]
            q_r = q.permute(0, 1, 2, 4, 3)            # [B, n_kv, q_mult, T_q, head_dim]
            k_r = k.unsqueeze(2)                      # [B, n_kv, 1,      head_dim, T_k]
            scores = torch.matmul(q_r, k_r)           # broadcast q_mult

            # Promote to fp32 for softmax stability
            scores = scores.float()

            # Apply key padding mask (additive) on T_k dim
            # pad_add: [B, 1, 1, T] -> [B, 1, 1, 1, T]
            scores = scores + pad_add.unsqueeze(2).float()

            # Append attention sink as one extra T_k+1 column.
            # opf: sink_scores = (S * log(2.0)).reshape(n_heads, q_mult)
            # Then expand to [B, T_q, n_heads, q_mult, 1]; here our layout is
            # [B, n_kv, q_mult, T_q, 1] — matching.
            sink_per_head = (self.sinks * math.log(2.0)).reshape(N_KV_HEADS, Q_MULT)
            sink_col = sink_per_head.view(1, N_KV_HEADS, Q_MULT, 1, 1).expand(
                B, N_KV_HEADS, Q_MULT, T, 1
            ).float()
            scores_with_sink = torch.cat([scores, sink_col], dim=-1)  # [...,T_k+1]

            # Softmax then drop the sink column
            w = torch.softmax(scores_with_sink, dim=-1)
            w = w[..., :-1].to(v.dtype)  # back to fp16 [B,2,7,T_q,T_k]

            # Attn out: w [B,2,7,T_q,T_k] · v [B,2,1,T_k,head_dim] -> [B,2,7,T_q,head_dim]
            v_r = v.permute(0, 1, 3, 2).unsqueeze(2)  # [B,2,1,T_k,head_dim]
            head_out = torch.matmul(w, v_r)           # broadcast q_mult

            # Reshape to [B, 14*64, 1, T_q] = [B, 896, 1, T]
            attn = head_out.permute(0, 1, 2, 4, 3)   # [B,2,7,head_dim,T_q]
            attn = attn.reshape(B, ATTN_OUT_DIM, 1, T)

            proj = self.out(attn)                    # [B, 640, 1, T]
            return residual + proj

    return AttnBlock0()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_PKG.exists() and not args.force:
        raise SystemExit(f"{OUT_PKG} exists. Use --force to overwrite.")

    print(f"[t2b] python: {sys.executable}")
    print(f"[t2b] loading weights from {WEIGHTS}")
    weights = _load_block0_weights()
    for k, v in weights.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    import torch
    import coremltools as ct
    print(f"[t2b] coremltools version: {ct.__version__}")

    mod = build_module(weights).eval()

    # Trace with example inputs
    x_ex = torch.zeros(1, D_MODEL, 1, T_SEQ, dtype=torch.float32)
    pad_ex = torch.zeros(1, 1, 1, T_SEQ, dtype=torch.float32)
    print("[t2b] tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex, pad_ex))

    print("[t2b] converting to CoreML (fp16, ALL compute units)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_in", shape=(1, D_MODEL, 1, T_SEQ), dtype=np.float16),
            ct.TensorType(name="pad_add", shape=(1, 1, 1, T_SEQ), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="x_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )

    if OUT_PKG.exists():
        import shutil
        shutil.rmtree(OUT_PKG)
    mlmodel.save(str(OUT_PKG))
    print(f"[t2b] wrote {OUT_PKG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
