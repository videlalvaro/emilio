"""Build fused attn+router ANE pack: AttnBlock → residual → MLP-RMSNorm → gate → softmax.

ONE predict() call replaces:
  - attn predict()           (was already on ANE)
  - CPU RMSNorm (mlp_norm)   (moved to ANE)
  - CPU cblas_sgemm gate     (moved to ANE)
  - CPU softmax              (moved to ANE)

Outputs:
  x_attn       [1, D, 1, T]       — attention output with residual (for post-MoE residual add)
  normed_x     [1, D, 1, T]       — MLP-normed x, fp16, ANE layout (direct expert input — NO repack!)
  router_probs [1, N_EXPERTS, 1, T] — full softmax over 128 experts

The Swift driver then only does:
  1. top-K argpartition on router_probs (pure index pick, microseconds)
  2. Feed normed_x DIRECTLY to expert packs (already in [B_PACK, D, 1, 1] compatible layout)
  3. Scatter + residual add on x_attn

This eliminates per-layer: 1 CPU RMSNorm, 1 cblas_sgemm, 1 softmax, 1 pack/unpack cycle.

Usage:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      python/privacy/build_pf_fused_attn_router_ane.py --layer 0 --force
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      python/privacy/build_pf_fused_attn_router_ane.py --all-layers --force
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
PKG_DIR = REPO_ROOT / "emilio" / "conv-ane"

D_MODEL = 640
N_Q_HEADS = 14
N_KV_HEADS = 2
HEAD_DIM = 64
Q_MULT = N_Q_HEADS // N_KV_HEADS
QKV_DIM = (N_Q_HEADS + 2 * N_KV_HEADS) * HEAD_DIM
ATTN_OUT_DIM = N_Q_HEADS * HEAD_DIM
N_EXPERTS = 128
RMS_EPS = 1e-5
ROPE_BASE = 150000.0
ROPE_SCALING = 32.0
ROPE_INIT_CTX = 4096
ROPE_NTK_ALPHA = 1.0
ROPE_NTK_BETA = 32.0
T_SEQ = 128
SAFE_NORM_K = 128.0


def _yarn_cos_sin(T: int):
    d_half = HEAD_DIM // 2
    freq = ROPE_BASE ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM)
    concentration = 0.1 * math.log(ROPE_SCALING) + 1.0
    low = (d_half * math.log(ROPE_INIT_CTX / (ROPE_NTK_BETA * 2 * math.pi))
           / math.log(ROPE_BASE))
    high = (d_half * math.log(ROPE_INIT_CTX / (ROPE_NTK_ALPHA * 2 * math.pi))
            / math.log(ROPE_BASE))
    interpolation = 1.0 / (ROPE_SCALING * freq)
    extrapolation = 1.0 / freq
    ramp = (np.arange(d_half, dtype=np.float64) - low) / (high - low)
    mask = 1.0 - np.clip(ramp, 0.0, 1.0)
    inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
    t = np.arange(T, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    cos = np.cos(freqs) * concentration
    sin = np.sin(freqs) * concentration
    return cos.astype(np.float32), sin.astype(np.float32)


def _load_all_weights(layer: int):
    from safetensors.torch import safe_open
    import torch as _torch
    out = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        # Attn weights
        ap = f"block.{layer}.attn"
        out["attn_norm_scale"] = f.get_tensor(f"{ap}.norm.scale").to(_torch.float32).cpu().numpy()
        out["qkv_w"] = f.get_tensor(f"{ap}.qkv.weight").to(_torch.float32).cpu().numpy()
        out["qkv_b"] = f.get_tensor(f"{ap}.qkv.bias").to(_torch.float32).cpu().numpy()
        out["out_w"] = f.get_tensor(f"{ap}.out.weight").to(_torch.float32).cpu().numpy()
        out["out_b"] = f.get_tensor(f"{ap}.out.bias").to(_torch.float32).cpu().numpy()
        out["sinks"] = f.get_tensor(f"{ap}.sinks").to(_torch.float32).cpu().numpy()
        # Router weights
        mp = f"block.{layer}.mlp"
        out["mlp_norm_scale"] = f.get_tensor(f"{mp}.norm.scale").to(_torch.float32).cpu().numpy()
        out["gate_w"] = f.get_tensor(f"{mp}.gate.weight").to(_torch.float32).cpu().numpy()
        out["gate_b"] = f.get_tensor(f"{mp}.gate.bias").to(_torch.float32).cpu().numpy()
    return out


def build_module(weights):
    import torch
    import torch.nn as nn
    cos_np, sin_np = _yarn_cos_sin(T_SEQ)

    class FusedAttnRouterBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # --- Attn weights ---
            self.attn_norm_scale = nn.Parameter(
                torch.from_numpy(weights["attn_norm_scale"]).to(torch.float32),
                requires_grad=False)
            self.qkv = nn.Conv2d(D_MODEL, QKV_DIM, 1, bias=True)
            self.qkv.weight = nn.Parameter(
                torch.from_numpy(weights["qkv_w"]).to(torch.float32)
                     .reshape(QKV_DIM, D_MODEL, 1, 1), requires_grad=False)
            self.qkv.bias = nn.Parameter(
                torch.from_numpy(weights["qkv_b"]).to(torch.float32),
                requires_grad=False)
            self.out = nn.Conv2d(ATTN_OUT_DIM, D_MODEL, 1, bias=True)
            self.out.weight = nn.Parameter(
                torch.from_numpy(weights["out_w"]).to(torch.float32)
                     .reshape(D_MODEL, ATTN_OUT_DIM, 1, 1), requires_grad=False)
            self.out.bias = nn.Parameter(
                torch.from_numpy(weights["out_b"]).to(torch.float32),
                requires_grad=False)
            self.sinks = nn.Parameter(
                torch.from_numpy(weights["sinks"]).to(torch.float32),
                requires_grad=False)
            self.register_buffer("cos_t", torch.from_numpy(cos_np).to(torch.float32))
            self.register_buffer("sin_t", torch.from_numpy(sin_np).to(torch.float32))

            # --- Router weights ---
            self.mlp_norm_scale = nn.Parameter(
                torch.from_numpy(weights["mlp_norm_scale"]).to(torch.float32),
                requires_grad=False)
            self.gate = nn.Conv2d(D_MODEL, N_EXPERTS, 1, bias=True)
            self.gate.weight = nn.Parameter(
                torch.from_numpy(weights["gate_w"]).to(torch.float32)
                     .reshape(N_EXPERTS, D_MODEL, 1, 1), requires_grad=False)
            self.gate.bias = nn.Parameter(
                torch.from_numpy(weights["gate_b"]).to(torch.float32),
                requires_grad=False)

        def _rmsnorm_attn(self, x):
            xs = x / SAFE_NORM_K
            var = (xs * xs).mean(dim=1, keepdim=True)
            xn = xs * torch.rsqrt(var + RMS_EPS)
            return xn * self.attn_norm_scale.view(1, -1, 1, 1)

        def _rmsnorm_mlp(self, x):
            xs = x / SAFE_NORM_K
            var = (xs * xs).mean(dim=1, keepdim=True)
            xn = xs * torch.rsqrt(var + RMS_EPS)
            return xn * self.mlp_norm_scale.view(1, -1, 1, 1)

        def _rope(self, x_chd):
            x_e = x_chd[:, :, 0::2, :]
            x_o = x_chd[:, :, 1::2, :]
            cos = self.cos_t.t().unsqueeze(0).unsqueeze(0)
            sin = self.sin_t.t().unsqueeze(0).unsqueeze(0)
            o_e = x_e * cos - x_o * sin
            o_o = x_o * cos + x_e * sin
            stacked = torch.stack([o_e, o_o], dim=3)
            B, H, _, _, T = stacked.shape
            return stacked.reshape(B, H, HEAD_DIM, T)

        def forward(self, x_in, pad_add):
            # ===== ATTENTION =====
            residual = x_in
            B = x_in.shape[0]; T = x_in.shape[-1]
            x_n = self._rmsnorm_attn(x_in).to(x_in.dtype)
            qkv = self.qkv(x_n).squeeze(2)
            q = qkv[:, :ATTN_OUT_DIM, :]
            k = qkv[:, ATTN_OUT_DIM:ATTN_OUT_DIM + N_KV_HEADS*HEAD_DIM, :]
            v = qkv[:, ATTN_OUT_DIM + N_KV_HEADS*HEAD_DIM:, :]
            q = q.reshape(B, N_KV_HEADS, Q_MULT, HEAD_DIM, T)
            k = k.reshape(B, N_KV_HEADS, HEAD_DIM, T)
            v = v.reshape(B, N_KV_HEADS, HEAD_DIM, T)
            q_flat = q.reshape(B, N_KV_HEADS * Q_MULT, HEAD_DIM, T)
            q_flat = self._rope(q_flat)
            q = q_flat.reshape(B, N_KV_HEADS, Q_MULT, HEAD_DIM, T)
            k = self._rope(k)
            qk_scale = 1.0 / math.sqrt(math.sqrt(HEAD_DIM))
            q = q * qk_scale; k = k * qk_scale
            q_r = q.permute(0, 1, 2, 4, 3)
            k_r = k.unsqueeze(2)
            scores = torch.matmul(q_r, k_r).float()
            scores = scores + pad_add.unsqueeze(2).float()
            sink_per_head = (self.sinks * math.log(2.0)).reshape(N_KV_HEADS, Q_MULT)
            sink_col = sink_per_head.view(1, N_KV_HEADS, Q_MULT, 1, 1).expand(
                B, N_KV_HEADS, Q_MULT, T, 1).float()
            scores_with_sink = torch.cat([scores, sink_col], dim=-1)
            w = torch.softmax(scores_with_sink, dim=-1)
            w = w[..., :-1].to(v.dtype)
            v_r = v.permute(0, 1, 3, 2).unsqueeze(2)
            head_out = torch.matmul(w, v_r)
            attn = head_out.permute(0, 1, 2, 4, 3).reshape(B, ATTN_OUT_DIM, 1, T)
            x_attn = residual + self.out(attn)  # [1, D, 1, T]

            # ===== ROUTER (on same ANE call) =====
            normed_x = self._rmsnorm_mlp(x_attn)  # [1, D, 1, T]
            gate_logits = self.gate(normed_x)       # [1, N_EXPERTS, 1, T]
            router_probs = torch.softmax(gate_logits, dim=1)

            return x_attn, normed_x, router_probs

    return FusedAttnRouterBlock()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all-layers", action="store_true",
                    help="Build for all 8 layers (ignores --layer).")
    args = ap.parse_args()

    layers = list(range(8)) if args.all_layers else [args.layer]

    import torch
    import coremltools as ct
    print(f"[fused] python: {sys.executable}")
    print(f"[fused] coremltools {ct.__version__}")

    for layer in layers:
        out_pkg = PKG_DIR / f"PF_fused_L{layer}_T128.mlpackage"
        if out_pkg.exists() and not args.force:
            print(f"[fused] {out_pkg.name} exists, skipping (--force to rebuild)")
            continue

        print(f"[fused] layer {layer} -> {out_pkg.name}")
        weights = _load_all_weights(layer)
        mod = build_module(weights).eval()

        x_ex = torch.zeros(1, D_MODEL, 1, T_SEQ, dtype=torch.float32)
        pad_ex = torch.zeros(1, 1, 1, T_SEQ, dtype=torch.float32)
        print(f"[fused] tracing L{layer}...")
        with torch.no_grad():
            traced = torch.jit.trace(mod, (x_ex, pad_ex))

        print(f"[fused] converting to CoreML (fp16, ALL)...")
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="x_in", shape=(1, D_MODEL, 1, T_SEQ), dtype=np.float16),
                ct.TensorType(name="pad_add", shape=(1, 1, 1, T_SEQ), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="x_attn", dtype=np.float16),
                ct.TensorType(name="normed_x", dtype=np.float16),
                ct.TensorType(name="router_probs", dtype=np.float16),
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        if out_pkg.exists():
            import shutil; shutil.rmtree(out_pkg)
        mlmodel.save(str(out_pkg))
        sz = sum(p.stat().st_size for p in out_pkg.rglob("*") if p.is_file()) / 1024**2
        print(f"[fused] wrote {out_pkg} ({sz:.2f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
