"""T5b: Parametrized AttentionBlock builder for any layer of openai/privacy-filter.

Same body as build_pf_attn0_ane.py but takes --layer N (0..7).
Output: emilio/conv-ane/PF_attn{N}_T128.mlpackage (FP16, fixed B=1, T=128).
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
RMS_EPS = 1e-5
ROPE_BASE = 150000.0
ROPE_SCALING = 32.0
ROPE_INIT_CTX = 4096
ROPE_NTK_ALPHA = 1.0
ROPE_NTK_BETA = 32.0
T_SEQ = 128


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


def _load_block_weights(layer: int):
    from safetensors.torch import safe_open
    import torch as _torch
    p = f"block.{layer}.attn"
    wanted = {
        "norm_scale":  f"{p}.norm.scale",
        "qkv_w":       f"{p}.qkv.weight",
        "qkv_b":       f"{p}.qkv.bias",
        "out_w":       f"{p}.out.weight",
        "out_b":       f"{p}.out.bias",
        "sinks":       f"{p}.sinks",
    }
    out = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        for short, full in wanted.items():
            out[short] = f.get_tensor(full).to(_torch.float32).cpu().numpy()
    return out


def build_module(weights, safe_norm: bool = False, safe_norm_k: float = 128.0):
    import torch
    import torch.nn as nn
    cos_np, sin_np = _yarn_cos_sin(T_SEQ)

    class AttnBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_scale = nn.Parameter(
                torch.from_numpy(weights["norm_scale"]).to(torch.float32),
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

        def _rmsnorm(self, x):
            if safe_norm:
                # fp16-safe: pre-divide by K so x^2 stays well below fp16 max.
                # mathematically identical: rsqrt(mean((x/K)^2)) == K * rsqrt(mean(x^2))
                # so (x) * rsqrt(mean(x^2) + eps) == (x/K) * rsqrt(mean((x/K)^2) + eps/K^2)
                # we drop the eps/K^2 term -- eps is 1e-5 and var is always >> eps in practice.
                xs = x / safe_norm_k
                var = (xs * xs).mean(dim=1, keepdim=True)
                xn = xs * torch.rsqrt(var + RMS_EPS)
                return xn * self.norm_scale.view(1, -1, 1, 1)
            xf = x.float()
            var = xf.pow(2).mean(dim=1, keepdim=True)
            xn = xf * torch.rsqrt(var + RMS_EPS)
            return xn * self.norm_scale.view(1, -1, 1, 1)

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
            residual = x_in
            B = x_in.shape[0]; T = x_in.shape[-1]
            x_n = self._rmsnorm(x_in).to(x_in.dtype)
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
            return residual + self.out(attn)
    return AttnBlock()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--precision", choices=["fp16", "fp32", "mixed"], default="fp16",
                    help="compute_precision for CoreML conversion. fp32 / mixed write to sibling _fp32 / _mixed paths.")
    ap.add_argument("--keep-fp32-ops", default="softmax",
                    help="Comma-separated op_types kept at fp32 when --precision=mixed. Default: softmax.")
    ap.add_argument("--safe-norm", action="store_true",
                    help="Use fp16-safe RMSNorm (pre-divide by K). Writes _safe suffix.")
    ap.add_argument("--safe-norm-k", type=float, default=128.0)
    args = ap.parse_args()
    base_suffix = {"fp16": "", "fp32": "_fp32", "mixed": "_mixed"}[args.precision]
    suffix = base_suffix + ("_safe" if args.safe_norm else "")
    out_pkg = PKG_DIR / f"PF_attn{args.layer}_T128{suffix}.mlpackage"
    if out_pkg.exists() and not args.force:
        print(f"[t5b] {out_pkg.name} exists, skipping (use --force to rebuild)")
        return 0
    print(f"[t5b] python: {sys.executable}")
    print(f"[t5b] layer {args.layer} -> {out_pkg.name}")
    weights = _load_block_weights(args.layer)
    import torch
    import coremltools as ct
    print(f"[t5b] coremltools {ct.__version__}")
    mod = build_module(weights, safe_norm=args.safe_norm,
                       safe_norm_k=args.safe_norm_k).eval()
    x_ex = torch.zeros(1, D_MODEL, 1, T_SEQ, dtype=torch.float32)
    pad_ex = torch.zeros(1, 1, 1, T_SEQ, dtype=torch.float32)
    print("[t5b] tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex, pad_ex))
    if args.precision == "fp32":
        cprec = ct.precision.FLOAT32
    elif args.precision == "mixed":
        keep = {s.strip() for s in args.keep_fp32_ops.split(",") if s.strip()}
        FP16ComputePrecision = ct.transform.FP16ComputePrecision
        # op_selector returns True => convert to fp16. False => keep fp32.
        cprec = FP16ComputePrecision(op_selector=lambda op: op.op_type not in keep)
        print(f"[t5b] mixed precision: keeping fp32 for ops: {sorted(keep)}")
    else:
        cprec = ct.precision.FLOAT16
    print(f"[t5b] converting to CoreML (compute_precision={args.precision}, ALL)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_in", shape=(1, D_MODEL, 1, T_SEQ), dtype=np.float16),
            ct.TensorType(name="pad_add", shape=(1, 1, 1, T_SEQ), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="x_out", dtype=np.float16)],
        compute_precision=cprec,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if out_pkg.exists():
        import shutil; shutil.rmtree(out_pkg)
    mlmodel.save(str(out_pkg))
    sz = sum(p.stat().st_size for p in out_pkg.rglob("*") if p.is_file()) / 1024**2
    print(f"[t5b] wrote {out_pkg} ({sz:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
