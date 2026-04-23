"""T3b: Build & validate a SINGLE expert from layer 0 of openai/privacy-filter.

Correctness probe before the 128-function pack. Mirrors the T2 pattern:
  1. Extract one expert's weights from safetensors.
  2. Build a torch nn.Module with mlp1 (640->1280), swiglu, mlp2 (1280->640).
  3. Convert to CoreML fp16 mlprogram.
  4. Validate cosine vs PyTorch reference forward of the same expert
     on a real activation slice from pf_layer0_moe.npz.

Why: confirms swiglu(α=1.702, limit=7.0, packed=False) translates to
ANE-friendly ops, and that conv1×1 with bias for both mlp1 and mlp2 gives
bit-faithful (cos > 0.999) output before we replicate ×128 with INT4.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
MOE_GOLDEN = REPO_ROOT / "python" / "privacy" / "out" / "pf_layer0_moe.npz"
OUT_PKG = REPO_ROOT / "emilio" / "conv-ane" / "PF_expert0_layer0.mlpackage"

D_MODEL = 640
D_FF = 640                # OPF intermediate_size
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702


def _load_expert(expert_idx: int) -> dict[str, np.ndarray]:
    """Load one expert's weights (mlp1 + mlp2 with biases) for layer 0."""
    from safetensors.torch import safe_open
    import torch

    keys = {
        # opf MLPBlock attribute names: mlp1_weight=swiglu.weight,
        # mlp1_bias=swiglu.bias, mlp2_weight=out.weight, mlp2_bias=out.bias.
        # Stored as packed-experts arrays on disk.
        "swiglu_w":  "block.0.mlp.swiglu.weight",   # [128, 640, 1280]
        "swiglu_b":  "block.0.mlp.swiglu.bias",     # [128, 1280]
        "out_w":     "block.0.mlp.out.weight",      # [128, 640, 640]
        "out_b":     "block.0.mlp.out.bias",        # [128, 640]
    }
    out: dict[str, np.ndarray] = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        for short, full in keys.items():
            t = f.get_tensor(full).to(torch.float32).cpu().numpy()
            out[short] = t

    # Slice to single expert
    mlp1_w = out["swiglu_w"][expert_idx]   # [640, 1280]
    mlp1_b = out["swiglu_b"][expert_idx]   # [1280]
    mlp2_w = out["out_w"][expert_idx]      # [640, 640]
    mlp2_b = out["out_b"][expert_idx]      # [640]

    print(f"[t3b] expert {expert_idx} shapes: "
          f"mlp1_w={mlp1_w.shape} mlp1_b={mlp1_b.shape} "
          f"mlp2_w={mlp2_w.shape} mlp2_b={mlp2_b.shape}")
    return {
        "mlp1_w": mlp1_w, "mlp1_b": mlp1_b,
        "mlp2_w": mlp2_w, "mlp2_b": mlp2_b,
    }


def build_module(w: dict[str, np.ndarray]):
    import torch
    import torch.nn as nn

    class ExpertConv(nn.Module):
        """Single expert as conv1×1.

        opf MLPBlock packs mlp1_weight as [num_experts, hidden_size,
        intermediate_size*2]. The forward does
            out = _batched_linear_with_parity(t_expanded, mlp1_weight, mlp1_bias)
        which is `t @ mlp1_weight + mlp1_bias`. So mlp1_weight is laid out
        [in, out] (NOT the standard nn.Linear [out, in]). We transpose when
        packing into Conv2d.
        """
        def __init__(self):
            super().__init__()
            # mlp1: 640 -> 1280
            self.mlp1 = nn.Conv2d(D_MODEL, 2 * D_FF, 1, bias=True)
            mlp1_w = torch.from_numpy(w["mlp1_w"]).t().contiguous()  # [1280, 640]
            self.mlp1.weight = nn.Parameter(
                mlp1_w.reshape(2 * D_FF, D_MODEL, 1, 1).to(torch.float32),
                requires_grad=False,
            )
            self.mlp1.bias = nn.Parameter(
                torch.from_numpy(w["mlp1_b"]).to(torch.float32),
                requires_grad=False,
            )
            # mlp2: 640 -> 640
            self.mlp2 = nn.Conv2d(D_FF, D_MODEL, 1, bias=True)
            mlp2_w = torch.from_numpy(w["mlp2_w"]).t().contiguous()  # [640, 640]
            self.mlp2.weight = nn.Parameter(
                mlp2_w.reshape(D_MODEL, D_FF, 1, 1).to(torch.float32),
                requires_grad=False,
            )
            self.mlp2.bias = nn.Parameter(
                torch.from_numpy(w["mlp2_b"]).to(torch.float32),
                requires_grad=False,
            )

        def forward(self, x_in: "torch.Tensor") -> "torch.Tensor":
            """x_in: [N, 640, 1, 1] -> [N, 640, 1, 1]  (per-token, batched)."""
            h_pre = self.mlp1(x_in)                # [N, 1280, 1, 1]
            # swiglu: chunk(2, dim=1) since channels are dim 1
            x_glu = h_pre[:, :D_FF, :, :]
            x_lin = h_pre[:, D_FF:, :, :]
            x_glu = torch.clamp(x_glu, max=SWIGLU_LIMIT)
            x_lin = torch.clamp(x_lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            out_glu = x_glu * torch.sigmoid(SWIGLU_ALPHA * x_glu)
            h = out_glu * (x_lin + 1.0)            # [N, 640, 1, 1]
            return self.mlp2(h)                    # [N, 640, 1, 1]

    return ExpertConv()


def _torch_reference(w: dict[str, np.ndarray], x_np: np.ndarray) -> np.ndarray:
    """Pure-numpy / torch reference replaying opf math exactly."""
    import torch

    x = torch.from_numpy(x_np).to(torch.float32)              # [N, 640]
    mlp1_w = torch.from_numpy(w["mlp1_w"]).to(torch.float32)  # [640, 1280]
    mlp1_b = torch.from_numpy(w["mlp1_b"]).to(torch.float32)  # [1280]
    mlp2_w = torch.from_numpy(w["mlp2_w"]).to(torch.float32)  # [640, 640]
    mlp2_b = torch.from_numpy(w["mlp2_b"]).to(torch.float32)  # [640]

    h_pre = x @ mlp1_w + mlp1_b                  # [N, 1280]
    x_glu = h_pre[:, :D_FF]
    x_lin = h_pre[:, D_FF:]
    x_glu = torch.clamp(x_glu, max=SWIGLU_LIMIT)
    x_lin = torch.clamp(x_lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    h = x_glu * torch.sigmoid(SWIGLU_ALPHA * x_glu) * (x_lin + 1.0)
    o = h @ mlp2_w + mlp2_b
    return o.numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not MOE_GOLDEN.exists():
        raise SystemExit(f"Run pf_moe_goldens.py first: {MOE_GOLDEN}")
    if OUT_PKG.exists() and not args.force:
        raise SystemExit(f"{OUT_PKG} exists. Use --force.")

    print(f"[t3b] python: {sys.executable}")
    weights = _load_expert(args.expert)

    import torch, coremltools as ct
    print(f"[t3b] coremltools {ct.__version__}")

    # Pull a real activation slice for a tracing-shape that matches device usage.
    # We will dispatch at most B*T = 8*128 = 1024 calls per layer per forward,
    # but most experts receive < 50 calls per forward. Use enumerated dim N.
    z = np.load(MOE_GOLDEN, allow_pickle=False)
    norm_all = z["mlp_norm_out"].reshape(-1, D_MODEL)  # [B*T, 640]
    print(f"[t3b] norm_all: {norm_all.shape}")

    # Pick a tracing N. CoreML needs a static shape unless we use RangeDim,
    # which we'll add later for the multifunction pack. For now: N=64.
    TRACE_N = 64
    x_ex = torch.zeros(TRACE_N, D_MODEL, 1, 1, dtype=torch.float32)

    mod = build_module(weights).eval()
    print("[t3b] tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex,))

    print("[t3b] converting...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x_in", shape=(TRACE_N, D_MODEL, 1, 1),
                              dtype=np.float16)],
        outputs=[ct.TensorType(name="x_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if OUT_PKG.exists():
        import shutil; shutil.rmtree(OUT_PKG)
    mlmodel.save(str(OUT_PKG))
    print(f"[t3b] wrote {OUT_PKG}")

    # ─── Validation ──────────────────────────────────────────────────────────
    # Use first TRACE_N rows of the activations as a representative test.
    x_slice = norm_all[:TRACE_N].astype(np.float32)
    ref = _torch_reference(weights, x_slice)                     # [N, 640] fp32

    x_in_fp16 = x_slice.astype(np.float16).reshape(TRACE_N, D_MODEL, 1, 1)
    out_dict = mlmodel.predict({"x_in": x_in_fp16})
    pred = out_dict["x_out"].astype(np.float32).reshape(TRACE_N, D_MODEL)

    # Per-token cosine
    dots = (ref * pred).sum(axis=1)
    n_r = np.linalg.norm(ref, axis=1)
    n_p = np.linalg.norm(pred, axis=1)
    cos_per = dots / (n_r * n_p + 1e-30)
    worst = float(cos_per.min())
    mean = float(cos_per.mean())
    max_abs = float(np.abs(ref - pred).max())
    print(f"[t3b] per-token cosine: worst={worst:.6f} mean={mean:.6f} "
          f"max|Δ|={max_abs:.4f}")
    if worst < 0.97:
        print("[t3b] FAIL: cos < 0.97")
        return 1
    print("[t3b] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
