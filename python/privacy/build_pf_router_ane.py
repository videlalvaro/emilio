"""Build per-layer router ANE pack: RMSNorm → gate_linear → softmax.

Output: emilio/conv-ane/PF_router_L{N}_T128.mlpackage
Input:  x_in [1, D, 1, T] fp16  (same ANE channel layout as attn packs)
Output: router_probs [1, N_EXPERTS, 1, T] fp16  (full softmax over 128 experts)

Top-K extraction stays CPU-side (pure index pick, ~microseconds).
Everything else — norm, gate GEMM, softmax — lands on ANE.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
PKG_DIR = REPO_ROOT / "emilio" / "conv-ane"

D_MODEL = 640
N_EXPERTS = 128
RMS_EPS = 1e-5
T_SEQ = 128
SAFE_NORM_K = 128.0


def _load_router_weights(layer: int):
    from safetensors.torch import safe_open
    import torch
    prefix = f"block.{layer}.mlp"
    out = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        out["norm_scale"] = f.get_tensor(f"{prefix}.norm.scale").to(torch.float32).cpu().numpy()
        out["gate_w"] = f.get_tensor(f"{prefix}.gate.weight").to(torch.float32).cpu().numpy()
        out["gate_b"] = f.get_tensor(f"{prefix}.gate.bias").to(torch.float32).cpu().numpy()
    return out


def build_module(weights):
    import torch
    import torch.nn as nn

    class RouterBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_scale = nn.Parameter(
                torch.from_numpy(weights["norm_scale"]).to(torch.float32),
                requires_grad=False)
            # Gate as Conv2d 1×1: [N_EXPERTS, D_MODEL, 1, 1]
            self.gate = nn.Conv2d(D_MODEL, N_EXPERTS, 1, bias=True)
            self.gate.weight = nn.Parameter(
                torch.from_numpy(weights["gate_w"]).to(torch.float32)
                     .reshape(N_EXPERTS, D_MODEL, 1, 1), requires_grad=False)
            self.gate.bias = nn.Parameter(
                torch.from_numpy(weights["gate_b"]).to(torch.float32),
                requires_grad=False)

        def _rmsnorm(self, x):
            # fp16-safe: pre-divide by K (same trick as attn packs)
            xs = x / SAFE_NORM_K
            var = (xs * xs).mean(dim=1, keepdim=True)
            xn = xs * torch.rsqrt(var + RMS_EPS)
            return xn * self.norm_scale.view(1, -1, 1, 1)

        def forward(self, x_in):
            # x_in: [1, D, 1, T]
            norm_out = self._rmsnorm(x_in)
            gate_logits = self.gate(norm_out)  # [1, N_EXPERTS, 1, T]
            router_probs = torch.softmax(gate_logits, dim=1)
            return router_probs

    return RouterBlock()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--all-layers", action="store_true",
                    help="Build for all 8 layers (ignores --layer).")
    args = ap.parse_args()

    layers = list(range(8)) if args.all_layers else [args.layer]

    import torch
    import coremltools as ct
    print(f"[router] python: {sys.executable}")
    print(f"[router] coremltools {ct.__version__}")

    for layer in layers:
        out_pkg = PKG_DIR / f"PF_router_L{layer}_T128.mlpackage"
        if out_pkg.exists() and not args.force:
            print(f"[router] {out_pkg.name} exists, skipping (--force to rebuild)")
            continue

        print(f"[router] layer {layer} -> {out_pkg.name}")
        weights = _load_router_weights(layer)
        mod = build_module(weights).eval()

        x_ex = torch.zeros(1, D_MODEL, 1, T_SEQ, dtype=torch.float32)
        print(f"[router] tracing L{layer}...")
        with torch.no_grad():
            traced = torch.jit.trace(mod, (x_ex,))

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="x_in", shape=(1, D_MODEL, 1, T_SEQ), dtype=np.float16),
            ],
            outputs=[ct.TensorType(name="router_probs", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        if out_pkg.exists():
            import shutil; shutil.rmtree(out_pkg)
        mlmodel.save(str(out_pkg))
        sz = sum(p.stat().st_size for p in out_pkg.rglob("*") if p.is_file()) / 1024**2
        print(f"[router] wrote {out_pkg} ({sz:.2f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
