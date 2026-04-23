"""Build tail ANE pack: final RMSNorm + unembed projection.

Replaces the last two CPU ops:
  - CPU rmsNorm(x, final_norm_scale)      → ANE
  - CPU cblas_sgemm(x, unembed^T) [640→33] → ANE

Input:  x [1, D_MODEL, 1, T_SEQ] fp16
Output: logits [1, N_LABELS, 1, T_SEQ] fp16

Usage:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      python/privacy/build_pf_tail_ane.py --force
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
PKG_DIR = REPO_ROOT / "emilio" / "conv-ane"

D_MODEL = 640
N_LABELS = 33
T_SEQ = 128
RMS_EPS = 1e-5
SAFE_NORM_K = 128.0


def _load_weights():
    from safetensors.torch import safe_open
    import torch as _torch
    out = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        out["final_norm_scale"] = f.get_tensor("norm.scale").to(_torch.float32).cpu().numpy()
        # Unembed: [N_LABELS, D_MODEL] — no bias in this model
        out["unembed_w"] = f.get_tensor("unembedding.weight").to(_torch.float32).cpu().numpy()
    return out


def build_module(weights):
    import torch
    import torch.nn as nn

    class TailBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_scale = nn.Parameter(
                torch.from_numpy(weights["final_norm_scale"]).to(torch.float32),
                requires_grad=False)
            self.unembed = nn.Conv2d(D_MODEL, N_LABELS, 1, bias=False)
            self.unembed.weight = nn.Parameter(
                torch.from_numpy(weights["unembed_w"]).to(torch.float32)
                     .reshape(N_LABELS, D_MODEL, 1, 1), requires_grad=False)

        def _rmsnorm(self, x):
            xs = x / SAFE_NORM_K
            var = (xs * xs).mean(dim=1, keepdim=True)
            xn = xs * torch.rsqrt(var + RMS_EPS)
            return xn * self.norm_scale.view(1, -1, 1, 1)

        def forward(self, x_in):
            xn = self._rmsnorm(x_in)
            return self.unembed(xn)

    return TailBlock()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_pkg = PKG_DIR / "PF_tail_T128.mlpackage"
    if out_pkg.exists() and not args.force:
        print(f"[tail] {out_pkg.name} exists, skipping (--force to rebuild)")
        return 0

    import torch
    import coremltools as ct
    print(f"[tail] python: {sys.executable}")
    print(f"[tail] coremltools {ct.__version__}")

    weights = _load_weights()
    mod = build_module(weights).eval()

    x_ex = torch.zeros(1, D_MODEL, 1, T_SEQ, dtype=torch.float32)
    print("[tail] tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex,))

    print("[tail] converting to CoreML (fp16, ALL)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_in", shape=(1, D_MODEL, 1, T_SEQ), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float16),
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
    print(f"[tail] wrote {out_pkg} ({sz:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
