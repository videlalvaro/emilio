"""Single-layer fp32-compute probe of PF_attn{L}_T128.

Writes a SIBLING artifact PF_attn{L}_T128_fp32.mlpackage (does NOT overwrite the
fp16 baseline). Inputs/outputs remain fp16; only compute_precision is bumped.
Used to test whether the L4 magnitude loss (||pred||/||gold||=0.74) is fixed by
fp32 internal compute, and whether it stays on ANE.

Usage (Xcode python3):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/privacy/build_pf_attn_fp32_probe.py --layer 4
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "privacy"))
from build_pf_attn_ane import build_module, _load_block_weights, D_MODEL, T_SEQ

PKG_DIR = REPO_ROOT / "emilio" / "conv-ane"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    args = ap.parse_args()
    out_pkg = PKG_DIR / f"PF_attn{args.layer}_T128_fp32.mlpackage"
    if out_pkg.exists():
        import shutil
        shutil.rmtree(out_pkg)
    print(f"[probe] python: {sys.executable}")
    print(f"[probe] layer {args.layer} -> {out_pkg.name} (FP32 compute)")
    weights = _load_block_weights(args.layer)
    import torch
    import coremltools as ct
    print(f"[probe] coremltools {ct.__version__}")
    mod = build_module(weights).eval()
    x_ex = torch.zeros(1, D_MODEL, 1, T_SEQ, dtype=torch.float32)
    pad_ex = torch.zeros(1, 1, 1, T_SEQ, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex, pad_ex))
    print("[probe] converting (compute_precision=FLOAT32, units=CPU_AND_NE) ...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_in", shape=(1, D_MODEL, 1, T_SEQ), dtype=np.float16),
            ct.TensorType(name="pad_add", shape=(1, 1, 1, T_SEQ), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="x_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    mlmodel.save(str(out_pkg))
    sz = sum(p.stat().st_size for p in out_pkg.rglob("*") if p.is_file()) / 1024**2
    print(f"[probe] wrote {out_pkg} ({sz:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
