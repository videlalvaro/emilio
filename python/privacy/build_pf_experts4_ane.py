"""T3c: validate the MultiFunctionDescriptor plumbing with 4 experts.

This is a mechanism check, not the full design. Builds one .mlpackage that
contains 4 expert functions (`expert_0`, `expert_1`, `expert_2`, `expert_3`)
and validates each one against its PyTorch reference.

Once this works, T3d will scale to 128 experts (and probably INT4 + gatekeeper
review since 128*1.6MB fp16 crosses the 96MB ANE residency cliff).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
MOE_GOLDEN = REPO_ROOT / "python" / "privacy" / "out" / "pf_layer0_moe.npz"
TMP_DIR = REPO_ROOT / "emilio" / "conv-ane" / "_pf_expert_tmp"
OUT_PKG = REPO_ROOT / "emilio" / "conv-ane" / "PF_experts4_layer0.mlpackage"

D_MODEL = 640
D_FF = 640
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
TRACE_N = 64
EXPERT_IDS = (0, 1, 2, 3)


def load_all_experts() -> dict[str, np.ndarray]:
    """Load packed expert tensors for layer 0 once."""
    from safetensors.torch import safe_open
    import torch

    keys = {
        "swiglu_w": "block.0.mlp.swiglu.weight",
        "swiglu_b": "block.0.mlp.swiglu.bias",
        "out_w":    "block.0.mlp.out.weight",
        "out_b":    "block.0.mlp.out.bias",
    }
    out: dict[str, np.ndarray] = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        for short, full in keys.items():
            out[short] = f.get_tensor(full).to(torch.float32).cpu().numpy()
    return out


def build_expert_module(packed: dict[str, np.ndarray], expert_idx: int):
    import torch
    import torch.nn as nn

    mlp1_w_in_out = packed["swiglu_w"][expert_idx]   # [640, 1280]
    mlp1_b = packed["swiglu_b"][expert_idx]
    mlp2_w_in_out = packed["out_w"][expert_idx]      # [640, 640]
    mlp2_b = packed["out_b"][expert_idx]

    class ExpertConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp1 = nn.Conv2d(D_MODEL, 2 * D_FF, 1, bias=True)
            self.mlp1.weight = nn.Parameter(
                torch.from_numpy(mlp1_w_in_out).t().contiguous()
                     .reshape(2 * D_FF, D_MODEL, 1, 1).to(torch.float32),
                requires_grad=False,
            )
            self.mlp1.bias = nn.Parameter(
                torch.from_numpy(mlp1_b).to(torch.float32),
                requires_grad=False,
            )
            self.mlp2 = nn.Conv2d(D_FF, D_MODEL, 1, bias=True)
            self.mlp2.weight = nn.Parameter(
                torch.from_numpy(mlp2_w_in_out).t().contiguous()
                     .reshape(D_MODEL, D_FF, 1, 1).to(torch.float32),
                requires_grad=False,
            )
            self.mlp2.bias = nn.Parameter(
                torch.from_numpy(mlp2_b).to(torch.float32),
                requires_grad=False,
            )

        def forward(self, x_in):
            h = self.mlp1(x_in)
            x_glu = h[:, :D_FF, :, :]
            x_lin = h[:, D_FF:, :, :]
            x_glu = torch.clamp(x_glu, max=SWIGLU_LIMIT)
            x_lin = torch.clamp(x_lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            out_glu = x_glu * torch.sigmoid(SWIGLU_ALPHA * x_glu)
            return self.mlp2(out_glu * (x_lin + 1.0))

    return ExpertConv()


def torch_reference(packed: dict[str, np.ndarray], expert_idx: int,
                    x_np: np.ndarray) -> np.ndarray:
    import torch
    x = torch.from_numpy(x_np).to(torch.float32)
    mlp1_w = torch.from_numpy(packed["swiglu_w"][expert_idx]).to(torch.float32)
    mlp1_b = torch.from_numpy(packed["swiglu_b"][expert_idx]).to(torch.float32)
    mlp2_w = torch.from_numpy(packed["out_w"][expert_idx]).to(torch.float32)
    mlp2_b = torch.from_numpy(packed["out_b"][expert_idx]).to(torch.float32)
    h = x @ mlp1_w + mlp1_b
    g, l = h[:, :D_FF], h[:, D_FF:]
    g = torch.clamp(g, max=SWIGLU_LIMIT)
    l = torch.clamp(l, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    o = g * torch.sigmoid(SWIGLU_ALPHA * g) * (l + 1.0)
    return (o @ mlp2_w + mlp2_b).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not MOE_GOLDEN.exists():
        raise SystemExit(f"Run pf_moe_goldens.py first: {MOE_GOLDEN}")
    if OUT_PKG.exists() and not args.force:
        raise SystemExit(f"{OUT_PKG} exists. Use --force.")

    print(f"[t3c] python: {sys.executable}")
    import torch, coremltools as ct
    print(f"[t3c] coremltools {ct.__version__}")

    packed = load_all_experts()

    # Step 1: build a single-expert .mlpackage per expert with function name
    # rewritten to "expert_{i}" via the mlmodel.functions API.
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    sub_paths: list[tuple[str, int]] = []
    for eid in EXPERT_IDS:
        print(f"[t3c] building expert {eid}")
        mod = build_expert_module(packed, eid).eval()
        x_ex = torch.zeros(TRACE_N, D_MODEL, 1, 1, dtype=torch.float32)
        with torch.no_grad():
            traced = torch.jit.trace(mod, (x_ex,))
        m = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x_in", shape=(TRACE_N, D_MODEL, 1, 1),
                                  dtype=np.float16)],
            outputs=[ct.TensorType(name="x_out", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        path = TMP_DIR / f"expert_{eid}.mlpackage"
        m.save(str(path))
        sub_paths.append((str(path), eid))

    # Step 2: combine into a multifunction package.
    print("[t3c] assembling multifunction package")
    desc = ct.utils.MultiFunctionDescriptor()
    for p, eid in sub_paths:
        # Source function name in a single-function mlprogram is "main".
        desc.add_function(p, src_function_name="main",
                          target_function_name=f"expert_{eid}")
    desc.default_function_name = f"expert_{EXPERT_IDS[0]}"

    if OUT_PKG.exists():
        shutil.rmtree(OUT_PKG)
    ct.utils.save_multifunction(desc, str(OUT_PKG))
    size_mb = sum(p.stat().st_size for p in OUT_PKG.rglob("*") if p.is_file()) / 1024**2
    print(f"[t3c] wrote {OUT_PKG} ({size_mb:.2f} MB)")

    # Step 3: validate each function loads + matches reference.
    z = np.load(MOE_GOLDEN, allow_pickle=False)
    norm_all = z["mlp_norm_out"].reshape(-1, D_MODEL)[:TRACE_N].astype(np.float32)

    for eid in EXPERT_IDS:
        m = ct.models.MLModel(str(OUT_PKG), function_name=f"expert_{eid}",
                              compute_units=ct.ComputeUnit.ALL)
        x_in = norm_all.astype(np.float16).reshape(TRACE_N, D_MODEL, 1, 1)
        out = m.predict({"x_in": x_in})["x_out"].astype(np.float32).reshape(TRACE_N, D_MODEL)
        ref = torch_reference(packed, eid, norm_all)
        cos = float((out * ref).sum() / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-30))
        max_abs = float(np.abs(out - ref).max())
        status = "PASS" if cos >= 0.97 else "FAIL"
        print(f"  expert_{eid}: cos={cos:.6f} max|Δ|={max_abs:.4f} [{status}]")
        if cos < 0.97:
            return 1

    # Cleanup intermediate packages
    shutil.rmtree(TMP_DIR)
    print("[t3c] PASS — multifunction plumbing verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
