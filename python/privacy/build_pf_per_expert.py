"""Per-expert mlpackage probe: build ONE expert as a standalone Conv2d-SwiGLU
graph, validate cos vs torch, report residency, save package.

If the single-expert call lands on ANE and bench shows ~0.4ms (matching the
top-4 ceiling probe divided by 4), we proceed to build all 128 experts.

Inputs:  x_in[B, D_MODEL, 1, 1] fp16
Outputs: y_out[B, D_MODEL, 1, 1] fp16

Architecture (per opf MLPBlock contract):
  pack1: Conv2d(D_MODEL → 2*D_FF)  + bias       [SwiGLU gate+linear]
  glu_clip / lin_clip / sigmoid / mul / (lin+1) [SwiGLU activation]
  pack2: Conv2d(D_FF → D_MODEL)    + bias       [output projection]
  NOTE: NO topk_w mul, NO sum-over-K. Caller does that.

Usage:
  python build_pf_per_expert.py --expert-id 0 [--quant fp16|int8]
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS   = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
ART_DIR   = REPO_ROOT / "emilio" / "conv-ane"

D_MODEL = 640
D_FF    = 640
B       = 64
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
COS_GATE = 0.985


def load_expert(expert_id: int) -> dict[str, np.ndarray]:
    from safetensors.torch import safe_open
    import torch
    keys = {
        "swi_w": "block.0.mlp.swiglu.weight",
        "swi_b": "block.0.mlp.swiglu.bias",
        "out_w": "block.0.mlp.out.weight",
        "out_b": "block.0.mlp.out.bias",
    }
    with safe_open(str(WEIGHTS), framework="pt") as f:
        full = {k: f.get_tensor(v).to(torch.float32).cpu().numpy()
                for k, v in keys.items()}
    return {
        "swi_w": full["swi_w"][expert_id],   # [D_MODEL, 2*D_FF]
        "swi_b": full["swi_b"][expert_id],   # [2*D_FF]
        "out_w": full["out_w"][expert_id],   # [D_FF, D_MODEL]  (per opf layout)
        "out_b": full["out_b"][expert_id],   # [D_MODEL]
    }


def build_expert_program(w: dict[str, np.ndarray], out_path: Path,
                         quant: str = "fp16") -> None:
    import torch, torch.nn as nn
    import coremltools as ct

    # Conv2d expects [Cout, Cin, 1, 1]
    pack1_w = w["swi_w"].T.reshape(2 * D_FF, D_MODEL, 1, 1).astype(np.float32)
    pack1_b = w["swi_b"].astype(np.float32)
    pack2_w = w["out_w"].T.reshape(D_MODEL, D_FF, 1, 1).astype(np.float32)
    pack2_b = w["out_b"].astype(np.float32)

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.pack1 = nn.Conv2d(D_MODEL, 2 * D_FF, 1, bias=True)
            self.pack1.weight = nn.Parameter(torch.from_numpy(pack1_w), requires_grad=False)
            self.pack1.bias   = nn.Parameter(torch.from_numpy(pack1_b), requires_grad=False)
            self.pack2 = nn.Conv2d(D_FF, D_MODEL, 1, bias=True)
            self.pack2.weight = nn.Parameter(torch.from_numpy(pack2_w), requires_grad=False)
            self.pack2.bias   = nn.Parameter(torch.from_numpy(pack2_b), requires_grad=False)

        def forward(self, x):
            h = self.pack1(x)                          # [B, 2*D_FF, 1, 1]
            glu = h[:, :D_FF, :, :]
            lin = h[:, D_FF:, :, :]
            glu = torch.clamp(glu, max=SWIGLU_LIMIT)
            lin = torch.clamp(lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)
            return self.pack2(mid)                     # [B, D_MODEL, 1, 1]

    mod = Expert().eval()
    x_ex = torch.zeros(B, D_MODEL, 1, 1, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex,))

    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x_in", shape=(B, D_MODEL, 1, 1), dtype=np.float16)],
        outputs=[ct.TensorType(name="y_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )

    if quant == "int8":
        from coremltools.optimize.coreml import (
            linear_quantize_weights, OpLinearQuantizerConfig, OptimizationConfig,
        )
        cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="LINEAR_SYMMETRIC", dtype="int8", granularity="per_channel"))
        m = linear_quantize_weights(m, config=cfg)

    tmp = out_path.with_suffix(".tmp.mlpackage")
    if tmp.exists(): shutil.rmtree(tmp)
    m.save(str(tmp))
    if out_path.exists(): shutil.rmtree(out_path)
    tmp.rename(out_path)


def torch_reference(w, x):
    import torch
    x = torch.from_numpy(x).to(torch.float32).reshape(B, D_MODEL)
    swi_w = torch.from_numpy(w["swi_w"]).to(torch.float32)   # [D_MODEL, 2*D_FF]
    swi_b = torch.from_numpy(w["swi_b"]).to(torch.float32)
    out_w = torch.from_numpy(w["out_w"]).to(torch.float32)   # [D_FF, D_MODEL]
    out_b = torch.from_numpy(w["out_b"]).to(torch.float32)
    h = x @ swi_w + swi_b                                     # [B, 2*D_FF]
    glu, lin = h[:, :D_FF], h[:, D_FF:]
    glu = torch.clamp(glu, max=SWIGLU_LIMIT)
    lin = torch.clamp(lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)  # [B, D_FF]
    y = mid @ out_w + out_b                                   # [B, D_MODEL]
    return y.numpy()


def cos_sim(a, b):
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def residency(pkg_path: Path) -> dict:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    compiled = pkg_path.with_suffix(".mlmodelc")
    if compiled.exists(): shutil.rmtree(compiled)
    compiled = Path(ct.utils.compile_model(str(pkg_path), str(compiled)))
    plan = MLComputePlan.load_from_path(path=str(compiled),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    program = plan.model_structure.program
    totals: dict[str, int] = {}
    big = []
    for fn_name, fn in program.functions.items():
        for op in fn.block.operations:
            d = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if d is None: continue
            dev = d.preferred_compute_device.__class__.__name__.replace("MLComputeDevice", "")
            totals[dev] = totals.get(dev, 0) + 1
            est = plan.get_estimated_cost_for_mlprogram_operation(op)
            cost = est.weight if est else 0
            if cost > 0.05:
                big.append((op.operator_name, dev, cost))
    big.sort(key=lambda x: -x[2])
    return {"totals": totals, "big": big}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-id", type=int, default=0)
    ap.add_argument("--quant", choices=["fp16", "int8"], default="fp16")
    args = ap.parse_args()

    print(f"[per-expert] coremltools probe  expert_id={args.expert_id}  quant={args.quant}")
    w = load_expert(args.expert_id)
    print(f"  shapes: swi_w={w['swi_w'].shape} out_w={w['out_w'].shape}")

    out_path = ART_DIR / f"PF_expert_{args.expert_id}_B{B}_{args.quant}.mlpackage"
    print(f"  building → {out_path.name}")
    build_expert_program(w, out_path, quant=args.quant)
    sz = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"  built  {sz:.2f} MB")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, D_MODEL, 1, 1)).astype(np.float32) * 0.5
    ref = torch_reference(w, x)
    import coremltools as ct
    m = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    pred = m.predict({"x_in": x.astype(np.float16)})["y_out"].reshape(B, D_MODEL)
    cos = cos_sim(ref, pred)
    print(f"  cos vs torch: {cos:.6f}  max|Δ|={np.abs(ref - pred).max():.4f}  "
          f"[{'PASS' if cos >= COS_GATE else 'FAIL'}]")
    if cos < COS_GATE:
        print("FAIL — quality gate not met"); sys.exit(1)

    r = residency(out_path)
    print(f"  residency: {r['totals']}")
    for n, dev, c in r["big"]:
        print(f"    {n:30s} -> {dev}  cost_weight={c:.4f}")

    print(f"\n  saved as: {out_path.name}")
    print(f"  next: xcrun coremlcompiler compile {out_path.name} {ART_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
