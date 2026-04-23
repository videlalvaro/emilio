"""Build all 128 experts as separate mlpackages, then compile to mlmodelc.
Loads weights ONCE, then loops experts.

  --quant fp16 | int8     (default fp16)
  --start N --end M       (default 0..128)
  --skip-existing         (default True)

Output: emilio/conv-ane/PF_expert_{0..127}_B64_{quant}.mlmodelc
"""
from __future__ import annotations
import argparse, shutil, sys, time
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


def load_all(layer: int = 0) -> dict[str, np.ndarray]:
    from safetensors.torch import safe_open
    import torch
    keys = {
        "swi_w": f"block.{layer}.mlp.swiglu.weight",
        "swi_b": f"block.{layer}.mlp.swiglu.bias",
        "out_w": f"block.{layer}.mlp.out.weight",
        "out_b": f"block.{layer}.mlp.out.bias",
    }
    with safe_open(str(WEIGHTS), framework="pt") as f:
        return {k: f.get_tensor(v).to(torch.float32).cpu().numpy()
                for k, v in keys.items()}


def build_one(w_all, expert_id: int, quant: str, out_path: Path) -> None:
    import torch, torch.nn as nn
    import coremltools as ct

    swi_w = w_all["swi_w"][expert_id]
    swi_b = w_all["swi_b"][expert_id]
    out_w = w_all["out_w"][expert_id]
    out_b = w_all["out_b"][expert_id]

    pack1_w = swi_w.T.reshape(2 * D_FF, D_MODEL, 1, 1).astype(np.float32)
    pack1_b = swi_b.astype(np.float32)
    pack2_w = out_w.T.reshape(D_MODEL, D_FF, 1, 1).astype(np.float32)
    pack2_b = out_b.astype(np.float32)

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
            h = self.pack1(x)
            glu = h[:, :D_FF, :, :]
            lin = h[:, D_FF:, :, :]
            glu = torch.clamp(glu, max=SWIGLU_LIMIT)
            lin = torch.clamp(lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)
            return self.pack2(mid)

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


def compile_one(pkg_path: Path) -> Path:
    import coremltools as ct
    compiled = pkg_path.with_suffix(".mlmodelc")
    if compiled.exists(): shutil.rmtree(compiled)
    compiled = Path(ct.utils.compile_model(str(pkg_path), str(compiled)))
    return compiled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--quant", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end",   type=int, default=128)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--keep-pkg", action="store_true", default=False,
                    help="keep .mlpackage in addition to .mlmodelc")
    ap.add_argument("--no-compile", action="store_true", default=False,
                    help="only build .mlpackage, do not compile to .mlmodelc")
    args = ap.parse_args()

    print(f"[build_all] layer={args.layer} quant={args.quant} range=[{args.start},{args.end})")
    w_all = load_all(args.layer)
    print(f"  loaded weights")

    t0 = time.time()
    built = 0; skipped = 0; total_size = 0
    for eid in range(args.start, args.end):
        pkg = ART_DIR / f"PF_expert_L{args.layer}_{eid}_B{B}_{args.quant}.mlpackage"
        mlc = pkg.with_suffix(".mlmodelc")
        # Skip if both desired artifacts exist.
        want_pkg = args.keep_pkg or args.no_compile
        have_mlc_target = mlc.exists() or args.no_compile
        have_pkg_target = (not want_pkg) or pkg.exists()
        if args.skip_existing and have_mlc_target and have_pkg_target:
            skipped += 1
            base = mlc if mlc.exists() else pkg
            sz = sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
            total_size += sz
            continue
        build_one(w_all, eid, args.quant, pkg)
        if not args.no_compile:
            compile_one(pkg)
        if not args.keep_pkg and not args.no_compile:
            shutil.rmtree(pkg)
        base = mlc if mlc.exists() else pkg
        sz = sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
        total_size += sz
        built += 1
        if (eid - args.start + 1) % 8 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (eid - args.start + 1) * (args.end - eid - 1)
            print(f"  [{eid+1}/{args.end}]  built={built}  skipped={skipped}  "
                  f"elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    elapsed = time.time() - t0
    print(f"\n[build_all] DONE  built={built}  skipped={skipped}  "
          f"elapsed={elapsed:.1f}s  total_size={total_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
