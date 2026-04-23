"""T3d: full 128-expert layer-0 multifunction pack at FP16.

Sequence:
  Phase 1 — Per-expert build with on-disk checkpointing (skip if exists).
  Phase 2 — Smoke: assemble a 1-function MFD pack from expert_0 and verify load.
            (ANE-validator probe runs separately on this artifact.)
  Phase 3 — Assemble all 128 into PF_experts128_layer0.mlpackage.
  Phase 4 — Validate ALL 128 functions independently (cos >= 0.97).

Per gatekeeper review: fp16 is intentionally above the 96 MB ANE cliff —
this is the reference baseline. T5 will INT4-quantize.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
MOE_GOLDEN = REPO_ROOT / "python" / "privacy" / "out" / "pf_layer0_moe.npz"
TMP_DIR = REPO_ROOT / "emilio" / "conv-ane" / "_pf_expert_cache"
SMOKE_PKG = REPO_ROOT / "emilio" / "conv-ane" / "PF_experts1_smoke.mlpackage"
OUT_PKG = REPO_ROOT / "emilio" / "conv-ane" / "PF_experts128_layer0.mlpackage"

D_MODEL = 640
D_FF = 640
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
TRACE_N = 64
NUM_EXPERTS = 128


def load_packed_experts() -> dict[str, np.ndarray]:
    from safetensors.torch import safe_open
    import torch as _t
    keys = {
        "swiglu_w": "block.0.mlp.swiglu.weight",
        "swiglu_b": "block.0.mlp.swiglu.bias",
        "out_w":    "block.0.mlp.out.weight",
        "out_b":    "block.0.mlp.out.bias",
    }
    out: dict[str, np.ndarray] = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        for k, v in keys.items():
            out[k] = f.get_tensor(v).to(_t.float32).cpu().numpy()
    return out


def build_expert_module(packed: dict[str, np.ndarray], eid: int):
    import torch
    import torch.nn as nn

    mlp1_w_in = packed["swiglu_w"][eid]   # [640, 1280] (in, out)
    mlp1_b = packed["swiglu_b"][eid]
    mlp2_w_in = packed["out_w"][eid]      # [640, 640]
    mlp2_b = packed["out_b"][eid]

    class ExpertConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp1 = nn.Conv2d(D_MODEL, 2 * D_FF, 1, bias=True)
            self.mlp1.weight = nn.Parameter(
                torch.from_numpy(mlp1_w_in).t().contiguous()
                     .reshape(2 * D_FF, D_MODEL, 1, 1).to(torch.float32),
                requires_grad=False,
            )
            self.mlp1.bias = nn.Parameter(
                torch.from_numpy(mlp1_b).to(torch.float32), requires_grad=False)
            self.mlp2 = nn.Conv2d(D_FF, D_MODEL, 1, bias=True)
            self.mlp2.weight = nn.Parameter(
                torch.from_numpy(mlp2_w_in).t().contiguous()
                     .reshape(D_MODEL, D_FF, 1, 1).to(torch.float32),
                requires_grad=False,
            )
            self.mlp2.bias = nn.Parameter(
                torch.from_numpy(mlp2_b).to(torch.float32), requires_grad=False)

        def forward(self, x_in):
            h = self.mlp1(x_in)
            g = torch.clamp(h[:, :D_FF, :, :], max=SWIGLU_LIMIT)
            l = torch.clamp(h[:, D_FF:, :, :], min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            return self.mlp2(g * torch.sigmoid(SWIGLU_ALPHA * g) * (l + 1.0))

    return ExpertConv()


def torch_reference(packed: dict[str, np.ndarray], eid: int,
                    x_np: np.ndarray) -> np.ndarray:
    import torch
    x = torch.from_numpy(x_np).to(torch.float32)
    mlp1_w = torch.from_numpy(packed["swiglu_w"][eid]).to(torch.float32)
    mlp1_b = torch.from_numpy(packed["swiglu_b"][eid]).to(torch.float32)
    mlp2_w = torch.from_numpy(packed["out_w"][eid]).to(torch.float32)
    mlp2_b = torch.from_numpy(packed["out_b"][eid]).to(torch.float32)
    h = x @ mlp1_w + mlp1_b
    g, l = h[:, :D_FF], h[:, D_FF:]
    g = torch.clamp(g, max=SWIGLU_LIMIT)
    l = torch.clamp(l, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return ((g * torch.sigmoid(SWIGLU_ALPHA * g) * (l + 1.0)) @ mlp2_w + mlp2_b).numpy()


def build_single_expert(packed, eid: int, dst: Path) -> None:
    """Convert one expert to .mlpackage. Skip if already exists."""
    import torch, coremltools as ct
    if dst.exists():
        return
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
    tmp = dst.with_suffix(".tmp.mlpackage")
    if tmp.exists():
        shutil.rmtree(tmp)
    m.save(str(tmp))
    tmp.rename(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Force rebuild of final pack (per-expert cache kept).")
    ap.add_argument("--purge-cache", action="store_true",
                    help="Delete per-expert cache and rebuild from scratch.")
    ap.add_argument("--smoke-only", action="store_true",
                    help="Stop after Phase 2 (1-function MFD smoke).")
    args = ap.parse_args()

    if not MOE_GOLDEN.exists():
        raise SystemExit(f"Run pf_moe_goldens.py first: {MOE_GOLDEN}")
    if OUT_PKG.exists() and not args.force and not args.smoke_only:
        raise SystemExit(f"{OUT_PKG} exists. Use --force.")

    print(f"[t3d] python: {sys.executable}")
    import coremltools as ct
    print(f"[t3d] coremltools {ct.__version__}")

    if args.purge_cache and TMP_DIR.exists():
        print(f"[t3d] purging {TMP_DIR}")
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Load weights once.
    print("[t3d] loading packed expert tensors")
    packed = load_packed_experts()
    print(f"  swiglu_w: {packed['swiglu_w'].shape}")

    # ── Phase 1: per-expert build with checkpoint ─────────────────────────
    t0 = time.time()
    last_target = 1 if args.smoke_only else NUM_EXPERTS
    for eid in range(last_target):
        dst = TMP_DIR / f"expert_{eid:03d}.mlpackage"
        if dst.exists():
            print(f"  [{eid:3d}/{last_target}] cached")
            continue
        s = time.time()
        build_single_expert(packed, eid, dst)
        elapsed = time.time() - s
        print(f"  [{eid:3d}/{last_target}] built in {elapsed:.1f}s")
    print(f"[t3d] phase 1 done in {time.time()-t0:.1f}s")

    # ── Phase 2: single-expert MFD smoke ──────────────────────────────────
    print("[t3d] phase 2: 1-function MFD smoke from expert_0")
    if SMOKE_PKG.exists():
        shutil.rmtree(SMOKE_PKG)
    desc1 = ct.utils.MultiFunctionDescriptor()
    desc1.add_function(str(TMP_DIR / "expert_000.mlpackage"),
                       src_function_name="main",
                       target_function_name="expert_0")
    desc1.default_function_name = "expert_0"
    ct.utils.save_multifunction(desc1, str(SMOKE_PKG))
    smoke_size = sum(p.stat().st_size for p in SMOKE_PKG.rglob("*") if p.is_file())
    print(f"  smoke pack: {SMOKE_PKG.name} ({smoke_size/1024**2:.2f} MB)")

    # Quick load + predict on smoke pack
    z = np.load(MOE_GOLDEN, allow_pickle=False)
    norm_all = z["mlp_norm_out"].reshape(-1, D_MODEL)[:TRACE_N].astype(np.float32)
    m_smoke = ct.models.MLModel(str(SMOKE_PKG), function_name="expert_0",
                                compute_units=ct.ComputeUnit.ALL)
    out = m_smoke.predict({"x_in": norm_all.astype(np.float16)
                                          .reshape(TRACE_N, D_MODEL, 1, 1)})
    pred = out["x_out"].astype(np.float32).reshape(TRACE_N, D_MODEL)
    ref = torch_reference(packed, 0, norm_all)
    cos_smoke = float((pred * ref).sum() /
                      (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-30))
    print(f"  smoke cos vs ref: {cos_smoke:.6f}")
    if cos_smoke < 0.97:
        print("[t3d] smoke FAIL — aborting")
        return 1

    if args.smoke_only:
        print(f"[t3d] smoke-only: stopping after Phase 2.")
        print(f"[t3d] smoke artifact: {SMOKE_PKG} ({smoke_size/1024**2:.2f} MB)")
        return 0

    # ── Phase 3: assemble all 128 ─────────────────────────────────────────
    print(f"[t3d] phase 3: assembling all {NUM_EXPERTS} experts")
    t1 = time.time()
    desc = ct.utils.MultiFunctionDescriptor()
    for eid in range(NUM_EXPERTS):
        desc.add_function(str(TMP_DIR / f"expert_{eid:03d}.mlpackage"),
                          src_function_name="main",
                          target_function_name=f"expert_{eid:03d}")
    desc.default_function_name = "expert_000"
    if OUT_PKG.exists():
        shutil.rmtree(OUT_PKG)
    ct.utils.save_multifunction(desc, str(OUT_PKG))
    pack_size = sum(p.stat().st_size for p in OUT_PKG.rglob("*") if p.is_file())
    print(f"  {OUT_PKG.name} ({pack_size/1024**2:.2f} MB) "
          f"assembled in {time.time()-t1:.1f}s")

    # ── Phase 4: validate ALL 128 ─────────────────────────────────────────
    print(f"[t3d] phase 4: per-function cosine validation")
    t2 = time.time()
    fails: list[tuple[int, float]] = []
    cosines: list[float] = []
    for eid in range(NUM_EXPERTS):
        m = ct.models.MLModel(str(OUT_PKG), function_name=f"expert_{eid:03d}",
                              compute_units=ct.ComputeUnit.ALL)
        out = m.predict({"x_in": norm_all.astype(np.float16)
                                         .reshape(TRACE_N, D_MODEL, 1, 1)})
        pred = out["x_out"].astype(np.float32).reshape(TRACE_N, D_MODEL)
        ref = torch_reference(packed, eid, norm_all)
        cos = float((pred * ref).sum() /
                    (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-30))
        cosines.append(cos)
        if cos < 0.97:
            fails.append((eid, cos))
        if eid % 16 == 0:
            print(f"  expert_{eid:03d}: cos={cos:.6f}")
    print(f"[t3d] phase 4 done in {time.time()-t2:.1f}s")

    arr = np.array(cosines)
    print(f"[t3d] cosine summary: worst={arr.min():.6f} median={np.median(arr):.6f} "
          f"mean={arr.mean():.6f} best={arr.max():.6f}")
    if fails:
        print(f"[t3d] FAIL: {len(fails)} experts < 0.97:")
        for e, c in fails[:10]:
            print(f"   expert_{e:03d}: cos={c:.6f}")
        return 1
    print(f"[t3d] PASS: all {NUM_EXPERTS} experts >= 0.97")
    print(f"[t3d] artifact: {OUT_PKG} ({pack_size/1024**2:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
