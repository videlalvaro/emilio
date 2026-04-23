"""T3.5a-fix: Build expert_000 with the ANE-friendly [1, C, 1, S] layout
and verify INT4 ANE residency.

The conv1×1 op is identical regardless of where the "batch" axis lives,
but ANE's planner reportedly prefers a single-batch tensor with the
sequence/token axis in the W (4th) position. The original T3 build used
[B=64, C, 1, 1], which the ane-validator probe showed lands on CPU even
at INT4. This rebuilds with [1, C, 1, S=64] and re-runs the same INT4
quant + cosine + (separate) ANE residency probe.

Run:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
    python/privacy/build_pf_expert000_layoutfix.py
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
DST_FP16 = REPO_ROOT / "emilio" / "conv-ane" / "PF_expert000_layoutfix_fp16.mlpackage"
DST_INT4 = REPO_ROOT / "emilio" / "conv-ane" / "PF_expert000_layoutfix_int4.mlpackage"

D_MODEL = 640
D_FF = 640
S_LEN = 64
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
EID = 0


def main() -> None:
    import torch
    import torch.nn as nn
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    from safetensors.torch import safe_open

    print(f"[layoutfix] python: ct {ct.__version__}")

    # 1) Load weights for expert 0.
    with safe_open(str(WEIGHTS), framework="pt") as f:
        sw_w = f.get_tensor("block.0.mlp.swiglu.weight").to(torch.float32)[EID]
        sw_b = f.get_tensor("block.0.mlp.swiglu.bias").to(torch.float32)[EID]
        ou_w = f.get_tensor("block.0.mlp.out.weight").to(torch.float32)[EID]
        ou_b = f.get_tensor("block.0.mlp.out.bias").to(torch.float32)[EID]

    # opf layout is [in, out]; nn.Conv2d wants [out, in, 1, 1].
    mlp1_w = sw_w.t().contiguous().reshape(2 * D_FF, D_MODEL, 1, 1)
    mlp2_w = ou_w.t().contiguous().reshape(D_MODEL, D_FF, 1, 1)

    class ExpertConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp1 = nn.Conv2d(D_MODEL, 2 * D_FF, 1, bias=True)
            self.mlp1.weight = nn.Parameter(mlp1_w, requires_grad=False)
            self.mlp1.bias = nn.Parameter(sw_b, requires_grad=False)
            self.mlp2 = nn.Conv2d(D_FF, D_MODEL, 1, bias=True)
            self.mlp2.weight = nn.Parameter(mlp2_w, requires_grad=False)
            self.mlp2.bias = nn.Parameter(ou_b, requires_grad=False)

        def forward(self, x_in):
            h = self.mlp1(x_in)
            g = torch.clamp(h[:, :D_FF, :, :], max=SWIGLU_LIMIT)
            l = torch.clamp(h[:, D_FF:, :, :], min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            return self.mlp2(g * torch.sigmoid(SWIGLU_ALPHA * g) * (l + 1.0))

    mod = ExpertConv().eval()

    # 2) Trace with the CORRECTED layout [1, C, 1, S].
    x_ex = torch.zeros(1, D_MODEL, 1, S_LEN, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex,))

    print("[layoutfix] converting FP16...")
    t0 = time.perf_counter()
    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x_in", shape=(1, D_MODEL, 1, S_LEN),
                              dtype=np.float16)],
        outputs=[ct.TensorType(name="x_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    print(f"[layoutfix] convert wall: {time.perf_counter()-t0:.1f}s")

    if DST_FP16.exists():
        shutil.rmtree(DST_FP16)
    m.save(str(DST_FP16))
    sz_fp16 = sum(f.stat().st_size for f in DST_FP16.rglob("*") if f.is_file()) / 1e6
    print(f"[layoutfix] FP16 saved → {DST_FP16.name} ({sz_fp16:.2f} MB)")

    # 3) INT4 quantize with the smoke-winning recipe (per_block g32).
    print("[layoutfix] INT4 grouped (g=32)...")
    t0 = time.perf_counter()
    qcfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", weight_threshold=0, dtype="int4",
            granularity="per_block", block_size=32))
    base = ct.models.MLModel(str(DST_FP16), compute_units=ct.ComputeUnit.CPU_AND_NE)
    quant = cto.linear_quantize_weights(base, config=qcfg)
    if DST_INT4.exists():
        shutil.rmtree(DST_INT4)
    quant.save(str(DST_INT4))
    sz_int4 = sum(f.stat().st_size for f in DST_INT4.rglob("*") if f.is_file()) / 1e6
    print(f"[layoutfix] INT4 saved → {DST_INT4.name} ({sz_int4:.2f} MB) "
          f"in {time.perf_counter()-t0:.1f}s")

    # 4) Cosine FP16 vs INT4 on a real input from the goldens.
    g_path = REPO_ROOT / "python" / "privacy" / "out" / "pf_layer0_moe.npz"
    if g_path.exists():
        g = np.load(g_path)
        norm = g["mlp_norm_out"].reshape(-1, D_MODEL).astype(np.float16)
        x = np.zeros((1, D_MODEL, 1, S_LEN), dtype=np.float16)
        # Take first 64 norm-tokens as input slots.
        x[0, :, 0, :] = norm[:S_LEN].T
        feed = {"x_in": x}
        m_fp = ct.models.MLModel(str(DST_FP16), compute_units=ct.ComputeUnit.CPU_AND_NE)
        m_q = ct.models.MLModel(str(DST_INT4), compute_units=ct.ComputeUnit.CPU_AND_NE)
        of = m_fp.predict(feed)["x_out"].astype(np.float32)
        oq = m_q.predict(feed)["x_out"].astype(np.float32)
        af = of.ravel(); aq = oq.ravel()
        c = float(af @ aq / (np.linalg.norm(af) * np.linalg.norm(aq) + 1e-12))
        print(f"[layoutfix] cosine fp16-vs-int4: {c:.6f}")

    print(f"\n[layoutfix] artifacts ready for ane-validator:")
    print(f"  FP16: {DST_FP16}")
    print(f"  INT4: {DST_INT4}")


if __name__ == "__main__":
    main()
