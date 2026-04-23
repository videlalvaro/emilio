"""MoE compute-ceiling probe (BOOK_ANALYSIS Knuth Vol 3 §6 sparse access).

Hypothesis: the 76 ms MoE call is dense compute over 128 experts. Top-4 routing
implies 32× compute waste. A model that ONLY computes 4 experts should be
~32× faster IF compute is the floor. If it's still ~76 ms, then the floor is
fixed-cost (loader, weight DMA, layout) and gather-then-compute won't help.

Builds same graph as build_pf_packed_iverson_v2 but with N_EXPERTS=4 (static
first 4 experts; no correctness — pure latency probe).

Cite: BOOK_ANALYSIS.md Stepanov §11 algebraic equivalence — top-k masked sum
is equivalent to sum-over-active-only.
"""
from __future__ import annotations
import shutil
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
ART_DIR = REPO_ROOT / "emilio" / "conv-ane"
TMP_DIR = ART_DIR / "_pf_topk4_tmp"

D_MODEL = 640
D_FF = 640
G = 4              # <-- only 4 experts (top-k ceiling)
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
TRACE_N = 64


def main() -> int:
    import coremltools as ct
    import torch
    import torch.nn as nn
    from safetensors.torch import safe_open

    assert "Xcode.app" in sys.executable, sys.executable
    print(f"[topk4] coremltools {ct.__version__}  G={G}  B={TRACE_N}")

    keys = {
        "swiglu_w": "block.0.mlp.swiglu.weight",
        "swiglu_b": "block.0.mlp.swiglu.bias",
        "out_w":    "block.0.mlp.out.weight",
        "out_b":    "block.0.mlp.out.bias",
    }
    packed = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        for short, full in keys.items():
            packed[short] = f.get_tensor(full).to(torch.float32).cpu().numpy()

    # Take first G experts only.
    swi_w = packed["swiglu_w"][:G]   # [G, 640, 1280]
    swi_b = packed["swiglu_b"][:G]   # [G, 1280]
    out_w = packed["out_w"][:G]      # [G, 640, 640]
    out_b = packed["out_b"][:G]      # [G, 640]

    swi_w_packed = np.stack(
        [swi_w[e].T for e in range(G)], axis=0
    ).reshape(G * 2 * D_FF, D_MODEL, 1, 1).astype(np.float32)
    swi_b_packed = swi_b.reshape(G * 2 * D_FF).astype(np.float32)
    pack2_w = np.concatenate(
        [out_w[e].T for e in range(G)], axis=1
    ).reshape(D_MODEL, G * D_FF, 1, 1).astype(np.float32)
    bias_proj_w = (out_b.astype(np.float32).T
                   ).reshape(D_MODEL, G, 1, 1).copy()

    class TopK4MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.pack1 = nn.Conv2d(D_MODEL, G * 2 * D_FF, 1, bias=True)
            self.pack1.weight = nn.Parameter(torch.from_numpy(swi_w_packed),
                                             requires_grad=False)
            self.pack1.bias = nn.Parameter(torch.from_numpy(swi_b_packed),
                                           requires_grad=False)
            self.pack2 = nn.Conv2d(G * D_FF, D_MODEL, 1, bias=False)
            self.pack2.weight = nn.Parameter(torch.from_numpy(pack2_w),
                                             requires_grad=False)
            self.bias_proj = nn.Conv2d(G, D_MODEL, 1, bias=False)
            self.bias_proj.weight = nn.Parameter(torch.from_numpy(bias_proj_w),
                                                 requires_grad=False)

        def forward(self, x_in, g_in):
            h = self.pack1(x_in)                       # [B, G*1280, 1, 1]
            B = h.shape[0]
            h = h.reshape(B, G, 2 * D_FF, 1)
            glu = h[:, :, :D_FF, :]
            lin = h[:, :, D_FF:, :]
            glu = torch.clamp(glu, max=SWIGLU_LIMIT)
            lin = torch.clamp(lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)
            g4 = g_in.reshape(B, G, 1, 1)
            mid = mid * g4
            chunk = mid.reshape(B, G * D_FF, 1, 1)
            acc = self.pack2(chunk)
            return acc + self.bias_proj(g_in.reshape(B, G, 1, 1))

    mod = TopK4MoE().eval()
    x_ex = torch.zeros(TRACE_N, D_MODEL, 1, 1)
    g_ex = torch.zeros(TRACE_N, G, 1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex, g_ex))

    print("[topk4] converting...")
    t0 = time.perf_counter()
    m = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_in", shape=(TRACE_N, D_MODEL, 1, 1),
                          dtype=np.float16),
            ct.TensorType(name="g_in", shape=(TRACE_N, G, 1, 1),
                          dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="x_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fp16_path = TMP_DIR / f"PF_topk4_B{TRACE_N}_fp16.mlpackage"
    if fp16_path.exists():
        shutil.rmtree(fp16_path)
    m.save(str(fp16_path))
    print(f"  built in {time.perf_counter()-t0:.1f}s")

    import coremltools.optimize.coreml as cto
    cfg = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", weight_threshold=0,
        dtype="int8", granularity="per_channel",
    )
    src = ct.models.MLModel(str(fp16_path),
                            compute_units=ct.ComputeUnit.CPU_AND_NE)
    quant = cto.linear_quantize_weights(
        src, config=cto.OptimizationConfig(global_config=cfg))
    int8_path = ART_DIR / f"PF_topk4_B{TRACE_N}_int8.mlpackage"
    if int8_path.exists():
        shutil.rmtree(int8_path)
    quant.save(str(int8_path))
    sz = sum(f.stat().st_size for f in int8_path.rglob("*") if f.is_file())
    print(f"[topk4] INT8 saved: {int8_path.name}  {sz/1024/1024:.2f} MB")
    print(f"[topk4] expected ratio vs 128-expert pack: {4/128:.4f} ({128/4}× lighter)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
