"""T3.5-Iverson v2: split Pack-2 into N sub-convs to fit ANE Cin limit.

Diagnosis from v1:
  * FP16 packing: cos = 1.000000 (structure correct).
  * INT8 per_channel: cos = 0.9999 PASS.
  * INT4 (all block sizes tried): cos ~ 0.26 FAIL — data-free INT4 quant of
    the 640x81920 Pack-2 weight is structurally too lossy.
  * INT8 artifact placement: pack1=ANE, pack2=CPU (Cin=81920 exceeds ANE
    compiler's max-Cin threshold per single conv op).

Stepanov Exp 19 fix: split the expert axis into N chunks and reduce the
down-projection as Σ_p Σ_{e∈pack_p}. Each sub-pack has Cin = (128/N)*D_FF
channels — in the ANE-friendly range when N ≥ 4.

Design (N=4):
  Pack-1 (unchanged): [128*2*D_FF, D_MODEL, 1, 1] single big conv.
                      pack1 already proven ANE at INT8.
  SwiGLU + gate-mask pointwise (single op chain).
  Split along expert axis into 4 chunks of 32 experts each.
  For each chunk p: sub-conv with weights [D_MODEL, 32*D_FF, 1, 1].
  Sum the 4 sub-conv outputs; add gate-weighted bias.

Each Pack-2 sub-conv: Cin = 32*640 = 20480 (4x below the failing 81920).
Per-op INT8 weight = 640 * 20480 = 13 MB -- well inside ANE sweet spot.
Total INT8 weights: pack1 (105 MB INT8→52 MB) + 4x13 MB = 104 MB.
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
ART_DIR = REPO_ROOT / "emilio" / "conv-ane"
TMP_DIR = ART_DIR / "_pf_packed_v2_tmp"

D_MODEL = 640
D_FF = 640
N_EXPERTS = 128
TOPK = 4
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
TRACE_N = 64  # overridable via --trace-n (Stepanov Exp.19: re-batch reduction)
COS_GATE_INTERMEDIATE = 0.985


def _check_interp() -> None:
    import coremltools as ct  # noqa: F401
    assert "Xcode.app" in sys.executable or ct.__version__.startswith("9"), \
        f"Wrong interpreter: {sys.executable} ct={ct.__version__}"


def load_packed_weights() -> dict[str, np.ndarray]:
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


def build_packed_program_v2(packed: dict[str, np.ndarray], n_splits: int,
                            out_path: Path) -> None:
    """Build MoE with Pack-2 split into n_splits sub-convs.

    n_splits must divide N_EXPERTS (128).
    """
    import torch
    import torch.nn as nn
    import coremltools as ct

    assert N_EXPERTS % n_splits == 0
    experts_per_chunk = N_EXPERTS // n_splits
    F_DIM = D_FF
    G = N_EXPERTS

    swi_w = packed["swiglu_w"]   # [128, 640, 1280]
    swi_b = packed["swiglu_b"]   # [128, 1280]
    out_w = packed["out_w"]      # [128, 640, 640]
    out_b = packed["out_b"]      # [128, 640]

    swi_w_packed = np.stack(
        [swi_w[e].T for e in range(G)], axis=0
    ).reshape(G * 2 * F_DIM, D_MODEL, 1, 1).astype(np.float32)
    swi_b_packed = swi_b.reshape(G * 2 * F_DIM).astype(np.float32)

    # Pack-2 split along expert axis. For chunk p, experts e in
    # [p*experts_per_chunk, (p+1)*experts_per_chunk), weight is
    # concat along IN axis: [D_MODEL, experts_per_chunk * D_FF].
    pack2_weights = []
    for p in range(n_splits):
        start = p * experts_per_chunk
        end = (p + 1) * experts_per_chunk
        chunk_w = np.concatenate(
            [out_w[e].T for e in range(start, end)], axis=1
        ).reshape(D_MODEL, experts_per_chunk * F_DIM, 1, 1).astype(np.float32)
        pack2_weights.append(chunk_w)

    out_b_table = out_b.astype(np.float32)

    class PackedMoEv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.pack1 = nn.Conv2d(D_MODEL, G * 2 * F_DIM, 1, bias=True)
            self.pack1.weight = nn.Parameter(
                torch.from_numpy(swi_w_packed), requires_grad=False)
            self.pack1.bias = nn.Parameter(
                torch.from_numpy(swi_b_packed), requires_grad=False)
            self.pack2_chunks = nn.ModuleList([
                nn.Conv2d(experts_per_chunk * F_DIM, D_MODEL, 1, bias=False)
                for _ in range(n_splits)
            ])
            for i, conv in enumerate(self.pack2_chunks):
                conv.weight = nn.Parameter(
                    torch.from_numpy(pack2_weights[i]), requires_grad=False)
            # Gate-weighted per-expert bias projection.
            self.bias_proj = nn.Conv2d(G, D_MODEL, 1, bias=False)
            self.bias_proj.weight = nn.Parameter(
                torch.from_numpy(out_b_table.T)
                     .reshape(D_MODEL, G, 1, 1).contiguous(),
                requires_grad=False,
            )

        def forward(self, x_in, g_in):
            h = self.pack1(x_in)                          # [B, G*1280, 1, 1]
            B = h.shape[0]
            h = h.reshape(B, G, 2 * F_DIM, 1)
            glu = h[:, :, :F_DIM, :]
            lin = h[:, :, F_DIM:, :]
            glu = torch.clamp(glu, max=SWIGLU_LIMIT)
            lin = torch.clamp(lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)
            # Apply gate
            g4 = g_in.reshape(B, G, 1, 1)
            mid = mid * g4                                # [B, G, 640, 1]
            # Split mid along expert axis into n_splits chunks and down-proj each
            acc = None
            for p, conv in enumerate(self.pack2_chunks):
                s = p * experts_per_chunk
                e = (p + 1) * experts_per_chunk
                chunk = mid[:, s:e, :, :].reshape(
                    B, experts_per_chunk * F_DIM, 1, 1)
                part = conv(chunk)                        # [B, 640, 1, 1]
                acc = part if acc is None else acc + part
            bias_term = self.bias_proj(g_in.reshape(B, G, 1, 1))
            return acc + bias_term

    mod = PackedMoEv2().eval()
    x_ex = torch.zeros(TRACE_N, D_MODEL, 1, 1, dtype=torch.float32)
    g_ex = torch.zeros(TRACE_N, G, 1, 1, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex, g_ex))

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
    tmp_path = out_path.with_suffix(".tmp.mlpackage")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    m.save(str(tmp_path))
    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.rename(out_path)


def torch_reference_packed(packed, n_active, x_np, g_np):
    import torch
    G = n_active
    x = torch.from_numpy(x_np).to(torch.float32)
    g = torch.from_numpy(g_np).to(torch.float32)
    swi_w = torch.from_numpy(packed["swiglu_w"][:G]).to(torch.float32)
    swi_b = torch.from_numpy(packed["swiglu_b"][:G]).to(torch.float32)
    out_w = torch.from_numpy(packed["out_w"][:G]).to(torch.float32)
    out_b = torch.from_numpy(packed["out_b"][:G]).to(torch.float32)
    h = torch.einsum("bm,emf->bef", x, swi_w) + swi_b
    glu, lin = h[..., :D_FF], h[..., D_FF:]
    glu = torch.clamp(glu, max=SWIGLU_LIMIT)
    lin = torch.clamp(lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)
    y_e = torch.einsum("bef,efd->bed", mid, out_w) + out_b
    out = (g.unsqueeze(-1) * y_e).sum(dim=1)
    return out.numpy()


def ane_residency_summary(pkg_path: Path) -> str:
    import coremltools as ct
    try:
        from coremltools.models.compute_plan import MLComputePlan
        compiled = pkg_path.with_suffix(".mlmodelc")
        if compiled.exists():
            shutil.rmtree(compiled)
        compiled = Path(ct.utils.compile_model(str(pkg_path), str(compiled)))
        plan = MLComputePlan.load_from_path(
            path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
        program = plan.model_structure.program
        if program is None:
            return "?"
        out = []
        for fn_name, fn in program.functions.items():
            for op in fn.block.operations:
                opn = op.operator_name or ""
                if any(k in opn for k in ("conv", "linear", "matmul")):
                    try:
                        d = plan.get_compute_device_usage_for_mlprogram_operation(op)
                        n = d.preferred_compute_device.__class__.__name__ if d else "?"
                        tag = "ANE" if "Neural" in n else ("GPU" if "GPU" in n else "CPU")
                    except Exception:
                        tag = "?"
                    out.append(f"{opn}:{tag}")
        return " | ".join(out) if out else "?"
    except Exception as e:
        return f"err:{type(e).__name__}:{e}"


def quantize_int8_per_channel(src_path: Path, dst_path: Path) -> None:
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    src = ct.models.MLModel(str(src_path),
                            compute_units=ct.ComputeUnit.CPU_AND_NE)
    cfg = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", weight_threshold=0,
        dtype="int8", granularity="per_channel",
    )
    quant = cto.linear_quantize_weights(
        src, config=cto.OptimizationConfig(global_config=cfg))
    tmp = dst_path.with_suffix(".tmp.mlpackage")
    if tmp.exists():
        shutil.rmtree(tmp)
    quant.save(str(tmp))
    if dst_path.exists():
        shutil.rmtree(dst_path)
    tmp.rename(dst_path)


def package_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024**2)


def validate(pkg_path: Path, packed) -> tuple[float, float]:
    import coremltools as ct
    z = np.load(MOE_GOLDEN, allow_pickle=False)
    norm_all = z["mlp_norm_out"].reshape(-1, D_MODEL).astype(np.float32)[:TRACE_N]
    topk_idx = z["topk_indices"].reshape(-1, TOPK).astype(np.int64)[:TRACE_N]
    topk_w = z["topk_weights"].reshape(-1, TOPK).astype(np.float32)[:TRACE_N]
    g_dense = np.zeros((TRACE_N, N_EXPERTS), dtype=np.float32)
    for b in range(TRACE_N):
        for k in range(TOPK):
            g_dense[b, topk_idx[b, k]] = topk_w[b, k]
    m = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.ALL)
    feed = {
        "x_in": norm_all.reshape(TRACE_N, D_MODEL, 1, 1).astype(np.float16),
        "g_in": g_dense.reshape(TRACE_N, N_EXPERTS, 1, 1).astype(np.float16),
    }
    pred = m.predict(feed)["x_out"].astype(np.float32).reshape(TRACE_N, D_MODEL)
    ref = torch_reference_packed(packed, n_active=N_EXPERTS, x_np=norm_all,
                                 g_np=g_dense)
    cos = float((pred * ref).sum() /
                (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-30))
    max_abs = float(np.abs(pred - ref).max())
    return cos, max_abs


def main() -> int:
    _check_interp()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=4,
                    help="Number of Pack-2 sub-convs (divides 128)")
    ap.add_argument("--also-int4", action="store_true",
                    help="Also try INT4 block32 on the split form (smaller matrices may work)")
    ap.add_argument("--trace-n", type=int, default=64,
                    help="Pack batch size (Stepanov re-batch test)")
    args = ap.parse_args()

    global TRACE_N
    TRACE_N = args.trace_n

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    N = args.n_splits
    print(f"[v2] n_splits={N}  experts_per_chunk={N_EXPERTS // N}  "
          f"pack2 sub-Cin={(N_EXPERTS // N) * D_FF}")

    import coremltools as ct
    print(f"[v2] coremltools {ct.__version__}")

    packed = load_packed_weights()

    fp16_pkg = TMP_DIR / f"PF_packed_v2_N{N}_B{TRACE_N}_fp16.mlpackage"
    if not fp16_pkg.exists():
        print(f"[v2] building FP16 split model ({fp16_pkg.name})")
        t0 = time.perf_counter()
        build_packed_program_v2(packed, n_splits=N, out_path=fp16_pkg)
        print(f"  built in {time.perf_counter()-t0:.1f}s, "
              f"{package_size_mb(fp16_pkg):.1f} MB")
    else:
        print(f"[v2] reusing {fp16_pkg}")

    print("[v2] validating FP16...")
    cos, ma = validate(fp16_pkg, packed)
    print(f"  FP16 cos={cos:.6f} max|Δ|={ma:.4f}")
    if cos < 0.999:
        print("  FP16 FAIL -> split packing bug; aborting")
        return 1

    print("[v2] ANE placement for FP16 split model:")
    devs = ane_residency_summary(fp16_pkg)
    print(f"  {devs}")

    int8_pkg = ART_DIR / f"PF_packed_iverson_v2_N{N}_B{TRACE_N}_int8.mlpackage"
    print(f"[v2] quantizing INT8 per_channel -> {int8_pkg.name}")
    quantize_int8_per_channel(fp16_pkg, int8_pkg)
    print(f"  size: {package_size_mb(int8_pkg):.2f} MB")

    cos, ma = validate(int8_pkg, packed)
    status = "PASS" if cos >= COS_GATE_INTERMEDIATE else "FAIL"
    print(f"  INT8 cos={cos:.6f} max|Δ|={ma:.4f} [{status}]")

    devs = ane_residency_summary(int8_pkg)
    print(f"  INT8 placement: {devs}")
    ane_ct = devs.count("ANE")
    cpu_ct = devs.count("CPU")
    print(f"  ops on ANE: {ane_ct}   on CPU: {cpu_ct}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
