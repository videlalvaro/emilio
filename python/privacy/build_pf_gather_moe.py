"""Dynamic-gather MoE pack: real top-4 architecture on ANE.

For each row b in [B], gather pack1 + pack2 weights for its 4 active experts,
run SwiGLU, weight-combine. Single mlpackage, dynamic indices per call.

Inputs:
  x_in     [B, D_MODEL] fp16  — RMSNorm output (post-norm activations)
  idx      [B, K]       int32 — top-4 expert indices per row
  topk_w   [B, K]       fp16  — softmax(top-k logits)/K, already includes the
                                /TOPK factor (matches pf_moe_goldens schema)

Output:
  x_out    [B, D_MODEL] fp16

Cite: BOOK_ANALYSIS Stepanov §11 (algebraic equivalence — top-k masked sum
== sum-over-active-only) + Knuth Vol 3 §6 (sparse access pattern).

opf MLPBlock multiplies the combine by EXPERTS_PER_TOKEN (=4) outside the
block. We follow that contract: this op returns the sum BEFORE the *4.
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
TMP_DIR = ART_DIR / "_pf_gather_tmp"

D_MODEL = 640
D_FF = 640
N_EXPERTS = 128
TOPK = 4
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
B = 64
COS_GATE = 0.985


def load_weights() -> dict:
    from safetensors.torch import safe_open
    import torch
    keys = {
        "swiglu_w": "block.0.mlp.swiglu.weight",   # [128, 640, 1280]
        "swiglu_b": "block.0.mlp.swiglu.bias",     # [128, 1280]
        "out_w":    "block.0.mlp.out.weight",      # [128, 640, 640]
        "out_b":    "block.0.mlp.out.bias",        # [128, 640]
    }
    out = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        for s, full in keys.items():
            out[s] = f.get_tensor(full).to(torch.float32).cpu().numpy()
    return out


def build_program(packed: dict, out_path: Path) -> None:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    # Layout per expert e:
    #   pack1[e] : [2*D_FF, D_MODEL]   (SwiGLU pre-projection, transposed for matmul)
    #   pack1_b[e]: [2*D_FF]
    #   pack2[e] : [D_MODEL, D_FF]     (down-projection)
    #   pack2_b[e]: [D_MODEL]
    swi_w = packed["swiglu_w"]   # [E, D_M, 2*D_FF]
    swi_b = packed["swiglu_b"]   # [E, 2*D_FF]
    out_w = packed["out_w"]      # [E, D_FF, D_M]
    out_b = packed["out_b"]      # [E, D_M]

    pack1_w = np.transpose(swi_w, (0, 2, 1)).astype(np.float16)  # [E, 2*D_FF, D_M]
    pack1_b = swi_b.astype(np.float16)                           # [E, 2*D_FF]
    pack2_w = np.transpose(out_w, (0, 2, 1)).astype(np.float16)  # [E, D_M, D_FF]
    pack2_b = out_b.astype(np.float16)                           # [E, D_M]

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(B, D_MODEL), dtype=types.fp16),
        mb.TensorSpec(shape=(B, TOPK), dtype=types.int32),
        mb.TensorSpec(shape=(B, TOPK), dtype=types.fp16),
    ], opset_version=ct.target.iOS18)
    def prog(x_in, idx, topk_w):
        # Constants
        W1 = mb.const(val=pack1_w, name="W1")     # [E, 2*D_FF, D_M]
        b1 = mb.const(val=pack1_b, name="b1")     # [E, 2*D_FF]
        W2 = mb.const(val=pack2_w, name="W2")     # [E, D_M, D_FF]
        b2 = mb.const(val=pack2_b, name="b2")     # [E, D_M]

        # Gather expert weights for each row's top-K experts
        g_W1 = mb.gather(x=W1, indices=idx, axis=0,
                         validate_indices=False, name="g_W1")  # [B, K, 2*D_FF, D_M]
        g_b1 = mb.gather(x=b1, indices=idx, axis=0,
                         validate_indices=False, name="g_b1")  # [B, K, 2*D_FF]
        g_W2 = mb.gather(x=W2, indices=idx, axis=0,
                         validate_indices=False, name="g_W2")  # [B, K, D_M, D_FF]
        g_b2 = mb.gather(x=b2, indices=idx, axis=0,
                         validate_indices=False, name="g_b2")  # [B, K, D_M]

        # Reshape x to broadcast across K experts:
        # x_in: [B, D_M] -> [B, 1, D_M, 1]
        x_b1 = mb.reshape(x=x_in, shape=(B, 1, D_MODEL, 1), name="x_b1")
        # h_pre = matmul(g_W1, x) -> [B, K, 2*D_FF, 1]
        h_pre = mb.matmul(x=g_W1, y=x_b1, name="h_pre")
        h_pre = mb.reshape(x=h_pre, shape=(B, TOPK, 2 * D_FF), name="h_pre_3d")
        h_pre = mb.add(x=h_pre, y=g_b1, name="h_add_b1")  # [B, K, 2*D_FF]

        # Split into glu / lin halves
        glu = mb.slice_by_index(x=h_pre, begin=[0, 0, 0],
                                end=[B, TOPK, D_FF], name="glu")
        lin = mb.slice_by_index(x=h_pre, begin=[0, 0, D_FF],
                                end=[B, TOPK, 2 * D_FF], name="lin")
        glu = mb.clip(x=glu, alpha=np.float16(-1e4), beta=np.float16(SWIGLU_LIMIT),
                      name="glu_clip")
        lin = mb.clip(x=lin, alpha=np.float16(-SWIGLU_LIMIT), beta=np.float16(SWIGLU_LIMIT),
                      name="lin_clip")

        gated = mb.mul(x=glu, y=mb.const(val=np.float16(SWIGLU_ALPHA)),
                       name="alpha_glu")
        sig = mb.sigmoid(x=gated, name="sig")
        glu_out = mb.mul(x=glu, y=sig, name="glu_sig")
        one = mb.const(val=np.float16(1.0))
        lin_p1 = mb.add(x=lin, y=one, name="lin_p1")
        mid = mb.mul(x=glu_out, y=lin_p1, name="mid")  # [B, K, D_FF]

        # Down-projection: y_per_k = matmul(g_W2, mid)
        # mid: [B, K, D_FF] -> [B, K, D_FF, 1]
        mid_4d = mb.reshape(x=mid, shape=(B, TOPK, D_FF, 1), name="mid_4d")
        y_per_k = mb.matmul(x=g_W2, y=mid_4d, name="y_per_k")  # [B, K, D_M, 1]
        y_per_k = mb.reshape(x=y_per_k, shape=(B, TOPK, D_MODEL),
                             name="y_per_k_3d")
        y_per_k = mb.add(x=y_per_k, y=g_b2, name="y_add_b2")  # [B, K, D_M]

        # Weight by topk_w and reduce-sum over K
        gw = mb.reshape(x=topk_w, shape=(B, TOPK, 1), name="gw_r")
        weighted = mb.mul(x=y_per_k, y=gw, name="weighted")
        x_out = mb.reduce_sum(x=weighted, axes=[1], keep_dims=False,
                              name="x_out")
        return x_out

    m = ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    if out_path.exists():
        shutil.rmtree(out_path)
    m.save(str(out_path))


def torch_reference(packed, x, idx, topk_w):
    """Reference: gather + SwiGLU + combine in numpy/torch."""
    import torch
    swi_w = torch.from_numpy(packed["swiglu_w"]).float()  # [E, D, 2F]
    swi_b = torch.from_numpy(packed["swiglu_b"]).float()  # [E, 2F]
    out_w = torch.from_numpy(packed["out_w"]).float()     # [E, F, D]
    out_b = torch.from_numpy(packed["out_b"]).float()     # [E, D]
    x_t = torch.from_numpy(x).float()                     # [B, D]
    idx_t = torch.from_numpy(idx).long()                  # [B, K]
    w_t = torch.from_numpy(topk_w).float()                # [B, K]

    g_swi_w = swi_w[idx_t]   # [B, K, D, 2F]
    g_swi_b = swi_b[idx_t]   # [B, K, 2F]
    g_out_w = out_w[idx_t]   # [B, K, F, D]
    g_out_b = out_b[idx_t]   # [B, K, D]

    # h = einsum(B,D ; B,K,D,2F -> B,K,2F)
    h = torch.einsum("bd,bkdf->bkf", x_t, g_swi_w) + g_swi_b
    glu = h[..., :D_FF].clamp(max=SWIGLU_LIMIT)
    lin = h[..., D_FF:].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)

    # y_per_k = einsum(B,K,F ; B,K,F,D -> B,K,D)
    y_per_k = torch.einsum("bkf,bkfd->bkd", mid, g_out_w) + g_out_b
    out = (y_per_k * w_t.unsqueeze(-1)).sum(dim=1)        # [B, D]
    return out.numpy()


def validate(pkg_path: Path, packed: dict) -> tuple[float, float]:
    import coremltools as ct
    z = np.load(MOE_GOLDEN)
    x_all = z["mlp_norm_out"].reshape(-1, D_MODEL).astype(np.float32)
    idx_all = z["topk_indices"].reshape(-1, TOPK).astype(np.int32)
    w_all = z["topk_weights"].reshape(-1, TOPK).astype(np.float32)
    x = x_all[:B]
    idx = idx_all[:B]
    topk_w = w_all[:B]

    m = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.ALL)
    feed = {
        "x_in": x.astype(np.float16),
        "idx": idx.astype(np.int32),
        "topk_w": topk_w.astype(np.float16),
    }
    pred = m.predict(feed)["x_out"].astype(np.float32).reshape(B, D_MODEL)
    ref = torch_reference(packed, x, idx, topk_w)
    cos = float((pred * ref).sum() /
                (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-30))
    max_abs = float(np.abs(pred - ref).max())
    return cos, max_abs


def report_residency(pkg_path: Path) -> dict:
    import coremltools as ct
    compiled = pkg_path.with_suffix(".mlmodelc")
    if compiled.exists():
        shutil.rmtree(compiled)
    compiled = Path(ct.utils.compile_model(str(pkg_path), str(compiled)))
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    from collections import Counter
    counts: Counter = Counter()
    big_ops = []
    for op in plan.model_structure.program.functions["main"].block.operations:
        info = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if info is None:
            continue
        dev = type(info.preferred_compute_device).__name__
        tag = "ANE" if "Neural" in dev else ("GPU" if "GPU" in dev else "CPU")
        counts[tag] += 1
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        cw = cost.weight if cost else None
        if cw and cw > 0.01:
            big_ops.append((op.operator_name, tag, cw))
    print(f"  totals: {dict(counts)}")
    for n, t, c in sorted(big_ops, key=lambda x: -x[2])[:10]:
        print(f"    {n:35s} -> {t}   cost_weight={c:.4f}")
    return dict(counts)


def main() -> int:
    assert "Xcode.app" in sys.executable, sys.executable
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["fp16", "int8"], default="fp16",
                    help="Initial run: fp16 only (gather + INT8 may not compose)")
    args = ap.parse_args()

    import coremltools as ct
    print(f"[gather-moe] coremltools {ct.__version__}  B={B}  TOPK={TOPK}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    packed = load_weights()
    fp16_path = TMP_DIR / f"PF_gather_moe_B{B}_fp16.mlpackage"

    if not fp16_path.exists():
        print(f"[gather-moe] building FP16 program...")
        t0 = time.perf_counter()
        build_program(packed, fp16_path)
        sz = sum(f.stat().st_size for f in fp16_path.rglob("*") if f.is_file())
        print(f"  built in {time.perf_counter()-t0:.1f}s  {sz/1024/1024:.2f} MB")
    else:
        print(f"[gather-moe] reusing {fp16_path}")

    print("[gather-moe] validating cos vs torch reference...")
    cos, ma = validate(fp16_path, packed)
    status = "PASS" if cos >= COS_GATE else "FAIL"
    print(f"  cos={cos:.6f}  max|Δ|={ma:.4f}  [{status}]")

    print("[gather-moe] residency:")
    counts = report_residency(fp16_path)

    if args.quant == "int8":
        print("[gather-moe] INT8 per_channel quant...")
        import coremltools.optimize.coreml as cto
        cfg = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", weight_threshold=0,
            dtype="int8", granularity="per_channel",
        )
        src = ct.models.MLModel(str(fp16_path),
                                compute_units=ct.ComputeUnit.CPU_AND_NE)
        q = cto.linear_quantize_weights(
            src, config=cto.OptimizationConfig(global_config=cfg))
        int8_path = ART_DIR / f"PF_gather_moe_B{B}_int8.mlpackage"
        if int8_path.exists():
            shutil.rmtree(int8_path)
        q.save(str(int8_path))
        sz = sum(f.stat().st_size for f in int8_path.rglob("*") if f.is_file())
        print(f"  INT8 saved: {int8_path.name}  {sz/1024/1024:.2f} MB")
        cos, ma = validate(int8_path, packed)
        status = "PASS" if cos >= COS_GATE else "FAIL"
        print(f"  INT8 cos={cos:.6f}  max|Δ|={ma:.4f}  [{status}]")
        report_residency(int8_path)
    else:
        # Save fp16 too as the artifact
        fp16_final = ART_DIR / f"PF_gather_moe_B{B}_fp16.mlpackage"
        if fp16_final.exists():
            shutil.rmtree(fp16_final)
        shutil.copytree(fp16_path, fp16_final)
        print(f"  saved as: {fp16_final.name}")

    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
