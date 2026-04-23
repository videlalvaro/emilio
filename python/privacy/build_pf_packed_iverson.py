"""T3.5-Iverson: pack the entire MoE (all 128 experts) into a single fused
tensor op, per Iverson APL +.× (BOOK_ANALYSIS Exp 17) and Stepanov semigroup
reduction (Exp 19). Precedent: python/moe/gemma_packed_experts.py.

Rationale:
  Single-expert MoE artifacts (~0.5-2.5 MB) are STRUCTURALLY too small for
  the ANE planner — it pins them to CPU regardless of layout/quant because
  ANE round-trip > CPU runtime at that size (T3.5a-fix proved this).

  Reframe the MoE as one inner product. The op
      out[d] = Σ_e g_e * Σ_f W_out[e,d,f] * φ(Σ_m W_swi[e,f,m] * x[m])
  factors via linearity of the down projection:
      out[d] = Σ_{e,f} W_out[e,d,f] * (g_e * φ(...))
  which is one big +.× — one fused matmul, not 128 small ones.

  Implementation uses two Conv2d 1×1 ops:
    Pack-1: weights [128*2*D_FF, D_MODEL, 1, 1]  (all experts' SwiGLU stacked)
    Pack-2: weights [D_MODEL, 128*D_FF, 1, 1]    (all experts' down stacked)

  Total INT4 G=32 ≈ 78 MB (~52 + ~26). Both individual ops < 96 MB cliff,
  > 12 MB floor — squarely in the ANE sweet spot.

  Cost: 32× wasted FLOPs (compute all 128 experts, mask to top-4). ANE is
  bandwidth-bound (~100 GB/s, ~35 TOPS) so FLOPs are nearly free.

Gates (per gatekeeper tweaks):
  - Step A: G=16 FP16 *probe* (~26 MB) → ane-validator must report ANE
    placement BEFORE we commit to the full G=128 build.
  - Step B: full G=128 INT4 G=32 build using `linear_int4_grouped_g32` config.
  - Step C: golden-validator vs replay of `pf_layer0_moe.npz` top-4 routing
    must hit cos ≥ 0.985 intermediate.
  - Layout stays (B, C, 1, S) end-to-end. No transpose. Gate enters as a
    static-shape dense FP16 tensor + mb.mul (no gather/scatter).
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
ART_DIR = REPO_ROOT / "emilio" / "conv-ane"
TMP_DIR = ART_DIR / "_pf_packed_tmp"

# Privacy-filter MoE shapes (verified against opf source, layer 0)
D_MODEL = 640
D_FF = 640
N_EXPERTS = 128
TOPK = 4
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
TRACE_N = 64  # batched-token shape used by attn/expert artifacts

# Gates
COS_GATE_INTERMEDIATE = 0.985


def _check_interp() -> None:
    # Per gatekeeper tweak #4: must use Xcode python3 (coremltools 9 only there).
    import coremltools as ct  # noqa: F401
    assert (
        "Xcode.app" in sys.executable or ct.__version__.startswith("9")
    ), f"Wrong interpreter: {sys.executable} ct={ct.__version__}; need Xcode python3 + coremltools 9"


def load_packed_weights() -> dict[str, np.ndarray]:
    """Load layer-0 MoE weights from opf checkpoint (fp32)."""
    from safetensors.torch import safe_open
    import torch

    keys = {
        "swiglu_w": "block.0.mlp.swiglu.weight",  # [128, 640, 1280] (in,out)
        "swiglu_b": "block.0.mlp.swiglu.bias",    # [128, 1280]
        "out_w":    "block.0.mlp.out.weight",     # [128, 640, 640]  (in,out)
        "out_b":    "block.0.mlp.out.bias",       # [128, 640]
    }
    out: dict[str, np.ndarray] = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        for short, full in keys.items():
            out[short] = f.get_tensor(full).to(torch.float32).cpu().numpy()
    return out


def build_packed_program(packed: dict[str, np.ndarray], n_active_experts: int,
                         out_path: Path) -> "object":
    """Build a fused-MoE mlpackage covering the FIRST `n_active_experts` experts.

    Layout: (B=TRACE_N, C, 1, 1) end-to-end. Inputs:
      x_in:   [B, D_MODEL, 1, 1] fp16 — post-norm per-token vector
      g_in:   [B, n_active, 1, 1] fp16 — dense gate weights (top-4 nonzero)

    Output:
      x_out:  [B, D_MODEL, 1, 1] fp16
    """
    import torch
    import torch.nn as nn
    import coremltools as ct

    G = n_active_experts
    F_DIM = D_FF  # 640

    # Slice weights to first G experts (G=128 = full model)
    swi_w = packed["swiglu_w"][:G]   # [G, 640, 1280]  (in, out)
    swi_b = packed["swiglu_b"][:G]   # [G, 1280]
    out_w = packed["out_w"][:G]      # [G, 640, 640]   (in, out)
    out_b = packed["out_b"][:G]      # [G, 640]

    # ---- Pack-1: stack all experts' swiglu along OUT axis.
    # Per-expert nn.Conv2d weight is [out=2*D_FF, in=D_MODEL, 1, 1].
    # opf stores weight as [in=640, out=1280]; transpose to [out, in] then stack.
    # Final packed shape: [G * 2*D_FF, D_MODEL, 1, 1].
    swi_w_packed = np.stack(
        [swi_w[e].T for e in range(G)], axis=0  # each [1280, 640]
    ).reshape(G * 2 * F_DIM, D_MODEL, 1, 1).astype(np.float32)
    swi_b_packed = swi_b.reshape(G * 2 * F_DIM).astype(np.float32)

    # ---- Pack-2: stack all experts' down along IN axis (each expert's
    # 640 features lives in a contiguous slice along the input-channel axis).
    # Per-expert weight: [out=D_MODEL, in=D_FF, 1, 1]. opf stores [in=640, out=640];
    # transpose to [out, in] = [640, 640] and concat along IN axis -> [640, G*640].
    out_w_packed = np.concatenate(
        [out_w[e].T for e in range(G)], axis=1  # [640, G*640]
    ).reshape(D_MODEL, G * F_DIM, 1, 1).astype(np.float32)
    # Pack-2 conv bias: zero. Per-expert b_out is folded in via gate-weighted
    # bias term computed separately (see GateBias below).
    out_b_packed = np.zeros((D_MODEL,), dtype=np.float32)

    # Per-expert b_out as a small linear: out_bias_term[b,d] = Σ_e g[b,e] * out_b[e,d]
    # weight matrix B = [128, 640]. We fold this into the model as a tiny Linear
    # consuming the (squeezed) gate vector.
    out_b_table = out_b.astype(np.float32)  # [G, 640]

    class PackedMoE(nn.Module):
        def __init__(self):
            super().__init__()
            # Pack-1: SwiGLU stack
            self.pack1 = nn.Conv2d(D_MODEL, G * 2 * F_DIM, 1, bias=True)
            self.pack1.weight = nn.Parameter(torch.from_numpy(swi_w_packed),
                                             requires_grad=False)
            self.pack1.bias = nn.Parameter(torch.from_numpy(swi_b_packed),
                                           requires_grad=False)
            # Pack-2: down stack
            self.pack2 = nn.Conv2d(G * F_DIM, D_MODEL, 1, bias=True)
            self.pack2.weight = nn.Parameter(torch.from_numpy(out_w_packed),
                                             requires_grad=False)
            self.pack2.bias = nn.Parameter(torch.from_numpy(out_b_packed),
                                           requires_grad=False)
            # Gate-weighted bias: B_table [G, 640]. As a linear over g:
            # bias_term = g @ B_table.  Implement as Conv2d 1x1 with weight
            # [640, G, 1, 1].
            self.bias_proj = nn.Conv2d(G, D_MODEL, 1, bias=False)
            self.bias_proj.weight = nn.Parameter(
                torch.from_numpy(out_b_table.T)  # [640, G]
                     .reshape(D_MODEL, G, 1, 1).contiguous(),
                requires_grad=False,
            )

        def forward(self, x_in, g_in):
            # x_in: [B, 640, 1, 1]    g_in: [B, G, 1, 1]
            h = self.pack1(x_in)                                  # [B, G*1280, 1, 1]
            B = h.shape[0]
            h = h.reshape(B, G, 2 * F_DIM, 1)                     # [B, G, 1280, 1]
            glu = h[:, :, :F_DIM, :]                              # [B, G, 640, 1]
            lin = h[:, :, F_DIM:, :]
            glu = torch.clamp(glu, max=SWIGLU_LIMIT)
            lin = torch.clamp(lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)  # [B,G,640,1]
            # Apply gate (broadcast multiply). g_in shape [B, G, 1, 1].
            g4 = g_in.reshape(B, G, 1, 1)
            mid = mid * g4                                        # [B, G, 640, 1]
            mid = mid.reshape(B, G * F_DIM, 1, 1)                 # [B, G*640, 1, 1]
            y = self.pack2(mid)                                   # [B, 640, 1, 1]
            # Gate-weighted bias term
            bias_term = self.bias_proj(g_in.reshape(B, G, 1, 1))  # [B, 640, 1, 1]
            return y + bias_term

    mod = PackedMoE().eval()
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
    return m


def torch_reference_packed(packed: dict[str, np.ndarray], n_active: int,
                           x_np: np.ndarray, g_np: np.ndarray) -> np.ndarray:
    """Compute Σ_e g_e * Expert_e(x) using the original opf weights (fp32)."""
    import torch
    G = n_active
    x = torch.from_numpy(x_np).to(torch.float32)              # [B, 640]
    g = torch.from_numpy(g_np).to(torch.float32)              # [B, G]
    swi_w = torch.from_numpy(packed["swiglu_w"][:G]).to(torch.float32)  # [G, 640, 1280]
    swi_b = torch.from_numpy(packed["swiglu_b"][:G]).to(torch.float32)  # [G, 1280]
    out_w = torch.from_numpy(packed["out_w"][:G]).to(torch.float32)     # [G, 640, 640]
    out_b = torch.from_numpy(packed["out_b"][:G]).to(torch.float32)     # [G, 640]

    # h[b,e,f] = x[b,m] * swi_w[e,m,f] + swi_b[e,f]   for f in 2*D_FF
    # einsum: "bm,emf->bef"
    h = torch.einsum("bm,emf->bef", x, swi_w) + swi_b           # [B, G, 1280]
    glu, lin = h[..., :D_FF], h[..., D_FF:]
    glu = torch.clamp(glu, max=SWIGLU_LIMIT)
    lin = torch.clamp(lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)  # [B, G, 640]
    # per-expert down: y_e[b,d] = mid[b,e,f] * out_w[e,f,d] + out_b[e,d]
    y_e = torch.einsum("bef,efd->bed", mid, out_w) + out_b       # [B, G, 640]
    # weighted sum
    out = (g.unsqueeze(-1) * y_e).sum(dim=1)                     # [B, 640]
    return out.numpy()


def ane_residency_summary(pkg_path: Path) -> str:
    """Compile and inspect compute-device placement for each linear/conv op."""
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


def quantize_int4_g32(src_path: Path, dst_path: Path) -> None:
    """Apply linear_int4_grouped_g32 (proven config from quantize_pf_experts_int4.py).

    Per gatekeeper note: G=32 along Cin (input channel axis). For our Conv2d 1x1:
      Pack-1 Cin = 640 → 20 groups. Pack-2 Cin = G*640 → many groups.
    """
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    src = ct.models.MLModel(str(src_path),
                            compute_units=ct.ComputeUnit.CPU_AND_NE)
    cfg = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", weight_threshold=0,
        dtype="int4", granularity="per_block", block_size=32,
    )
    quantized = cto.linear_quantize_weights(
        src, config=cto.OptimizationConfig(global_config=cfg))
    tmp = dst_path.with_suffix(".tmp.mlpackage")
    if tmp.exists():
        shutil.rmtree(tmp)
    quantized.save(str(tmp))
    if dst_path.exists():
        shutil.rmtree(dst_path)
    tmp.rename(dst_path)


def package_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024**2)


def step_a_probe_g16(packed) -> bool:
    """Build G=16 FP16 probe, check ANE placement."""
    print("\n=== STEP A: G=16 FP16 ANE-residency probe ===")
    probe_pkg = TMP_DIR / "_probe_fp16_DO_NOT_BENCH.mlpackage"
    if probe_pkg.exists():
        shutil.rmtree(probe_pkg)
    build_packed_program(packed, n_active_experts=16, out_path=probe_pkg)
    sz = package_size_mb(probe_pkg)
    print(f"  built: {probe_pkg.name} ({sz:.2f} MB)")
    devs = ane_residency_summary(probe_pkg)
    print(f"  placement: {devs}")
    # PASS if at least one of the two big convs (pack1/pack2 weights) is on ANE
    ok = devs.count("ANE") >= 2
    print(f"  result: {'PASS — ANE residency confirmed' if ok else 'FAIL — fell to CPU'}")
    # Discard probe per gatekeeper tweak #1
    shutil.rmtree(probe_pkg)
    compiled = probe_pkg.with_suffix(".mlmodelc")
    if compiled.exists():
        shutil.rmtree(compiled)
    return ok


def step_b_full_int4(packed) -> tuple[Path, Path]:
    """Build full G=128 FP16 then quantize to INT4 G=32.
    Returns (fp16_path, int4_path). Caller decides whether to keep fp16."""
    print("\n=== STEP B: full G=128 build + INT4 G=32 quant ===")
    fp16_pkg = TMP_DIR / "PF_packed128_fp16_intermediate.mlpackage"
    if not fp16_pkg.exists():
        build_packed_program(packed, n_active_experts=128, out_path=fp16_pkg)
    fp16_sz = package_size_mb(fp16_pkg)
    print(f"  fp16 intermediate: {fp16_sz:.2f} MB")

    int4_pkg = ART_DIR / "PF_packed_iverson_128_int4.mlpackage"
    quantize_int4_g32(fp16_pkg, int4_pkg)
    int4_sz = package_size_mb(int4_pkg)
    print(f"  int4 final: {int4_pkg.name} ({int4_sz:.2f} MB)")
    return fp16_pkg, int4_pkg


def step_c_golden_validate(int4_pkg: Path, packed) -> bool:
    """Replay top-4 routing from pf_layer0_moe.npz, compare to torch reference."""
    print("\n=== STEP C: golden-validator (cos >= %.3f) ===" % COS_GATE_INTERMEDIATE)
    import coremltools as ct
    z = np.load(MOE_GOLDEN, allow_pickle=False)
    norm_all = z["mlp_norm_out"].reshape(-1, D_MODEL).astype(np.float32)[:TRACE_N]
    topk_idx = z["topk_indices"].reshape(-1, TOPK).astype(np.int64)[:TRACE_N]
    topk_w = z["topk_weights"].reshape(-1, TOPK).astype(np.float32)[:TRACE_N]

    # Build dense gate vector [B, 128] with top-4 nonzero entries
    g_dense = np.zeros((TRACE_N, N_EXPERTS), dtype=np.float32)
    for b in range(TRACE_N):
        for k in range(TOPK):
            g_dense[b, topk_idx[b, k]] = topk_w[b, k]

    # On-device prediction
    m = ct.models.MLModel(str(int4_pkg), compute_units=ct.ComputeUnit.ALL)
    feed = {
        "x_in": norm_all.reshape(TRACE_N, D_MODEL, 1, 1).astype(np.float16),
        "g_in": g_dense.reshape(TRACE_N, N_EXPERTS, 1, 1).astype(np.float16),
    }
    pred = m.predict(feed)["x_out"].astype(np.float32).reshape(TRACE_N, D_MODEL)

    # Torch reference (full 128 experts; mask is in g_dense)
    ref = torch_reference_packed(packed, n_active=N_EXPERTS, x_np=norm_all,
                                 g_np=g_dense)

    cos = float((pred * ref).sum() /
                (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-30))
    max_abs = float(np.abs(pred - ref).max())
    print(f"  cos={cos:.6f}  max|Δ|={max_abs:.4f}")
    ok = cos >= COS_GATE_INTERMEDIATE
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def step_d_final_residency(int4_pkg: Path) -> str:
    print("\n=== STEP D: final ANE-residency on G=128 INT4 pack ===")
    devs = ane_residency_summary(int4_pkg)
    print(f"  placement: {devs}")
    return devs


def main() -> int:
    _check_interp()
    if not WEIGHTS.exists():
        raise SystemExit(f"Missing weights: {WEIGHTS}")
    if not MOE_GOLDEN.exists():
        raise SystemExit(f"Run pf_moe_goldens.py first: {MOE_GOLDEN}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-probe", action="store_true",
                    help="Skip Step A G=16 probe (use only if already known to pass)")
    args = ap.parse_args()

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[iverson] python: {sys.executable}")
    import coremltools as ct
    print(f"[iverson] coremltools {ct.__version__}")

    print("[iverson] loading layer-0 MoE weights")
    packed = load_packed_weights()
    print(f"  swiglu_w {packed['swiglu_w'].shape}, out_w {packed['out_w'].shape}")

    if not args.skip_probe:
        if not step_a_probe_g16(packed):
            print("\n[iverson] ABORT: G=16 probe did not land on ANE. "
                  "Investigate before scaling to 128.")
            return 1

    fp16_pkg, int4_pkg = step_b_full_int4(packed)

    print("\n--- FP16 intermediate validation (isolates quant from packing) ---")
    fp16_ok = step_c_golden_validate(fp16_pkg, packed)
    if not fp16_ok:
        print("\n[iverson] FAIL: FP16 cos failed -> packing bug, not quant.")
        print(f"  Keeping FP16 intermediate at: {fp16_pkg}")
        return 2
    print("  FP16 PASS -> packing is correct, proceeding to INT4 validation.")

    print("\n--- INT4 validation ---")
    if not step_c_golden_validate(int4_pkg, packed):
        print("\n[iverson] FAIL: INT4 cos failed (FP16 was OK -> quant config too lossy).")
        return 3
    # cleanup fp16 intermediate
    shutil.rmtree(fp16_pkg)
    devs = step_d_final_residency(int4_pkg)

    print("\n[iverson] DONE")
    print(f"  artifact: {int4_pkg}")
    print(f"  size:     {package_size_mb(int4_pkg):.2f} MB")
    print(f"  devices:  {devs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
