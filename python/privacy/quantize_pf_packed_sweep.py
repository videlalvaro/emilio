"""T3.5-Iverson quant sweep.

FP16 intermediate packing gives cos=1.000000 vs torch reference (packing is
proven). INT4 `linear_int4_grouped_g32` on the big packed Pack-1/Pack-2
weights collapses to cos=0.263.

Per-expert INT4 G=32 worked (cos 0.9935) because each expert had a small
[640, 1280] and [640, 640] matrix. The packed matrices are very different:
  Pack-1: [163840, 640, 1, 1]  — high Cout, small Cin
  Pack-2: [640, 81920, 1, 1]   — small Cout, huge Cin

For Pack-2, per-block along Cin=81920 with block_size=32 gives 2560 blocks
per output row — should be fine. Suspicion: the issue is that the dense
gate-mask multiplies activations by zero for 124/128 experts, BUT the quant
error is on the WEIGHTS. Error accumulates over all 81920 inputs in the
inner product, of which 2560 are nonzero. Noise contribution is ~ sqrt(2560)
× per-weight-error, which can easily destroy cosine.

Try granularities that place more scales where it matters:
  V1: int8 per_channel           — sanity check (should cos >= 0.999)
  V2: int4 per_channel           — fewest bits, one scale per Cout row
  V3: int4 per_block block=16    — twice as many scales as b=32
  V4: int4 per_block block=8     — four times as many (aligned to ANE G=8 law)
  V5: int4 per_block block=4     — diagnostic

Pick smallest variant that hits cos >= 0.985.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "privacy"))

# Reuse helpers from the build script
from build_pf_packed_iverson import (  # type: ignore
    load_packed_weights, torch_reference_packed, ane_residency_summary,
    package_size_mb, MOE_GOLDEN, TMP_DIR, ART_DIR, D_MODEL, N_EXPERTS, TOPK,
    TRACE_N, _check_interp,
)

FP16_SRC = TMP_DIR / "PF_packed128_fp16_intermediate.mlpackage"
OUT_DIR = TMP_DIR / "quant_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COS_GATE = 0.985


def make_variants():
    import coremltools.optimize.coreml as cto
    v = []
    v.append(("int8_per_channel", cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", weight_threshold=0, dtype="int8",
        granularity="per_channel")))
    v.append(("int4_per_channel", cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", weight_threshold=0, dtype="int4",
        granularity="per_channel")))
    for bs in (64, 32, 16, 8):
        v.append((f"int4_block{bs}", cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", weight_threshold=0, dtype="int4",
            granularity="per_block", block_size=bs)))
    return v


def validate(pkg_path: Path, packed: dict[str, np.ndarray]) -> tuple[float, float]:
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


def sweep() -> list[tuple]:
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    if not FP16_SRC.exists():
        raise SystemExit(f"Missing FP16 source: {FP16_SRC}. Build it first.")
    packed = load_packed_weights()
    src = ct.models.MLModel(str(FP16_SRC),
                            compute_units=ct.ComputeUnit.CPU_AND_NE)

    rows = []
    for name, cfg in make_variants():
        out_pkg = OUT_DIR / f"PF_packed128_{name}.mlpackage"
        if out_pkg.exists():
            shutil.rmtree(out_pkg)
        t0 = time.perf_counter()
        try:
            quant = cto.linear_quantize_weights(
                src, config=cto.OptimizationConfig(global_config=cfg))
            quant.save(str(out_pkg))
        except Exception as e:
            print(f"[{name}] quant FAILED: {e!r}")
            rows.append((name, None, None, None, f"quant err: {e!r}"))
            continue
        qt = time.perf_counter() - t0

        sz = package_size_mb(out_pkg)
        try:
            cos, max_abs = validate(out_pkg, packed)
        except Exception as e:
            rows.append((name, sz, None, None, f"predict err: {e!r}"))
            continue
        status = "PASS" if cos >= COS_GATE else "FAIL"
        print(f"[{name:20s}] size={sz:6.2f}MB  cos={cos:.6f}  "
              f"max|Δ|={max_abs:7.3f}  quant={qt:5.1f}s  {status}")
        rows.append((name, sz, cos, max_abs, status))
    return rows


def main() -> int:
    _check_interp()
    rows = sweep()
    print("\n=== sweep summary ===")
    print(f"{'variant':22s}  {'size_MB':>8s}  {'cos':>8s}  {'max|d|':>8s}  status")
    passing = []
    for r in rows:
        name, sz, cos, ma, status = r
        sz_s = f"{sz:8.2f}" if sz else "   err  "
        cos_s = f"{cos:8.6f}" if cos is not None else "   err  "
        ma_s = f"{ma:8.3f}" if ma is not None else "   err  "
        print(f"{name:22s}  {sz_s}  {cos_s}  {ma_s}  {status}")
        if status == "PASS":
            passing.append((sz, name))
    if not passing:
        print("\nNo variant passed. Need different approach (calibration / GPTQ).")
        return 1
    passing.sort()
    best_sz, best_name = passing[0]
    print(f"\nWINNER: {best_name} ({best_sz:.2f} MB)")
    best_pkg = OUT_DIR / f"PF_packed128_{best_name}.mlpackage"
    final = ART_DIR / "PF_packed_iverson_128_best.mlpackage"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(best_pkg, final)
    devs = ane_residency_summary(final)
    print(f"Final artifact: {final}")
    print(f"Placement: {devs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
