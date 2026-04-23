"""T4: build packed-MoE INT8 artifacts for ALL 8 privacy-filter layers,
validate each against per-layer goldens, and report ANE residency.

Reuses the proven v2 design:
  - Pack-1: one big SwiGLU stack [128*1280, 640, 1, 1]
  - Pack-2: split into N_SPLITS=4 sub-convs, summed (Stepanov)
  - INT8 per_channel quant
  - cos gate >= 0.985 per layer (intermediate)

Per-layer build is independent + skip-existing, so a partial run can be resumed.
Final summary table prints per-layer cos, max|Δ|, size, ANE/CPU placement.
"""
from __future__ import annotations
import argparse, shutil, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "privacy"))

# Reuse helpers
from build_pf_packed_iverson_v2 import (  # type: ignore
    build_packed_program_v2, torch_reference_packed,
    ane_residency_summary, quantize_int8_per_channel,
    package_size_mb, _check_interp,
    D_MODEL, D_FF, N_EXPERTS, TOPK, TRACE_N,
)

# Gatekeeper tweak #1: tighter gate (0.985 -> 0.99) since errors compound across
# 8 layers (0.985^8 ~= 0.886 worst case). 0.99^8 ~= 0.923.
COS_GATE_INTERMEDIATE = 0.99

WEIGHTS = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
ALL_GOLDEN = REPO_ROOT / "python" / "privacy" / "out" / "pf_alllayers_moe.npz"
ART_DIR = REPO_ROOT / "emilio" / "conv-ane"
TMP_DIR = ART_DIR / "_pf_t4_tmp"
N_LAYERS = 8
N_SPLITS = 4


def load_layer_weights(layer_idx: int) -> dict[str, np.ndarray]:
    from safetensors.torch import safe_open
    import torch
    keys = {
        "swiglu_w": f"block.{layer_idx}.mlp.swiglu.weight",
        "swiglu_b": f"block.{layer_idx}.mlp.swiglu.bias",
        "out_w":    f"block.{layer_idx}.mlp.out.weight",
        "out_b":    f"block.{layer_idx}.mlp.out.bias",
    }
    out = {}
    with safe_open(str(WEIGHTS), framework="pt") as f:
        for k, full in keys.items():
            out[k] = f.get_tensor(full).to(torch.float32).cpu().numpy()
    return out


def validate(pkg_path: Path, layer_idx: int, packed) -> dict:
    """Returns dict with cos_vs_ref, cos_vs_golden, max_abs.
    cos_vs_ref: pred vs my torch_reference_packed (kernel correctness)
    cos_vs_golden: pred vs the actual opf forward delta (cross-check that the
      reference math agrees with opf -- catches reference-math drift).
    """
    import coremltools as ct
    z = np.load(ALL_GOLDEN, allow_pickle=False)
    norm_all = z[f"L{layer_idx}_mlp_norm_out"].reshape(-1, D_MODEL).astype(np.float32)[:TRACE_N]
    topk_idx = z[f"L{layer_idx}_topk_indices"].reshape(-1, TOPK).astype(np.int64)[:TRACE_N]
    topk_w = z[f"L{layer_idx}_topk_weights"].reshape(-1, TOPK).astype(np.float32)[:TRACE_N]
    golden_delta = z[f"L{layer_idx}_mlp_delta"].reshape(-1, D_MODEL).astype(np.float32)[:TRACE_N]
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
    # opf MoE multiplies the per-token combine by experts_per_token (=4) at the end.
    # Account for that when comparing against golden_delta.
    pred_scaled = pred * TOPK
    cos_ref = float((pred * ref).sum() /
                    (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-30))
    cos_gold = float((pred_scaled * golden_delta).sum() /
                     (np.linalg.norm(pred_scaled) * np.linalg.norm(golden_delta) + 1e-30))
    max_abs = float(np.abs(pred - ref).max())
    return {"cos_ref": cos_ref, "cos_gold": cos_gold, "max_abs": max_abs}


def process_layer(layer_idx: int, force: bool) -> dict:
    print(f"\n=========== LAYER {layer_idx} ===========")
    fp16_pkg = TMP_DIR / f"L{layer_idx}_fp16.mlpackage"
    int8_pkg = ART_DIR / f"PF_packed_iverson_L{layer_idx}_N{N_SPLITS}_int8.mlpackage"

    if int8_pkg.exists() and not force:
        print(f"[L{layer_idx}] skip: {int8_pkg.name} exists")
    else:
        packed = load_layer_weights(layer_idx)
        if not fp16_pkg.exists() or force:
            if fp16_pkg.exists():
                shutil.rmtree(fp16_pkg)
            t0 = time.perf_counter()
            print(f"[L{layer_idx}] build FP16 ({N_SPLITS}-split)")
            build_packed_program_v2(packed, n_splits=N_SPLITS, out_path=fp16_pkg)
            print(f"  built in {time.perf_counter()-t0:.1f}s, "
                  f"{package_size_mb(fp16_pkg):.1f} MB")
        else:
            print(f"[L{layer_idx}] reusing FP16 {fp16_pkg.name}")

        # Quick FP16 sanity (cheap, abort early if packing wrong for this layer)
        v = validate(fp16_pkg, layer_idx, packed)
        if v["cos_ref"] < 0.999:
            print(f"  FP16 cos_ref={v['cos_ref']:.6f} FAIL; aborting layer {layer_idx}")
            return {"layer": layer_idx, "status": "FP16_FAIL", "cos_ref": v["cos_ref"]}

        print(f"[L{layer_idx}] quantize INT8 per_channel")
        quantize_int8_per_channel(fp16_pkg, int8_pkg)
        print(f"  size: {package_size_mb(int8_pkg):.2f} MB")

    packed = load_layer_weights(layer_idx)
    v = validate(int8_pkg, layer_idx, packed)
    devs = ane_residency_summary(int8_pkg)
    ane_ct = devs.count("ANE"); cpu_ct = devs.count("CPU")
    status = "PASS" if (v["cos_ref"] >= COS_GATE_INTERMEDIATE and cpu_ct == 0) else "FAIL"
    print(f"[L{layer_idx}] cos_ref={v['cos_ref']:.6f}  cos_gold={v['cos_gold']:.6f}  "
          f"max|Δ|={v['max_abs']:.4f}  ANE={ane_ct} CPU={cpu_ct}  [{status}]")
    return {
        "layer": layer_idx,
        "size_mb": package_size_mb(int8_pkg),
        "cos_ref": v["cos_ref"],
        "cos_gold": v["cos_gold"],
        "max_abs": v["max_abs"],
        "ane": ane_ct,
        "cpu": cpu_ct,
        "status": status,
        "artifact": int8_pkg.name,
    }


def main() -> int:
    _check_interp()
    if not WEIGHTS.exists():
        raise SystemExit(f"Missing weights: {WEIGHTS}")
    if not ALL_GOLDEN.exists():
        raise SystemExit(
            f"Missing {ALL_GOLDEN}. Run pf_alllayers_moe_goldens.py "
            f"with .venv313 first.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--layers", type=str, default="0-7",
                    help="layer range, e.g. '0-7' or '3,5,7'")
    ap.add_argument("--keep-fp16", action="store_true",
                    help="Keep FP16 intermediates after INT8 succeeds")
    args = ap.parse_args()

    if "-" in args.layers:
        a, b = args.layers.split("-")
        layers = list(range(int(a), int(b) + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for li in layers:
        try:
            rows.append(process_layer(li, args.force))
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({"layer": li, "status": f"ERR:{type(e).__name__}"})
        # cleanup FP16 intermediate after success unless keeping
        if not args.keep_fp16:
            fp16_pkg = TMP_DIR / f"L{li}_fp16.mlpackage"
            if fp16_pkg.exists() and rows[-1].get("status") == "PASS":
                shutil.rmtree(fp16_pkg)

    print("\n========== T4 SUMMARY ==========")
    print(f"{'L':>2}  {'size_MB':>8}  {'cos_ref':>10}  {'cos_gold':>10}  "
          f"{'max|d|':>8}  {'ANE':>4}  {'CPU':>4}  status")
    all_pass = True
    cum_cos = 1.0
    n_ok = 0
    for r in rows:
        if "cos_ref" in r:
            print(f"{r['layer']:>2}  {r['size_mb']:8.2f}  {r['cos_ref']:10.6f}  "
                  f"{r['cos_gold']:10.6f}  {r['max_abs']:8.3f}  {r['ane']:>4}  "
                  f"{r['cpu']:>4}  {r['status']}")
            cum_cos *= r['cos_ref']; n_ok += 1
            if r['status'] != 'PASS':
                all_pass = False
        else:
            print(f"{r['layer']:>2}  {'-':>8}  {'-':>10}  {'-':>10}  "
                  f"{'-':>8}  {'-':>4}  {'-':>4}  {r['status']}")
            all_pass = False
    if n_ok > 0:
        gmean = cum_cos ** (1.0 / n_ok)
        print(f"\ncumulative cos_ref product = {cum_cos:.6f}   geometric mean = {gmean:.6f}")
        print(f"(end-to-end logits cos lower bound = product, gate is 0.97)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
