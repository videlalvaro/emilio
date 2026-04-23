"""Validate fused attn+router ANE pack vs golden reference.

Compares all 3 outputs:
  x_attn       — cosine vs golden L{n}_attn_out (from pf_attn_alllayers.npz)
  normed_x     — cosine vs golden L{n}_mlp_norm_out (from pf_alllayers_moe.npz)
  router_probs — top-4 index match rate vs golden L{n}_topk_indices

Usage:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/privacy/validate_fused_attn_router.py --layer 0
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
GOLDEN_ATTN = REPO / "python" / "privacy" / "out" / "pf_attn_alllayers.npz"
GOLDEN_MOE  = REPO / "python" / "privacy" / "out" / "pf_alllayers_moe.npz"
PKG_DIR = REPO / "emilio" / "conv-ane"

D_MODEL = 640
T_SEQ = 128
N_EXPERTS = 128
TOPK = 4
COS_THRESH = 0.97


def cosine(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    return dot / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--sentence", type=int, default=0, help="Which sentence (0-7)")
    args = ap.parse_args()
    L = args.layer
    S = args.sentence

    import coremltools as ct
    print(f"[val] coremltools {ct.__version__}")

    # Load golden references
    za = np.load(str(GOLDEN_ATTN), allow_pickle=False)
    zm = np.load(str(GOLDEN_MOE), allow_pickle=False)

    # Golden: attn_in is input to the block, attn_out is after attn+residual
    attn_in  = za[f"L{L}_attn_in"][S]    # [T, D] float32
    attn_out = za[f"L{L}_attn_out"][S]   # [T, D] float32

    # Golden: mlp_norm_out is the RMSNorm'd x for MLP
    norm_out = zm[f"L{L}_mlp_norm_out"][S]  # [T, D] float32

    # Golden: topk indices and weights
    topk_idx = zm[f"L{L}_topk_indices"][S]  # [T, 4] int32
    topk_wts = zm[f"L{L}_topk_weights"][S]  # [T, 4] float32

    # Construct attention mask — sentence 0 is fully unmasked for simplicity
    # (the golden was computed with the real mask, but for sentence 0 all tokens are valid)
    # Read the real mask from swift weights
    swift_dir = PKG_DIR / "PF_swift"
    mask_all = np.fromfile(str(swift_dir / "attention_mask.bin"), dtype=np.int32)
    mask = mask_all[S * T_SEQ : (S + 1) * T_SEQ]
    pad_add = np.where(mask > 0, 0.0, -1e4).astype(np.float16).reshape(1, 1, 1, T_SEQ)

    # Prepare input: x_in [1, D, 1, T] fp16
    x_in = attn_in.T.reshape(1, D_MODEL, 1, T_SEQ).astype(np.float16)

    # Load fused model
    pkg = str(PKG_DIR / f"PF_fused_L{L}_T128.mlpackage")
    print(f"[val] loading {pkg}")

    # Test on CPU_ONLY
    cfg_cpu = ct.models.MLModel
    model_cpu = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_ONLY)
    out_cpu = model_cpu.predict({"x_in": x_in, "pad_add": pad_add})

    # Test on ALL (CPU + ANE)
    model_ane = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.ALL)

    # Warmup
    for _ in range(3):
        model_ane.predict({"x_in": x_in, "pad_add": pad_add})

    N_RUNS = 20
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        out_ane = model_ane.predict({"x_in": x_in, "pad_add": pad_add})
    t_ane = (time.perf_counter() - t0) / N_RUNS * 1000

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        out_cpu2 = model_cpu.predict({"x_in": x_in, "pad_add": pad_add})
    t_cpu = (time.perf_counter() - t0) / N_RUNS * 1000

    # Figure out output keys (coremltools may rename tuple outputs)
    keys = sorted(out_ane.keys())
    print(f"[val] output keys: {keys}")
    for k in keys:
        v = out_ane[k]
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    # Map outputs — coremltools names tuple outputs as var_NNN or x_attn etc.
    # Try direct names first, fall back to positional
    def get_out(d, name, fallback_idx):
        if name in d:
            return np.array(d[name], dtype=np.float32)
        # Fallback: sorted keys by index
        return np.array(d[keys[fallback_idx]], dtype=np.float32)

    x_attn_ane = get_out(out_ane, "x_attn", 0)
    normed_ane = get_out(out_ane, "normed_x", 1)
    router_ane = get_out(out_ane, "router_probs", 2)

    # Reshape to [T, D] for comparison
    # Output shapes: x_attn [1,D,1,T], normed_x [1,D,1,T], router_probs [1,128,1,T]
    x_attn_td = x_attn_ane.reshape(D_MODEL, T_SEQ).T   # [T, D]
    normed_td = normed_ane.reshape(D_MODEL, T_SEQ).T    # [T, D]
    router_te = router_ane.reshape(N_EXPERTS, T_SEQ).T   # [T, 128]

    # 1) x_attn vs golden attn_out
    cos_attn = cosine(x_attn_td, attn_out)
    print(f"\n[val] x_attn  cos vs golden: {cos_attn:.6f}  {'PASS' if cos_attn >= COS_THRESH else 'FAIL'}")

    # 2) normed_x vs golden mlp_norm_out
    cos_norm = cosine(normed_td, norm_out)
    print(f"[val] normed_x cos vs golden: {cos_norm:.6f}  {'PASS' if cos_norm >= COS_THRESH else 'FAIL'}")

    # 3) Router: top-4 match rate
    fused_top4 = np.argsort(-router_te, axis=1)[:, :TOPK]  # [T, 4]
    match_count = 0
    total = T_SEQ * TOPK
    for t in range(T_SEQ):
        golden_set = set(topk_idx[t])
        for k in range(TOPK):
            if fused_top4[t, k] in golden_set:
                match_count += 1
    match_rate = match_count / total
    print(f"[val] top-4 match rate:       {match_rate:.4f}  {'PASS' if match_rate >= 0.95 else 'FAIL'}")

    # Timing
    print(f"\n[val] Timing (avg of {N_RUNS} runs):")
    print(f"  CPU:     {t_cpu:.3f} ms")
    print(f"  ANE/ALL: {t_ane:.3f} ms")
    print(f"  Speedup: {t_cpu/t_ane:.2f}x")

    # Overall
    ok = cos_attn >= COS_THRESH and cos_norm >= COS_THRESH and match_rate >= 0.95
    print(f"\n[val] Overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
