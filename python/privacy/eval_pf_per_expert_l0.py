"""Phase A: validate per-expert dispatch correctness on layer 0 ONLY.

Loads 128 PF_expert_{i}_B64_fp16.mlmodelc, runs per-expert dispatch for each
sentence in pf_layer0_moe.npz, computes the weighted-sum on host, compares
against a torch reference computed from raw weights.

Pass gate: cos >= 0.985 vs torch reference.
If PASS, the per-expert integration is correct and we can commit to building
layers 1-7.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS   = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights" / "model.safetensors"
GOLDEN    = REPO_ROOT / "python" / "privacy" / "out" / "pf_layer0_moe.npz"
ART_DIR   = REPO_ROOT / "emilio" / "conv-ane"

D_MODEL = 640
D_FF    = 640
N_EXPERTS = 128
TOPK    = 4
B       = 64
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
COS_GATE = 0.985


def torch_reference_moe(w_all, norm_out, topk_idx, topk_w):
    """Compute MoE(norm_out) using the same recipe as the dense pack:
       y = sum_k topk_w[b,k] * Expert(idx[b,k])(norm_out[b])
    Returns [B, D_MODEL] in float32.
    """
    import torch
    swi_w = torch.from_numpy(w_all["swi_w"]).float()
    swi_b = torch.from_numpy(w_all["swi_b"]).float()
    out_w = torch.from_numpy(w_all["out_w"]).float()
    out_b = torch.from_numpy(w_all["out_b"]).float()
    x = torch.from_numpy(norm_out).float()         # [B, D_MODEL]
    idx = torch.from_numpy(topk_idx).long()        # [B, K]
    w   = torch.from_numpy(topk_w).float()         # [B, K]
    Bn = x.shape[0]
    out = torch.zeros(Bn, D_MODEL)
    for b in range(Bn):
        for k in range(TOPK):
            e = int(idx[b, k])
            sw = swi_w[e]; sb = swi_b[e]
            ow = out_w[e]; ob = out_b[e]
            h = x[b] @ sw + sb                     # [2*D_FF]
            glu, lin = h[:D_FF], h[D_FF:]
            glu = torch.clamp(glu, max=SWIGLU_LIMIT)
            lin = torch.clamp(lin, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            mid = glu * torch.sigmoid(SWIGLU_ALPHA * glu) * (lin + 1.0)  # [D_FF]
            y = mid @ ow + ob                      # [D_MODEL]
            out[b] += w[b, k] * y
    return out.numpy()


def per_expert_dispatch(experts, norm_out, topk_idx, topk_w):
    """Mimic the Swift runtime. For each distinct expert in this batch:
       1. Build B'-sized input by GATHERING rows that route to it.
       2. Call expert.predict (B'=full batch, padded with zeros for unused rows).
       3. SCATTER weighted contribution back into output.
    Per-expert call is full B=64 (matches the compiled mlmodelc batch size);
    we mask out unused rows by simply not summing them.
    """
    Bn = norm_out.shape[0]
    out = np.zeros((Bn, D_MODEL), dtype=np.float32)
    # For each expert that's actually called this batch, compute once.
    # Build per-expert input: full B=64 rows, padded with zeros for inactive.
    distinct = sorted(set(int(i) for i in topk_idx.flatten()))
    # Active mask: for each (b, expert) which (k, weight)
    expert_to_rows: dict[int, list[tuple[int, float]]] = {}
    for b in range(Bn):
        for k in range(TOPK):
            e = int(topk_idx[b, k]); w = float(topk_w[b, k])
            expert_to_rows.setdefault(e, []).append((b, w))

    for e in distinct:
        # Build B=64 input: fill rows from norm_out for the b's that route to e,
        # zeros for the rest. Then ANE gives us back B outputs and we only use
        # the relevant rows.
        x_in = np.zeros((B, D_MODEL, 1, 1), dtype=np.float16)
        for b, _w in expert_to_rows[e]:
            x_in[b, :, 0, 0] = norm_out[b].astype(np.float16)
        y = experts[e].predict({"x_in": x_in})["y_out"]   # [B, D_MODEL, 1, 1]
        y = y.reshape(B, D_MODEL).astype(np.float32)
        for b, w in expert_to_rows[e]:
            out[b] += w * y[b]
    return out


def cos(a, b):
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print(f"[per-expert-l0] loading golden + weights")
    z = np.load(GOLDEN)
    norm_out_all = z["mlp_norm_out"]               # [8, 128, 640]
    topk_idx_all = z["topk_indices"]               # [8, 128, 4]
    topk_w_all   = z["topk_weights"]               # [8, 128, 4]
    # softmax(topk_logits) — but the dense-pack gating uses /TOPK convention.
    # The pf_layer0_moe.npz stores raw topk_weights = softmax(top4 logits).
    # The MoE per opf contract: y = sum_k w_k * E(idx_k), where w_k = softmax/K
    # is what's used inside the existing pack. We follow the SAME convention.
    # So we divide here.
    topk_w_all = topk_w_all / TOPK

    # Load expert weights once for torch reference.
    from safetensors.torch import safe_open
    import torch
    keys = {
        "swi_w": "block.0.mlp.swiglu.weight",
        "swi_b": "block.0.mlp.swiglu.bias",
        "out_w": "block.0.mlp.out.weight",
        "out_b": "block.0.mlp.out.bias",
    }
    with safe_open(str(WEIGHTS), framework="pt") as f:
        w_all = {k: f.get_tensor(v).to(torch.float32).cpu().numpy()
                 for k, v in keys.items()}

    print(f"[per-expert-l0] loading 128 expert mlmodelc on ANE")
    import coremltools as ct
    experts = []
    for e in range(N_EXPERTS):
        p = ART_DIR / f"PF_expert_L0_{e}_B{B}_fp16.mlpackage"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        experts.append(ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE))

    # Run on the first sentence (B=128, but we operate in chunks of B=64 to match compiled shape).
    cos_per_chunk = []
    for sent in range(min(2, norm_out_all.shape[0])):
        no = norm_out_all[sent]                    # [128, 640]
        ti = topk_idx_all[sent]                    # [128, 4]
        tw = topk_w_all[sent]                      # [128, 4]
        # Two chunks of B=64.
        for off in (0, 64):
            no_c = no[off:off+B]
            ti_c = ti[off:off+B]
            tw_c = tw[off:off+B]
            t0 = time.time()
            ref = torch_reference_moe(w_all, no_c, ti_c, tw_c)
            t_ref = time.time() - t0
            t0 = time.time()
            pred = per_expert_dispatch(experts, no_c, ti_c, tw_c)
            t_pred = time.time() - t0
            c = cos(ref, pred)
            cos_per_chunk.append(c)
            print(f"  sent {sent}  off {off}  cos={c:.6f}  "
                  f"ref={t_ref*1000:.0f}ms  pred={t_pred*1000:.0f}ms  "
                  f"max|Δ|={np.abs(ref-pred).max():.4f}")

    cmin = min(cos_per_chunk); cmean = sum(cos_per_chunk) / len(cos_per_chunk)
    print(f"\n[per-expert-l0] {len(cos_per_chunk)} chunks  cos min={cmin:.6f}  mean={cmean:.6f}  "
          f"gate={COS_GATE}  [{'PASS' if cmin >= COS_GATE else 'FAIL'}]")
    return 0 if cmin >= COS_GATE else 1


if __name__ == "__main__":
    sys.exit(main())
