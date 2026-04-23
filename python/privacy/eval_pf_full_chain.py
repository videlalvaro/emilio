"""T5b: full ANE-chain end-to-end logits validation for privacy-filter.

CPU does ONLY embedding + final norm + lm_head.
Each AttentionBlock and MLPBlock output is replaced via forward-hook with the
prediction from the corresponding ANE artifact:
  PF_attn{N}_T128.mlpackage           (FP16, fixed B=1, T=128)
  PF_packed_iverson_L{N}_N4_int8.mlpackage   (INT8 packed-MoE, fixed batch=64)

Compares final logits against pf_golden.npz to measure how the full
8x(attn FP16 + moe INT8) chain compounds end-to-end.

Run with .venv313 (has both opf and coremltools):
    PYTHONPATH=python/privacy/_vendor_src/opf_src \\
        .venv313/bin/python python/privacy/eval_pf_full_chain.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights"
GOLDEN = REPO_ROOT / "python" / "privacy" / "out" / "pf_golden.npz"
PKG_DIR = REPO_ROOT / "emilio" / "conv-ane"
MOE_FMT = "PF_packed_iverson_L{n}_N4_int8.mlpackage"
ATTN_FMT = "PF_attn{n}_T128.mlpackage"

D_MODEL = 640
N_EXPERTS = 128
TOPK = 4
N_LAYERS = 8
T_SEQ = 128

sys.path.insert(0, str(REPO_ROOT / "python" / "privacy" / "_vendor_src" / "opf_src"))


# --- Attention replacement -------------------------------------------------

def make_attn_hook(layer_idx: int, mlmodel, pad_add_per_sentence):
    """Replace AttentionBlock.forward output with PF_attn{N}_T128 prediction.
    Each pack has fixed B=1, T=128, so we loop over sentences.
    pad_add_per_sentence: list of np.float16 arrays of shape (1,1,1,T).
    """
    def hook(module, inputs, _output):
        x = inputs[0]                                  # opf passes [B, T, 640] bf16
        orig_dtype = x.dtype
        B, T, D = x.shape
        assert T == T_SEQ, f"attn pack baked at T={T_SEQ}, got T={T}"
        x_np = x.detach().to(torch.float32).cpu().numpy()
        out_b = np.zeros((B, T, D), dtype=np.float32)
        for i in range(B):
            x_in = x_np[i].T[None, :, None, :].astype(np.float16)  # (1,640,1,128)
            r = mlmodel.predict({"x_in": x_in,
                                 "pad_add": pad_add_per_sentence[i]})["x_out"]
            out_b[i] = r.reshape(D, T).T.astype(np.float32)
        return torch.from_numpy(out_b).to(orig_dtype)
    return hook


# --- MoE replacement (same as T5a) ----------------------------------------

def make_moe_hook(layer_idx: int, mlmodel):
    def hook(module, inputs, _output):
        x = inputs[0]
        orig_shape = x.shape
        if x.dim() == 2:
            B, _ = x.shape; T = 1
            x_3d = x.unsqueeze(1)
        else:
            B, T, _ = x.shape
            x_3d = x
        norm_out = module.norm(x_3d).to(torch.float32)
        gate_logits = F.linear(norm_out,
                               module.gate.weight.to(torch.float32),
                               module.gate.bias.to(torch.float32))
        topk = torch.topk(gate_logits, k=TOPK, dim=-1, sorted=True)
        weights = F.softmax(topk.values, dim=-1) / TOPK
        BT = B * T
        flat_w = weights.reshape(BT, TOPK)
        flat_i = topk.indices.reshape(BT, TOPK)
        g_dense = torch.zeros(BT, N_EXPERTS, dtype=torch.float32)
        g_dense.scatter_(1, flat_i, flat_w)
        x_np = norm_out.reshape(BT, D_MODEL, 1, 1).to(torch.float16).numpy()
        g_np = g_dense.reshape(BT, N_EXPERTS, 1, 1).to(torch.float16).numpy()
        pack_batch = mlmodel.get_spec().description.input[0].type.multiArrayType.shape[0]
        out_chunks = []
        for s in range(0, BT, pack_batch):
            e = min(s + pack_batch, BT)
            xc = x_np[s:e]; gc = g_np[s:e]
            if xc.shape[0] < pack_batch:
                pad_n = pack_batch - xc.shape[0]
                xc = np.concatenate([xc, np.zeros((pad_n, D_MODEL, 1, 1), np.float16)], 0)
                gc = np.concatenate([gc, np.zeros((pad_n, N_EXPERTS, 1, 1), np.float16)], 0)
            r = mlmodel.predict({"x_in": xc, "g_in": gc})["x_out"]
            out_chunks.append(r[: e - s])
        pred = np.concatenate(out_chunks, axis=0)
        pred_t = torch.from_numpy(pred.astype(np.float32)).reshape(B, T, D_MODEL)
        delta = (pred_t * TOPK).to(x_3d.dtype)
        return (x_3d + delta).reshape(orig_shape)
    return hook


# --- Per-expert MoE replacement (Option A) --------------------------------

def make_per_expert_moe_hook(layer_idx: int, experts: list, B_PACK: int = 64):
    """Replace MLPBlock with host-side gather -> per-expert dispatch -> scatter.

    `experts` is a list of N_EXPERTS MLModel objects (one per expert) for this layer.
    Uses a ThreadPoolExecutor for concurrent dispatch (CoreML releases the GIL).
    """
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=16)

    def hook(module, inputs, _output):
        x = inputs[0]
        orig_shape = x.shape
        if x.dim() == 2:
            B, _ = x.shape; T = 1
            x_3d = x.unsqueeze(1)
        else:
            B, T, _ = x.shape
            x_3d = x
        norm_out = module.norm(x_3d).to(torch.float32)
        gate_logits = F.linear(norm_out,
                               module.gate.weight.to(torch.float32),
                               module.gate.bias.to(torch.float32))
        topk = torch.topk(gate_logits, k=TOPK, dim=-1, sorted=True)
        weights = F.softmax(topk.values, dim=-1) / TOPK   # (B,T,K)
        BT = B * T
        flat_w = weights.reshape(BT, TOPK).numpy().astype(np.float32)         # (BT,K)
        flat_i = topk.indices.reshape(BT, TOPK).numpy().astype(np.int32)       # (BT,K)
        x_flat = norm_out.reshape(BT, D_MODEL).to(torch.float16).numpy()       # (BT,D)

        out_flat = np.zeros((BT, D_MODEL), dtype=np.float32)

        # Process in chunks of B_PACK rows so each expert call is fixed-size.
        for s in range(0, BT, B_PACK):
            e_end = min(s + B_PACK, BT)
            n_rows = e_end - s
            chunk_i = flat_i[s:e_end]   # (n_rows, K)
            chunk_w = flat_w[s:e_end]   # (n_rows, K)
            chunk_x = x_flat[s:e_end]   # (n_rows, D)

            # Build per-expert assignments: which rows + per-row contribution weight
            # (a row may use the same expert multiple times in topk — sum weights).
            per_expert_rows: dict = {}
            for r in range(n_rows):
                for k in range(TOPK):
                    eid = int(chunk_i[r, k])
                    w = float(chunk_w[r, k])
                    if w == 0.0:
                        continue
                    per_expert_rows.setdefault(eid, []).append((r, w))

            def dispatch_one(eid_rows):
                eid, rows = eid_rows
                pad = np.zeros((B_PACK, D_MODEL, 1, 1), dtype=np.float16)
                for slot, (r, _w) in enumerate(rows):
                    pad[slot, :, 0, 0] = chunk_x[r]
                y = experts[eid].predict({"x_in": pad})["y_out"]   # (B_PACK,D,1,1)
                return eid, rows, y

            results = list(pool.map(dispatch_one, per_expert_rows.items()))

            for eid, rows, y in results:
                y2 = y.reshape(B_PACK, D_MODEL).astype(np.float32)
                for slot, (r, w) in enumerate(rows):
                    out_flat[s + r] += w * y2[slot]

        pred_t = torch.from_numpy(out_flat).reshape(B, T, D_MODEL)
        delta = (pred_t * TOPK).to(x_3d.dtype)
        return (x_3d + delta).reshape(orig_shape)
    return hook


def main() -> int:
    import coremltools as ct
    import os
    print(f"[t5b-full] coremltools {ct.__version__}")
    USE_PER_EXPERT = os.environ.get("PF_PER_EXPERT", "0") == "1"
    print(f"[t5b-full] MoE mode: {'per-expert (Option A)' if USE_PER_EXPERT else 'dense pack (legacy)'}")

    if not GOLDEN.exists():
        raise SystemExit(f"missing {GOLDEN}")
    z = np.load(GOLDEN, allow_pickle=False)
    input_ids = torch.from_numpy(z["input_ids"].astype(np.int64))
    attn_mask = torch.from_numpy(z["attention_mask"].astype(np.int64))
    golden_logits = z["logits"].astype(np.float32)
    print(f"[t5b-full] golden logits {golden_logits.shape}")

    # Pre-compute pad_add per sentence (fp16, -1e4 at padded positions)
    mask_np = attn_mask.numpy().astype(np.int32)
    pad_add_per = []
    for i in range(mask_np.shape[0]):
        pa = np.where(mask_np[i] > 0, 0.0, -1e4).astype(np.float16)
        pad_add_per.append(pa[None, None, None, :])

    print("[t5b-full] loading 8 attn ANE packs ...")
    attn_packs = []
    for n in range(N_LAYERS):
        ap = PKG_DIR / ATTN_FMT.format(n=n)
        if not ap.exists():
            raise SystemExit(f"missing {ap}")
        attn_packs.append(ct.models.MLModel(str(ap), compute_units=ct.ComputeUnit.ALL))
        print(f"  L{n}: attn loaded")

    moe_packs = []
    per_layer_experts: list = []
    if USE_PER_EXPERT:
        print(f"[t5b-full] loading {N_LAYERS}x{N_EXPERTS} per-expert mlpackages ...")
        t_load = time.time()
        for n in range(N_LAYERS):
            experts = []
            for e in range(N_EXPERTS):
                p = PKG_DIR / f"PF_expert_L{n}_{e}_B64_fp16.mlpackage"
                if not p.exists():
                    raise SystemExit(f"missing {p}")
                experts.append(ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.CPU_AND_NE))
            per_layer_experts.append(experts)
            print(f"  L{n}: {N_EXPERTS} experts loaded ({time.time()-t_load:.1f}s elapsed)")
    else:
        for n in range(N_LAYERS):
            mp = PKG_DIR / MOE_FMT.format(n=n)
            if not mp.exists():
                raise SystemExit(f"missing {mp}")
            moe_packs.append(ct.models.MLModel(str(mp), compute_units=ct.ComputeUnit.ALL))
            print(f"  L{n}: moe loaded")

    from opf._model.model import Transformer
    print("[t5b-full] loading opf Transformer (CPU bf16)")
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=torch.device("cpu"))
    model.eval()
    torch.manual_seed(0); torch.set_num_threads(8)

    handles = []
    for n in range(N_LAYERS):
        handles.append(model.block[n].attn.register_forward_hook(
            make_attn_hook(n, attn_packs[n], pad_add_per)))
        if USE_PER_EXPERT:
            handles.append(model.block[n].mlp.register_forward_hook(
                make_per_expert_moe_hook(n, per_layer_experts[n])))
        else:
            handles.append(model.block[n].mlp.register_forward_hook(
                make_moe_hook(n, moe_packs[n])))

    print(f"[t5b-full] full-ANE-chain forward B={input_ids.shape[0]} T={input_ids.shape[1]}")
    t0 = time.time()
    with torch.inference_mode():
        logits = model(input_ids, attention_mask=attn_mask)
    dt = time.time() - t0
    for h in handles:
        h.remove()
    pred_logits = logits.detach().to(torch.float32).cpu().numpy()
    print(f"[t5b-full] forward took {dt:.1f}s; pred logits {pred_logits.shape}")

    valid = attn_mask.numpy().astype(bool).reshape(-1)
    p_flat = pred_logits.reshape(-1, pred_logits.shape[-1])[valid]
    g_flat = golden_logits.reshape(-1, golden_logits.shape[-1])[valid]
    eps = 1e-30
    cos_per_tok = (p_flat * g_flat).sum(axis=1) / (
        np.linalg.norm(p_flat, axis=1) * np.linalg.norm(g_flat, axis=1) + eps)
    cos_global = float(((p_flat * g_flat).sum()) / (
        np.linalg.norm(p_flat) * np.linalg.norm(g_flat) + eps))
    cos_min = float(cos_per_tok.min())
    cos_mean = float(cos_per_tok.mean())
    cos_p01 = float(np.percentile(cos_per_tok, 1))
    p_arg = p_flat.argmax(axis=-1); g_arg = g_flat.argmax(axis=-1)
    top1_agree = float((p_arg == g_arg).mean())

    print(f"\n[t5b-full] === RESULT ===")
    print(f"  global logits cosine          : {cos_global:.6f}  (gate 0.95)")
    print(f"  per-token cosine mean / min   : {cos_mean:.6f} / {cos_min:.6f}")
    print(f"  per-token cosine 1st-pctile   : {cos_p01:.6f}")
    print(f"  top-1 argmax agreement        : {top1_agree*100:.2f}%  (gate 95%)")
    B, T, _ = pred_logits.shape
    print(f"\n  per-sentence cos / top1 agree:")
    for b in range(B):
        m_b = attn_mask.numpy()[b].astype(bool)
        if not m_b.any():
            continue
        pb = pred_logits[b][m_b]; gb = golden_logits[b][m_b]
        cos_b = float(((pb*gb).sum()) / (np.linalg.norm(pb)*np.linalg.norm(gb) + eps))
        agree_b = float((pb.argmax(-1) == gb.argmax(-1)).mean())
        print(f"    s{b}: T_valid={m_b.sum():3d}  cos={cos_b:.6f}  top1={agree_b*100:6.2f}%")

    status = "PASS" if (cos_global >= 0.95 and top1_agree >= 0.95) else "FAIL"
    print(f"\n  T5b-full status: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
