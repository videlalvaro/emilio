"""T5a: hybrid end-to-end logits validation for privacy-filter.

CPU does embedding + attention + final norm + lm_head (via opf forward).
Each MLPBlock output is REPLACED via forward-hook with the prediction from the
ANE INT8 packed-MoE artifact for that layer. Compares final logits against
pf_golden.npz to measure how MoE-only INT8 error compounds end-to-end.

Run with the Xcode python (coremltools 9):
    /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
        python/privacy/eval_pf_hybrid_chain.py

Expects:
    python/privacy/out/pf_golden.npz                 (from pf_ref.py)
    python/privacy/_vendor_src/weights/              (opf checkpoint)
    emilio/conv-ane/PF_packed_iverson_L{0..7}_N4_int8.mlpackage
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
PKG_FMT = "PF_packed_iverson_L{n}_N4_int8.mlpackage"

D_MODEL = 640
N_EXPERTS = 128
TOPK = 4
N_LAYERS = 8

# Make sure vendored opf import works.
sys.path.insert(0, str(REPO_ROOT / "python" / "privacy" / "_vendor_src" / "opf_src"))


def make_replace_hook(layer_idx: int, mlmodel):
    """Forward hook on MLPBlock that replaces the entire block output with
    block_in + TOPK * ANE_packed_moe(norm_out, dense_gate)."""
    def hook(module, inputs, _output):
        x = inputs[0]                               # [B, T, 640] (or [BT,640])
        orig_shape = x.shape
        if x.dim() == 2:
            B, _ = x.shape; T = 1
            x_3d = x.unsqueeze(1)
        else:
            B, T, _ = x.shape
            x_3d = x
        norm_out_native = module.norm(x_3d)         # [B, T, 640] in opf dtype (bf16)
        norm_out = norm_out_native.to(torch.float32)
        # opf gate uses F.linear directly (matches goldens script)
        gate_logits = F.linear(norm_out,
                               module.gate.weight.to(torch.float32),
                               module.gate.bias.to(torch.float32))
        topk = torch.topk(gate_logits, k=TOPK, dim=-1, sorted=True)
        weights = F.softmax(topk.values, dim=-1) / TOPK    # opf normalisation
        # Build dense [B, T, N_EXPERTS] gate.
        BT = B * T
        flat_w = weights.reshape(BT, TOPK)
        flat_i = topk.indices.reshape(BT, TOPK)
        g_dense = torch.zeros(BT, N_EXPERTS, dtype=torch.float32)
        g_dense.scatter_(1, flat_i, flat_w)
        # ANE feed. Pack expects fixed batch (TRACE_N=64); chunk if needed.
        x_np = norm_out.reshape(BT, D_MODEL, 1, 1).to(torch.float16).numpy()
        g_np = g_dense.reshape(BT, N_EXPERTS, 1, 1).to(torch.float16).numpy()
        # Probe pack batch from first input shape.
        pack_batch = mlmodel.get_spec().description.input[0].type.multiArrayType.shape[0]
        out_chunks = []
        for s in range(0, BT, pack_batch):
            e = min(s + pack_batch, BT)
            xc = x_np[s:e]
            gc = g_np[s:e]
            if xc.shape[0] < pack_batch:
                pad_n = pack_batch - xc.shape[0]
                xc = np.concatenate([xc, np.zeros((pad_n, D_MODEL, 1, 1), np.float16)], 0)
                gc = np.concatenate([gc, np.zeros((pad_n, N_EXPERTS, 1, 1), np.float16)], 0)
            r = mlmodel.predict({"x_in": xc, "g_in": gc})["x_out"]
            out_chunks.append(r[: e - s])
        pred = np.concatenate(out_chunks, axis=0)              # [BT, 640, 1, 1]
        pred_t = torch.from_numpy(pred.astype(np.float32)).reshape(B, T, D_MODEL)
        # opf MLPBlock returns x + experts_per_token * combine; our ANE pack returns
        # the un-scaled combine, so multiply by TOPK.
        delta = (pred_t * TOPK).to(x_3d.dtype)
        out = x_3d + delta
        return out.reshape(orig_shape)
    return hook


def main() -> int:
    import coremltools as ct  # only needed inside Xcode python
    print(f"[t5a] coremltools {ct.__version__}")

    if not GOLDEN.exists():
        raise SystemExit(f"missing {GOLDEN}")
    z = np.load(GOLDEN, allow_pickle=False)
    input_ids = torch.from_numpy(z["input_ids"].astype(np.int64))
    attn_mask = torch.from_numpy(z["attention_mask"].astype(np.int64))
    golden_logits = z["logits"].astype(np.float32)               # [B, T, V_classes]
    print(f"[t5a] golden logits {golden_logits.shape}")

    # Load all 8 ANE packs (CPU+ANE; ALL is fine, we want correctness here).
    print("[t5a] loading 8 ANE INT8 packs ...")
    packs = []
    for n in range(N_LAYERS):
        p = PKG_DIR / PKG_FMT.format(n=n)
        if not p.exists():
            raise SystemExit(f"missing {p}")
        m = ct.models.MLModel(str(p), compute_units=ct.ComputeUnit.ALL)
        packs.append(m)
        print(f"  L{n}: loaded ({p.name})")

    # Load opf model on CPU.
    from opf._model.model import Transformer
    print("[t5a] loading opf Transformer (CPU bf16)")
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=torch.device("cpu"))
    model.eval()
    torch.manual_seed(0); torch.set_num_threads(8)

    # Install replacement hooks on every MLPBlock.
    handles = []
    for n in range(N_LAYERS):
        h = model.block[n].mlp.register_forward_hook(make_replace_hook(n, packs[n]))
        handles.append(h)

    print(f"[t5a] hybrid forward B={input_ids.shape[0]} T={input_ids.shape[1]}")
    t0 = time.time()
    with torch.inference_mode():
        logits = model(input_ids, attention_mask=attn_mask)
    dt = time.time() - t0
    for h in handles:
        h.remove()
    pred_logits = logits.detach().to(torch.float32).cpu().numpy()
    print(f"[t5a] forward took {dt:.1f}s; pred logits {pred_logits.shape}")

    # Per-token cosine.
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

    # Argmax agreement.
    p_arg = p_flat.argmax(axis=-1)
    g_arg = g_flat.argmax(axis=-1)
    top1_agree = float((p_arg == g_arg).mean())

    # Per-sentence cosine.
    print(f"\n[t5a] === RESULT ===")
    print(f"  global logits cosine          : {cos_global:.6f}  (gate 0.97)")
    print(f"  per-token cosine mean / min   : {cos_mean:.6f} / {cos_min:.6f}")
    print(f"  per-token cosine 1st-percentile: {cos_p01:.6f}")
    print(f"  top-1 argmax agreement        : {top1_agree*100:.2f}%")

    # Per-sentence breakdown.
    B, T, _ = pred_logits.shape
    print(f"\n  per-sentence cos / top1 agree:")
    for b in range(B):
        m_b = attn_mask.numpy()[b].astype(bool)
        if not m_b.any():
            continue
        pb = pred_logits[b][m_b]
        gb = golden_logits[b][m_b]
        cos_b = float(((pb * gb).sum()) / (np.linalg.norm(pb)*np.linalg.norm(gb) + eps))
        agree_b = float((pb.argmax(-1) == gb.argmax(-1)).mean())
        print(f"    s{b}: T_valid={m_b.sum():3d}  cos={cos_b:.6f}  top1={agree_b*100:6.2f}%")

    status = "PASS" if (cos_global >= 0.97 and top1_agree >= 0.95) else "FAIL"
    print(f"\n  T5a status: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
