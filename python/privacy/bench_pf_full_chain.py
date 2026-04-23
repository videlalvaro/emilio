"""T6-A: latency bench for privacy-filter full ANE chain (attn FP16 + MoE INT8).

Methodology:
  - BOOK_ANALYSIS.md (Knuth Vol 1, §1.2.10): median + IQR for skewed timing distributions;
    p99 from n=20 is "indicative" only.
  - ANE_CHAIN_SCHEMA.md: residency confirmed at T5b (cos=0.998895, top-1=100%).

Two configs (no MPS — same GPU rail as ANE residency, would confuse attribution):
  1. ANE chain   — coremltools MLModel(compute_units=CPU_AND_NE) for each pack, opf hooks.
  2. CPU chain   — same coremltools MLModel(compute_units=CPU_ONLY).

Run with Xcode python (coremltools 9):
    /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
        python/privacy/bench_pf_full_chain.py
"""
from __future__ import annotations
import sys, time, platform, subprocess
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

D_MODEL = 640; N_EXPERTS = 128; TOPK = 4; N_LAYERS = 8; T_SEQ = 128
WARMUP = 5; ITERS = 20

sys.path.insert(0, str(REPO_ROOT / "python" / "privacy" / "_vendor_src" / "opf_src"))


def make_attn_hook(mlmodel, pad_add_per):
    def hook(module, inputs, _output):
        x = inputs[0]
        orig_dtype = x.dtype
        B, T, D = x.shape
        x_np = x.detach().to(torch.float32).cpu().numpy()
        out_b = np.zeros((B, T, D), dtype=np.float32)
        for i in range(B):
            x_in = x_np[i].T[None, :, None, :].astype(np.float16)
            r = mlmodel.predict({"x_in": x_in,
                                 "pad_add": pad_add_per[i]})["x_out"]
            out_b[i] = r.reshape(D, T).T.astype(np.float32)
        return torch.from_numpy(out_b).to(orig_dtype)
    return hook


def make_moe_hook(mlmodel):
    def hook(module, inputs, _output):
        x = inputs[0]
        orig_shape = x.shape
        x_3d = x.unsqueeze(1) if x.dim() == 2 else x
        B, T, _ = x_3d.shape
        norm_out = module.norm(x_3d).to(torch.float32)
        gl = F.linear(norm_out,
                      module.gate.weight.to(torch.float32),
                      module.gate.bias.to(torch.float32))
        topk = torch.topk(gl, k=TOPK, dim=-1, sorted=True)
        weights = F.softmax(topk.values, dim=-1) / TOPK
        BT = B * T
        flat_w = weights.reshape(BT, TOPK); flat_i = topk.indices.reshape(BT, TOPK)
        g_dense = torch.zeros(BT, N_EXPERTS, dtype=torch.float32)
        g_dense.scatter_(1, flat_i, flat_w)
        x_np = norm_out.reshape(BT, D_MODEL, 1, 1).to(torch.float16).numpy()
        g_np = g_dense.reshape(BT, N_EXPERTS, 1, 1).to(torch.float16).numpy()
        pack_batch = mlmodel.get_spec().description.input[0].type.multiArrayType.shape[0]
        out_chunks = []
        for s in range(0, BT, pack_batch):
            e = min(s + pack_batch, BT); xc = x_np[s:e]; gc = g_np[s:e]
            if xc.shape[0] < pack_batch:
                pad_n = pack_batch - xc.shape[0]
                xc = np.concatenate([xc, np.zeros((pad_n, D_MODEL, 1, 1), np.float16)], 0)
                gc = np.concatenate([gc, np.zeros((pad_n, N_EXPERTS, 1, 1), np.float16)], 0)
            r = mlmodel.predict({"x_in": xc, "g_in": gc})["x_out"]
            out_chunks.append(r[: e - s])
        pred = np.concatenate(out_chunks, axis=0)
        pred_t = torch.from_numpy(pred.astype(np.float32)).reshape(B, T, D_MODEL)
        return (x_3d + (pred_t * TOPK).to(x_3d.dtype)).reshape(orig_shape)
    return hook


def stats_ms(samples_ns):
    a = np.array(samples_ns, dtype=np.float64) / 1e6
    return {
        "n": len(a),
        "median": float(np.median(a)),
        "p25":    float(np.percentile(a, 25)),
        "p75":    float(np.percentile(a, 75)),
        "min":    float(a.min()),
        "max":    float(a.max()),
        "p90":    float(np.percentile(a, 90)),
        "p99":    float(np.percentile(a, 99)),
    }


def fmt(s):
    return (f"  median={s['median']:.1f} ms  IQR=[{s['p25']:.1f}, {s['p75']:.1f}]  "
            f"min={s['min']:.1f}  max={s['max']:.1f}  p90={s['p90']:.1f}  "
            f"p99={s['p99']:.1f} (n=20, indicative)")


def power_source():
    try:
        out = subprocess.check_output(["pmset", "-g", "ps"], text=True, timeout=2)
        return out.strip().split("\n")[0]
    except Exception:
        return "unknown"


def run_config(label, compute_unit, input_ids, attn_mask, pad_add_per):
    import coremltools as ct
    print(f"\n[bench] === config: {label} (compute_units={compute_unit}) ===")
    print("[bench]   loading 8 attn + 8 moe packs...")
    attn_packs, moe_packs = [], []
    for n in range(N_LAYERS):
        attn_packs.append(ct.models.MLModel(
            str(PKG_DIR / ATTN_FMT.format(n=n)), compute_units=compute_unit))
        moe_packs.append(ct.models.MLModel(
            str(PKG_DIR / MOE_FMT.format(n=n)), compute_units=compute_unit))

    from opf._model.model import Transformer
    print("[bench]   loading opf Transformer (CPU bf16)")
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=torch.device("cpu"))
    model.eval()
    handles = []
    for n in range(N_LAYERS):
        handles.append(model.block[n].attn.register_forward_hook(
            make_attn_hook(attn_packs[n], pad_add_per)))
        handles.append(model.block[n].mlp.register_forward_hook(
            make_moe_hook(moe_packs[n])))

    samples = []
    with torch.inference_mode():
        for i in range(WARMUP + ITERS):
            t0 = time.perf_counter_ns()
            _ = model(input_ids, attention_mask=attn_mask)
            dt = time.perf_counter_ns() - t0
            if i >= WARMUP:
                samples.append(dt)
            tag = "warmup" if i < WARMUP else "timed "
            print(f"[bench]   {tag} {i:2d}: {dt/1e6:7.1f} ms")
    for h in handles:
        h.remove()
    return stats_ms(samples)


def main() -> int:
    print(f"[bench] T6-A privacy-filter full ANE chain latency")
    print(f"[bench] platform: {platform.platform()}")
    print(f"[bench] python:   {sys.version.split()[0]}")
    print(f"[bench] threads:  torch={torch.get_num_threads()}  (will pin to 8)")
    print(f"[bench] power:    {power_source()}")
    torch.set_num_threads(8); torch.manual_seed(0)

    if not GOLDEN.exists():
        raise SystemExit(f"missing {GOLDEN}")
    z = np.load(GOLDEN, allow_pickle=False)
    input_ids = torch.from_numpy(z["input_ids"].astype(np.int64))
    attn_mask = torch.from_numpy(z["attention_mask"].astype(np.int64))
    B, T = input_ids.shape
    n_valid = int(attn_mask.sum().item())
    print(f"[bench] input shape: B={B}  T={T}  total valid tokens={n_valid}")

    mask_np = attn_mask.numpy().astype(np.int32)
    pad_add_per = [np.where(mask_np[i] > 0, 0.0, -1e4)
                   .astype(np.float16)[None, None, None, :] for i in range(B)]

    import coremltools as ct
    s_ane = run_config("ANE chain", ct.ComputeUnit.CPU_AND_NE,
                       input_ids, attn_mask, pad_add_per)
    s_cpu = run_config("CPU chain", ct.ComputeUnit.CPU_ONLY,
                       input_ids, attn_mask, pad_add_per)

    print(f"\n[bench] === RESULT ===")
    print(f"[bench] ANE chain (CPU_AND_NE):"); print(fmt(s_ane))
    print(f"[bench] CPU chain (CPU_ONLY):");   print(fmt(s_cpu))
    speedup = s_cpu["median"] / s_ane["median"]
    tok_per_sec_ane = (n_valid * 1000.0) / s_ane["median"]
    tok_per_sec_cpu = (n_valid * 1000.0) / s_cpu["median"]
    print(f"\n[bench] speedup (median CPU / median ANE) = {speedup:.2f}×")
    print(f"[bench] tok/s ANE = {tok_per_sec_ane:.1f}  "
          f"tok/s CPU = {tok_per_sec_cpu:.1f}  "
          f"(over {n_valid} valid tokens / forward)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
