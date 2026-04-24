"""gemma_reap.py — T1.1 from GEMMA_ANE_RESEARCH.md.

REAP expert pruning for Gemma-4-26B-A4B (arXiv:2510.13999).

Pipeline (driven by --mode):
  calibrate    : run a small forward pass, hook every router, accumulate
                 score[layer, expert] = sum_tokens softmax(router_logits)[expert]
                 over the top_k=8 selected experts. Saves
                 python/moe/out/gemma_reap_scores.npz.
  mask         : load scores, keep top-K experts per layer (default 64),
                 save python/moe/out/gemma_reap_mask.npz.
  validate     : load mask, re-run reference forward with dropped experts
                 hard-masked (set router logit to -inf), compare last-token
                 logits cosine + greedy continuation top-1 agreement vs
                 python/moe/out/gemma_golden.npz. PASS if cosine >= 0.97
                 and >= 14/16 top-1 matches. PRINTS verdict; does NOT
                 promote the mask if FAIL.

Run with: .venv313/bin/python python/moe/gemma_reap.py <mode> [opts]

This is the simpler "router-gate-weighted" REAP variant. The full paper
also multiplies by ||expert_output||_2; that requires monkey-patching
the fused experts.gate_up_proj forward and is left for v2 if v1 fails
the cosine gate.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR  = Path("models/gemma-4-26b-a4b")
OUT_DIR    = Path("python/moe/out"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SCORES_NPZ = OUT_DIR / "gemma_reap_scores.npz"
MASK_NPZ   = OUT_DIR / "gemma_reap_mask.npz"
GOLDEN_NPZ = OUT_DIR / "gemma_golden.npz"

# Tiny smoke text (used only when --corpus smoke is requested).
SMOKE_TEXT = (
    "The Apple Neural Engine is a 16-core specialized accelerator that excels "
    "at int8 and int4 matrix multiplications. In this study we benchmark how "
    "well a sparse mixture-of-experts model maps onto it, focusing on routing "
    "locality, weight-streaming bandwidth, and end-to-end token throughput.\n\n"
    "We find that the dominant cost at long context is attention KV streaming, "
    "while expert MLPs remain cache-resident when packed in groups of eight or "
    "more INT4-quantized experts per program. The router gate distribution is "
    "long-tailed: a small subset of experts dominates per-token activation, "
    "suggesting REAP-style pruning can recover ~50% of compute without loss."
)

# Acceptance thresholds (gatekeeper contract)
COSINE_THRESHOLD = 0.97
TOPK_MATCH_THRESHOLD = 14  # out of 16


# ----------------------------- helpers --------------------------------------

def load_model():
    print("loading tokenizer + model (bf16, CPU)...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    cfg = model.config.text_config
    return tok, model, cfg


def find_router_modules(model, n_experts: int):
    """Return list of (layer_idx, router_module). Mirrors the fallback path
    in gemma_inspect.py since `.router` is not a plain Linear."""
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features == n_experts:
            try:
                layer_idx = int(name.split(".layers.")[1].split(".")[0])
            except Exception:
                continue
            out.append((layer_idx, mod, name))
    out.sort(key=lambda t: t[0])
    return out


# ----------------------------- calibrate ------------------------------------

def _load_calib_ids(tok, args):
    """Return a 1-D LongTensor of token ids for calibration."""
    if args.corpus == "smoke":
        ids = tok(SMOKE_TEXT, return_tensors="pt").input_ids[0]
    elif args.corpus == "wikitext":
        from datasets import load_dataset  # lazy import
        print("  loading wikitext-2-raw-v1 (train) ...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(x for x in ds["text"] if x.strip())
        ids = tok(text, return_tensors="pt").input_ids[0]
    else:
        raise SystemExit(f"unknown --corpus {args.corpus!r}")
    print(f"  corpus={args.corpus} total_tokens={ids.shape[0]:,}  "
          f"requested={args.calib_tokens:,}")
    return ids[: args.calib_tokens]


def cmd_calibrate(args):
    tok, model, cfg = load_model()
    n_experts = cfg.num_experts
    top_k = cfg.top_k_experts
    n_layers = cfg.num_hidden_layers
    print(f"  n_experts={n_experts}  top_k={top_k}  n_layers={n_layers}")

    routers = find_router_modules(model, n_experts)
    print(f"  found {len(routers)} router linears")

    scores = np.zeros((n_layers, n_experts), dtype=np.float64)
    counts = np.zeros((n_layers,), dtype=np.int64)

    def make_hook(layer_idx):
        def hook(_mod, _inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            flat = logits.reshape(-1, logits.shape[-1]).float()  # (T, E)
            probs = torch.softmax(flat, dim=-1)
            topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
            # Vectorized scatter-add (avoid the T*top_k python loop that
            # made multi-thousand-token calibration intractable in v1).
            ti = topk_idx.cpu().numpy().reshape(-1)            # (T*top_k,)
            tv = topk_vals.cpu().numpy().reshape(-1).astype(np.float64)
            np.add.at(scores[layer_idx], ti, tv)
            counts[layer_idx] += topk_idx.shape[0]
        return hook

    handles = [mod.register_forward_hook(make_hook(li)) for li, mod, _ in routers]

    ids = _load_calib_ids(tok, args)
    seq_len = args.seq_len
    n_chunks = math.ceil(ids.shape[0] / seq_len)
    print(f"  forwarding {ids.shape[0]:,} tokens in {n_chunks} chunk(s) of {seq_len}")
    actual_tokens = 0
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0)
            t0 = time.time()
            model(chunk)
            actual_tokens += chunk.shape[1]
            print(f"    chunk {i+1}/{n_chunks}  tokens={chunk.shape[1]}  "
                  f"elapsed={time.time()-t0:.1f}s")
            # Per-chunk checkpoint: a long forward can OOM or be ctrl-C'd;
            # this makes each chunk's contribution durable (gatekeeper rule 6).
            np.savez(SCORES_NPZ,
                     scores=scores, counts=counts,
                     n_experts=n_experts, top_k=top_k, n_layers=n_layers,
                     calib_tokens=actual_tokens,
                     corpus=args.corpus, seq_len=seq_len)
    for h in handles:
        h.remove()

    np.savez(SCORES_NPZ,
             scores=scores, counts=counts,
             n_experts=n_experts, top_k=top_k, n_layers=n_layers,
             calib_tokens=actual_tokens,
             corpus=args.corpus, seq_len=seq_len)
    print(f"saved → {SCORES_NPZ}  ({actual_tokens:,} tokens, corpus={args.corpus})")
    print(f"  per-layer score stats (min/median/max):")
    for li in range(n_layers):
        s = scores[li]
        print(f"    L{li:02d}  min={s.min():.4f}  med={np.median(s):.4f}  "
              f"max={s.max():.4f}  nonzero={(s>0).sum()}/{n_experts}")


# ----------------------------- mask -----------------------------------------

def cmd_mask(args):
    if not SCORES_NPZ.exists():
        raise SystemExit(f"missing {SCORES_NPZ} — run `calibrate` first")
    d = np.load(SCORES_NPZ)
    scores = d["scores"]                      # (L, E)
    n_layers, n_experts = scores.shape
    keep = args.keep
    if keep <= 0 or keep > n_experts:
        raise SystemExit(f"--keep must be in (0, {n_experts}]")

    # Top-keep per layer by score
    keep_idx = np.argsort(-scores, axis=1)[:, :keep]   # (L, keep)
    keep_idx.sort(axis=1)
    drop_mask = np.ones_like(scores, dtype=bool)
    for li in range(n_layers):
        drop_mask[li, keep_idx[li]] = False

    # How much of the gate mass do we drop?
    total_mass = scores.sum(axis=1)
    dropped_mass = (scores * drop_mask).sum(axis=1)
    frac_dropped = dropped_mass / np.clip(total_mass, 1e-12, None)

    np.savez(MASK_NPZ,
             keep_idx=keep_idx, drop_mask=drop_mask,
             frac_gate_mass_dropped=frac_dropped,
             keep_per_layer=keep)
    print(f"saved → {MASK_NPZ}")
    print(f"  keep={keep}/{n_experts} per layer ({keep/n_experts*100:.1f}%)")
    print(f"  fraction of gate mass dropped per layer (lower = safer):")
    for li in range(n_layers):
        print(f"    L{li:02d}  {frac_dropped[li]*100:5.2f}%")
    print(f"  mean dropped mass: {frac_dropped.mean()*100:.2f}%")


# ----------------------------- validate -------------------------------------

def cmd_validate(args):
    if not MASK_NPZ.exists():
        raise SystemExit(f"missing {MASK_NPZ} — run `mask` first")
    if not GOLDEN_NPZ.exists():
        raise SystemExit(f"missing {GOLDEN_NPZ} — run `gemma_ref.py` first")

    md = np.load(MASK_NPZ)
    drop_mask = md["drop_mask"]               # (L, E) bool
    gd = np.load(GOLDEN_NPZ)
    prompt_ids = torch.tensor(gd["prompt_ids"]).long().unsqueeze(0)
    golden_last_logits = gd["logits_full"][-1]
    golden_next_ids = gd["next_token_ids"]
    n_gen = len(golden_next_ids)

    tok, model, cfg = load_model()
    routers = find_router_modules(model, cfg.num_experts)

    # Hook each router output and add -inf to dropped expert positions
    NEG_INF = torch.tensor(-1e9, dtype=torch.bfloat16)

    def make_mask_hook(layer_idx):
        drop_row = torch.tensor(drop_mask[layer_idx])  # (E,) bool
        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                logits = out[0]
                rest = out[1:]
                logits = logits.masked_fill(drop_row.to(logits.device), NEG_INF.to(logits.dtype))
                return (logits, *rest)
            return out.masked_fill(drop_row.to(out.device), NEG_INF.to(out.dtype))
        return hook

    handles = [mod.register_forward_hook(make_mask_hook(li)) for li, mod, _ in routers]

    # Build teacher-forced sequence: prompt + ALL n_gen golden tokens.
    # Index map (prompt_len = P):
    #   all_logits[P-1]     predicts next_token_ids[0]   ← matches golden_last_logits
    #   all_logits[P-1+s]   predicts next_token_ids[s]   for s in 0..n_gen-1   (top-1 check)
    #   all_logits[P+s]     predicts the token AFTER next_token_ids[s]         (cosine vs
    #                                                       golden's next_token_logits[s])
    prompt_len = prompt_ids.shape[1]
    full_ids = torch.cat([
        prompt_ids,
        torch.tensor(golden_next_ids.astype(np.int64)).unsqueeze(0),
    ], dim=1)  # (1, prompt_len + n_gen)

    print(f"teacher-forced forward on {full_ids.shape[1]} tokens "
          f"(prompt={prompt_len}, gen={n_gen})...")
    with torch.no_grad():
        out = model(full_ids)
        all_logits = out.logits[0].float().cpu().numpy()  # (T, V)
    for h in handles:
        h.remove()

    # Prompt-final cosine (apples-to-apples with golden_last_logits)
    logits_last = all_logits[prompt_len - 1]
    a, b = logits_last, golden_last_logits
    cos_last = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    argmax_pruned = int(logits_last.argmax())
    argmax_golden = int(golden_last_logits.argmax())
    print(f"  cosine(prompt_last) = {cos_last:.4f}   (threshold {COSINE_THRESHOLD})")
    print(f"  argmax pruned={argmax_pruned}  golden={argmax_golden}  "
          f"{'MATCH' if argmax_pruned == argmax_golden else 'MISS'}")

    # Top-1 under teacher forcing: at each step s, after seeing prompt + golden[:s],
    # does the model's argmax match next_token_ids[s]?
    tf_top1_match = 0
    pred_ids = []
    for s in range(n_gen):
        p = int(all_logits[prompt_len - 1 + s].argmax())
        pred_ids.append(p)
        if p == int(golden_next_ids[s]):
            tf_top1_match += 1

    # Per-step cosine vs golden's recorded next_token_logits.
    # Golden's next_token_logits[s] was produced AFTER feeding next_token_ids[s];
    # in our TF forward that's all_logits[prompt_len + s].
    golden_step_logits = gd["next_token_logits"]  # (n_gen, V)
    cos_steps = []
    for s in range(n_gen):
        x = all_logits[prompt_len + s]
        y = golden_step_logits[s]
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        cos_steps.append(c)
    cos_steps = np.array(cos_steps)
    print(f"  per-step cosine (teacher-forced): mean={cos_steps.mean():.4f} "
          f"min={cos_steps.min():.4f}  (threshold {COSINE_THRESHOLD})")
    print(f"  teacher-forced top-1: {tf_top1_match}/{n_gen}   (threshold {TOPK_MATCH_THRESHOLD})")
    print(f"  golden : {golden_next_ids.tolist()}")
    print(f"  pred   : {pred_ids}")
    print(f"  cos/step: " + " ".join(f"{c:.3f}" for c in cos_steps))

    # Gate (matches golden-validator.agent.md):
    #   cos(prompt_last) >= 0.97  AND  TF_top1 >= 14/16
    # mean_step_cos is reported but not gated (advisory only).
    # min_step_cos is reported but NOT gated: over open-ended continuation a
    # single near-tie at a high-entropy position (e.g. sentence-final
    # punctuation) makes the dense model itself low-cosine vs its own
    # snapshot; gating on min penalises distribution entropy, not pruning damage.
    passed = (cos_last >= COSINE_THRESHOLD) \
             and (tf_top1_match >= TOPK_MATCH_THRESHOLD)
    print()
    keep_n = drop_mask.shape[1] - int(drop_mask.sum(axis=1).max())
    print(f"# golden-validator (REAP keep={keep_n}, teacher-forced): "
          f"{'PASS' if passed else 'FAIL'}")

    report = {
        "cosine_prompt_last": cos_last,
        "cosine_step_mean": float(cos_steps.mean()),
        "cosine_step_min": float(cos_steps.min()),
        "cosine_per_step": cos_steps.tolist(),
        "teacher_forced_top1_matches": tf_top1_match,
        "n_gen": n_gen,
        "argmax_match": argmax_pruned == argmax_golden,
        "cosine_threshold": COSINE_THRESHOLD,
        "topk_threshold": TOPK_MATCH_THRESHOLD,
        "passed": bool(passed),
    }
    rep_path = OUT_DIR / "gemma_reap_validate.json"
    rep_path.write_text(json.dumps(report, indent=2))
    print(f"saved report → {rep_path}")

    if not passed:
        raise SystemExit(1)


# ----------------------------- pack_locality --------------------------------

LOCALITY_NPZ = OUT_DIR / "gemma_reap_locality.npz"
PACK_SIZES   = (4, 8, 16, 32)


def cmd_pack_locality(args):
    """Measure routing locality on the pruned model.

    For each MoE layer:
      1. Apply the keep-mask via NEG_INF on dropped router logits
         (same hook as `validate`) so top-8 is chosen only from survivors.
      2. Run a forward pass on a chunk of wikitext-2.
      3. For each token, count distinct packs spanned by its top-8 in two
         indexing schemes:
           - raw: pack_id = expert_id // G (original 128-expert space)
           - remap: surviving experts re-indexed to 0..K-1 in keep_idx order;
                    pack_id = remap[expert_id] // G  (compact K-expert space)
      4. Report mean distinct-packs per layer and overall.
    """
    if not MASK_NPZ.exists():
        raise SystemExit(f"missing {MASK_NPZ} — run `mask` first")
    md = np.load(MASK_NPZ)
    drop_mask = md["drop_mask"]                  # (L, E) bool
    keep_idx  = md["keep_idx"]                    # (L, K) int
    L, E = drop_mask.shape
    K = keep_idx.shape[1]

    # Per-layer remap table: original_expert_id -> compact_id (or -1 if dropped)
    remap = -np.ones((L, E), dtype=np.int64)
    for li in range(L):
        for compact, orig in enumerate(keep_idx[li]):
            remap[li, orig] = compact

    tok, model, cfg = load_model()
    n_experts = cfg.num_experts
    top_k     = cfg.top_k_experts
    assert n_experts == E and cfg.num_hidden_layers == L

    routers = find_router_modules(model, n_experts)

    # captured per-layer top-k indices (numpy, original space)
    routings: dict[int, list[np.ndarray]] = {}

    NEG_INF = torch.tensor(-1e9, dtype=torch.bfloat16)

    def make_hook(layer_idx):
        drop_row = torch.tensor(drop_mask[layer_idx])
        def hook(_mod, _inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            masked = logits.masked_fill(drop_row.to(logits.device),
                                        NEG_INF.to(logits.dtype))
            flat = masked.reshape(-1, masked.shape[-1]).float()
            tk = torch.topk(flat, k=top_k, dim=-1).indices.cpu().numpy()
            routings.setdefault(layer_idx, []).append(tk)
            # Return masked logits so downstream MoE sees the pruned routing.
            if isinstance(out, tuple):
                return (masked, *out[1:])
            return masked
        return hook

    handles = [mod.register_forward_hook(make_hook(li)) for li, mod, _ in routers]

    ids = _load_calib_ids(tok, args)
    seq_len = args.seq_len
    n_chunks = math.ceil(ids.shape[0] / seq_len)
    print(f"  forwarding {ids.shape[0]:,} tokens in {n_chunks} chunk(s) of {seq_len}")
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0)
            t0 = time.time()
            model(chunk)
            print(f"    chunk {i+1}/{n_chunks}  tokens={chunk.shape[1]}  "
                  f"elapsed={time.time()-t0:.1f}s")
    for h in handles:
        h.remove()

    # Compute per-layer mean distinct-packs in raw and remap spaces.
    print("\n  per-layer mean distinct-packs (raw 128-expert space)")
    print(f"    {'L':>3s}  " + "  ".join(f"G={g:>2d}" for g in PACK_SIZES))
    raw_summary    = np.zeros((L, len(PACK_SIZES)), dtype=np.float64)
    remap_summary  = np.zeros((L, len(PACK_SIZES)), dtype=np.float64)
    for li in sorted(routings):
        topk_all = np.concatenate(routings[li], axis=0)   # (T, top_k)
        T = topk_all.shape[0]
        # validate: every chosen expert is a kept expert (mask worked)
        kept_set = set(int(x) for x in keep_idx[li])
        leakage = int(sum(1 for v in topk_all.reshape(-1) if int(v) not in kept_set))
        if leakage:
            print(f"    L{li:02d}  WARNING leakage={leakage}/{topk_all.size}")
        compact = remap[li, topk_all]                     # (T, top_k)
        for gi, G in enumerate(PACK_SIZES):
            raw_summary[li, gi]   = np.mean(
                [len(set((topk_all[t] // G).tolist())) for t in range(T)])
            remap_summary[li, gi] = np.mean(
                [len(set((compact [t] // G).tolist())) for t in range(T)])
        print(f"    L{li:02d}  " + "  ".join(f"{v:5.2f}" for v in raw_summary[li]))

    print("\n  per-layer mean distinct-packs (REMAP 64-expert compact space)")
    print(f"    {'L':>3s}  " + "  ".join(f"G={g:>2d}" for g in PACK_SIZES))
    for li in sorted(routings):
        print(f"    L{li:02d}  " + "  ".join(f"{v:5.2f}" for v in remap_summary[li]))

    raw_mean   = raw_summary.mean(axis=0)
    remap_mean = remap_summary.mean(axis=0)
    print("\n  MEAN over layers:")
    print("    pack     " + "  ".join(f"G={g:>2d}" for g in PACK_SIZES))
    print(f"    raw      " + "  ".join(f"{v:5.2f}" for v in raw_mean))
    print(f"    remap K={K} " + "  ".join(f"{v:5.2f}" for v in remap_mean))

    # Compare to pre-prune baseline (gemma_routing.npz from gemma_inspect.py)
    pre_npz = OUT_DIR / "gemma_routing.npz"
    if pre_npz.exists():
        pre = np.load(pre_npz)
        pre_mean = pre["distinct_packs"].mean(axis=0)
        print(f"    pre-prune " + "  ".join(f"{v:5.2f}" for v in pre_mean))
        delta = remap_mean - pre_mean
        print(f"    Δ (remap-pre) " + "  ".join(f"{v:+5.2f}" for v in delta))

    np.savez(LOCALITY_NPZ,
             raw_distinct=raw_summary,
             remap_distinct=remap_summary,
             pack_sizes=np.array(PACK_SIZES),
             keep_per_layer=K, n_experts=E, top_k=top_k)
    print(f"\n  saved → {LOCALITY_NPZ}")


# ----------------------------- cli ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("calibrate", help="capture router-gate scores")
    a.add_argument("--corpus", choices=["wikitext", "smoke"], default="wikitext",
                   help="calibration corpus (default wikitext-2-raw-v1 train)")
    a.add_argument("--calib-tokens", type=int, default=8192,
                   help="cap total calibration tokens (default 8192)")
    a.add_argument("--seq-len", type=int, default=2048,
                   help="per-forward chunk size (default 2048)")
    a.set_defaults(func=cmd_calibrate)

    b = sub.add_parser("mask", help="derive top-K-per-layer keep mask")
    b.add_argument("--keep", type=int, default=64,
                   help="experts kept per layer (default 64 = 50%% prune)")
    b.set_defaults(func=cmd_mask)

    c = sub.add_parser("validate", help="cosine + top-1 vs golden, with mask applied")
    c.set_defaults(func=cmd_validate)

    d = sub.add_parser("pack_locality",
                       help="distinct-packs-per-token on pruned model")
    d.add_argument("--corpus", choices=["wikitext", "smoke"], default="wikitext")
    d.add_argument("--calib-tokens", type=int, default=2048,
                   help="tokens to forward (default 2048)")
    d.add_argument("--seq-len", type=int, default=2048)
    d.set_defaults(func=cmd_pack_locality)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
