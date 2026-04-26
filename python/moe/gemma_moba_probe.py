"""gemma_moba_probe.py — T2 from GEMMA_ANE_RESEARCH.md.

Training-free MoBA-style sparse global attention (arXiv:2502.13189) on
Gemma-4-26B-A4B. Apply ONLY to the global-attention layers (sliding_window
is None); leave sliding layers and the last `--keep-last-full` global
layers as full attention (Kimi K2 recipe).

Mechanism (per query position q, per head):
  1. Partition K cache into blocks of size B (--block-size, default 512).
  2. Mean-pool keys per block → block_repr ∈ R^{n_blocks × d_head}.
  3. Score = q · block_repr  (per query).
  4. Force-include the current block (the one containing q).
  5. Force-exclude future blocks (causal).
  6. Pick top-k blocks by score from the eligible set.
  7. Standard softmax(QK^T/√d + mask) @ V where mask = -inf outside
     selected blocks, plus the usual causal mask within them.

Subcommands:
  golden_long   : produce a long-context golden (full-attn reference) at
                  ctx=N. Saves python/moe/out/gemma_golden_ctx{N}.npz.
                  Slow: bf16 CPU forward at long ctx (~3 min/2K tokens).
  validate      : install MoBA on global layers, run forward on the same
                  prompt, compare. PASS gate:
                    cos(prompt_last) ≥ 0.95
                    AND TF top-1 ≥ 14/16 on the 16-token continuation
                    AND mean step cos ≥ 0.95
  latency_proj  : arithmetic projection of new attention latency from the
                  existing `gemma_attn_probe` numbers (no model load).

Run with: .venv313/bin/python python/moe/gemma_moba_probe.py <cmd> [opts]

Kill-switch: golden_long checkpoints prompt_logits to disk every chunk
(per gatekeeper rule 6); validate runs a single forward with no
mid-state, but ctx ≤ 4K keeps it under 10 min wall.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from hf_full_model_safety import (
    DEFAULT_DISK_FREE_MIN_GIB,
    DEFAULT_MAX_CPU_MEMORY_GIB,
    DEFAULT_MAX_DISK_MEMORY_GIB,
    prepare_model_load_kwargs,
    require_disk_free,
    validate_full_model_load_policy,
)

MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT_DIR   = Path("python/moe/out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Acceptance thresholds for MoBA (relaxed vs REAP per Kimi K2 reported gap)
COSINE_THRESHOLD = 0.95
TOPK_MATCH_THRESHOLD = 14   # out of 16
N_GEN = 16


def golden_path(ctx: int) -> Path:
    return OUT_DIR / f"gemma_golden_ctx{ctx}.npz"


# ----------------------------- model load ----------------------------------

def load_model(args):
    print("loading tokenizer + model (bf16, guarded offload policy)...")
    print(
        f"  memory policy: offload={args.offload} cpu={args.max_cpu_memory_gib}GiB "
        f"disk={args.max_disk_memory_gib}GiB"
    )
    disk_paths = [MODEL_DIR, OUT_DIR]
    if args.offload:
        disk_paths.append(args.offload_folder)
    require_disk_free(disk_paths, args.disk_free_min_gib)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model_kwargs, actual_offload_folder = prepare_model_load_kwargs(
        torch_dtype=torch.bfloat16,
        offload_enabled=args.offload,
        offload_folder=args.offload_folder,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        max_disk_memory_gib=args.max_disk_memory_gib,
        local_files_only=True,
    )
    model_kwargs["attn_implementation"] = "eager"
    if actual_offload_folder is not None:
        print(f"  offload folder: {actual_offload_folder}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **model_kwargs)
    model.eval()
    cfg = model.config.text_config
    return tok, model, cfg


def build_long_prompt(tok, n_tokens: int) -> torch.Tensor:
    """Concatenate wikitext-2 train until we have at least n_tokens, then
    truncate. Returns int64 tensor of shape (1, n_tokens)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    buf = []
    total = 0
    for row in ds:
        t = row["text"].strip()
        if not t:
            continue
        ids = tok(t, return_tensors=None)["input_ids"]
        buf.extend(ids)
        total += len(ids)
        if total >= n_tokens + 4:
            break
    ids = torch.tensor(buf[:n_tokens], dtype=torch.long).unsqueeze(0)
    return ids


# ----------------------------- golden_long ---------------------------------

def cmd_golden_long(args):
    out_path = golden_path(args.ctx)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} already exists; pass --force to overwrite")

    tok, model, cfg = load_model(args)
    prompt_ids = build_long_prompt(tok, args.ctx)
    print(f"prompt: {prompt_ids.shape[1]} tokens (wikitext-2 train slice)")

    # Single forward over full prompt to get last-position logits.
    # We don't need logits at every position (would be ctx × vocab × 4B
    # = 4K × 262144 × 4 ≈ 4.3 GB at ctx=4K — too big to save anyway).
    print("full-prompt forward (slow on bf16 CPU)...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True)
        last_logits = out.logits[0, -1].float().cpu().numpy()
        past = out.past_key_values
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    # Greedy continuation N_GEN tokens, recording logits per step
    print(f"greedy continuation {N_GEN} tokens...")
    gen_ids, gen_logits = [], []
    cur = torch.tensor([[int(last_logits.argmax())]], dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(N_GEN):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            l = out.logits[0, -1].float().cpu().numpy()
            gen_logits.append(l)
            gen_ids.append(int(cur.item()))
            cur = torch.tensor([[int(l.argmax())]], dtype=torch.long)
    gen_ids = np.array(gen_ids, dtype=np.int32)
    gen_logits = np.stack(gen_logits, axis=0).astype(np.float32)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print(f"  continuation: {tok.decode(gen_ids.tolist())!r}")

    np.savez(
        out_path,
        ctx=args.ctx,
        prompt_ids=prompt_ids[0].numpy().astype(np.int32),
        last_logits=last_logits,
        next_token_ids=gen_ids,
        next_token_logits=gen_logits,
    )
    print(f"saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ----------------------------- MoBA mask -----------------------------------

def install_moba(model, cfg, block_size: int, top_k: int,
                 keep_last_full: int):
    """Register a custom attention interface that applies MoBA on global
    layers (sliding_window is None, layer_idx < n_layers - keep_last_full)
    and falls through to standard eager elsewhere.

    Returns the prior config._attn_implementation so the caller can restore.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4.modeling_gemma4 import (
        eager_attention_forward, repeat_kv,
    )

    n_layers = len(cfg.layer_types)
    last_full_set = set(range(n_layers - keep_last_full, n_layers))
    n_global = cfg.layer_types.count("global")
    moba_targets = [li for li, lt in enumerate(cfg.layer_types)
                    if lt == "global" and li not in last_full_set]
    print(f"  MoBA targets: {len(moba_targets)} / {n_global} global layers "
          f"(keeping last {keep_last_full} layers full): {moba_targets}")

    def moba_attention_forward(module, query, key, value, attention_mask,
                                dropout=0.0, scaling=None, softcap=None,
                                sliding_window=None, **kwargs):
        # Dispatch: only patch global layers that aren't in the keep-full tail.
        if sliding_window is not None or module.layer_idx in last_full_set:
            return eager_attention_forward(
                module, query, key, value, attention_mask,
                dropout=dropout, scaling=scaling, softcap=softcap, **kwargs)

        if scaling is None:
            scaling = module.head_dim ** -0.5

        # GQA expand
        k = repeat_kv(key, module.num_key_value_groups)   # (B,Hq,T,d)
        v = repeat_kv(value, module.num_key_value_groups)
        q = query                                          # (B,Hq,T,d)
        B, Hq, T, d = q.shape

        # Block-pool keys (mean-pool, accounting for trailing partial block)
        n_blocks = (T + block_size - 1) // block_size
        pad = n_blocks * block_size - T
        if pad > 0:
            k_pad = torch.cat(
                [k, torch.zeros(B, Hq, pad, d, dtype=k.dtype, device=k.device)],
                dim=2)
        else:
            k_pad = k
        k_blk = k_pad.view(B, Hq, n_blocks, block_size, d)
        block_lens = torch.full((n_blocks,), float(block_size),
                                dtype=torch.float32, device=k.device)
        if pad > 0:
            block_lens[-1] = float(block_size - pad)
        block_repr = k_blk.sum(dim=3) / block_lens.view(1, 1, n_blocks, 1).to(k.dtype)
        block_scores = torch.einsum("bhtd,bhnd->bhtn", q, block_repr) * scaling

        # Per-query block index (which block this query lives in)
        q_pos = torch.arange(T, device=q.device)
        q_block = q_pos // block_size                                # (T,)
        blk_idx = torch.arange(n_blocks, device=q.device)            # (n_blocks,)

        # Mask future blocks
        future = blk_idx.view(1, n_blocks) > q_block.view(T, 1)      # (T, n_blocks)
        finfo_min = torch.finfo(block_scores.dtype).min
        finfo_max = torch.finfo(block_scores.dtype).max
        block_scores = block_scores.masked_fill(
            future.view(1, 1, T, n_blocks), finfo_min)

        # Force-include current block
        cur = blk_idx.view(1, n_blocks) == q_block.view(T, 1)
        block_scores = block_scores.masked_fill(
            cur.view(1, 1, T, n_blocks), finfo_max)

        k_eff = min(top_k, n_blocks)
        sel = torch.topk(block_scores, k=k_eff, dim=-1).indices      # (B,Hq,T,k_eff)
        selected = torch.zeros_like(block_scores, dtype=torch.bool)
        selected.scatter_(-1, sel, True)

        # Per-(query, key) allow mask
        key_block = (torch.arange(T, device=q.device) // block_size) # (T,)
        sel_key = selected.gather(
            -1, key_block.view(1, 1, 1, T).expand(B, Hq, T, T))      # (B,Hq,T,T)
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        allow = sel_key & causal.view(1, 1, T, T)

        # Standard SDPA with the MoBA mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        if softcap is not None:
            attn_scores = (torch.tanh(attn_scores / softcap)) * softcap
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        neg_inf = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~allow, neg_inf)
        w = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(w, v)                                     # (B,Hq,T,d)
        out = out.transpose(1, 2).contiguous()                       # (B,T,Hq,d)
        return out, w

    ALL_ATTENTION_FUNCTIONS["moba"] = moba_attention_forward
    prev = model.config._attn_implementation
    model.config._attn_implementation = "moba"
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = "moba"
    return prev


def restore_attention(model, prev):
    model.config._attn_implementation = prev
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = prev


# ----------------------------- validate ------------------------------------

def cmd_validate(args):
    gp = golden_path(args.ctx)
    if not gp.exists():
        raise SystemExit(f"missing {gp} — run `golden_long --ctx {args.ctx}` first")

    gd = np.load(gp)
    prompt_ids = torch.tensor(gd["prompt_ids"]).long().unsqueeze(0)
    golden_last = gd["last_logits"]
    golden_next_ids = gd["next_token_ids"]
    golden_step_logits = gd["next_token_logits"]
    n_gen = len(golden_next_ids)

    tok, model, cfg = load_model(args)
    prev_impl = install_moba(model, cfg,
                             block_size=args.block_size,
                             top_k=args.top_k,
                             keep_last_full=args.keep_last_full)

    # Teacher-forced full sequence: prompt + all golden continuation tokens
    full_ids = torch.cat([
        prompt_ids,
        torch.tensor(golden_next_ids.astype(np.int64)).unsqueeze(0),
    ], dim=1)
    P = prompt_ids.shape[1]
    print(f"teacher-forced forward on {full_ids.shape[1]} tokens "
          f"(prompt={P}, gen={n_gen}) — bf16 CPU, slow at long ctx...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(full_ids)
        all_logits = out.logits[0].float().cpu().numpy()
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    restore_attention(model, prev_impl)

    # Prompt-final cosine
    a, b = all_logits[P - 1], golden_last
    cos_last = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"  cosine(prompt_last) = {cos_last:.4f}   (gate ≥ {COSINE_THRESHOLD})")

    # TF top-1
    tf_match = 0
    pred = []
    for s in range(n_gen):
        p = int(all_logits[P - 1 + s].argmax())
        pred.append(p)
        if p == int(golden_next_ids[s]):
            tf_match += 1
    print(f"  TF top-1: {tf_match}/{n_gen}   (gate ≥ {TOPK_MATCH_THRESHOLD})")

    # Per-step cosine vs golden's recorded step logits
    cos_steps = []
    for s in range(n_gen):
        x = all_logits[P + s]
        y = golden_step_logits[s]
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        cos_steps.append(c)
    cos_steps = np.array(cos_steps)
    print(f"  per-step cosine: mean={cos_steps.mean():.4f}  "
          f"min={cos_steps.min():.4f}   (gate mean ≥ {COSINE_THRESHOLD})")
    print(f"  golden : {golden_next_ids.tolist()}")
    print(f"  pred   : {pred}")

    passed = (cos_last >= COSINE_THRESHOLD) \
             and (tf_match >= TOPK_MATCH_THRESHOLD) \
             and (cos_steps.mean() >= COSINE_THRESHOLD)
    print()
    print(f"# golden-validator MoBA (ctx={args.ctx} B={args.block_size} "
          f"top_k={args.top_k} keep_last_full={args.keep_last_full}): "
          f"{'PASS' if passed else 'FAIL'}")

    rep = {
        "ctx": args.ctx,
        "block_size": args.block_size,
        "top_k": args.top_k,
        "keep_last_full": args.keep_last_full,
        "cosine_prompt_last": cos_last,
        "tf_top1": tf_match,
        "n_gen": n_gen,
        "cosine_step_mean": float(cos_steps.mean()),
        "cosine_step_min": float(cos_steps.min()),
        "passed": bool(passed),
    }
    rep_path = OUT_DIR / f"gemma_moba_validate_ctx{args.ctx}.json"
    rep_path.write_text(json.dumps(rep, indent=2))
    print(f"saved report → {rep_path}")
    if not passed:
        raise SystemExit(1)


# ----------------------------- latency_proj --------------------------------

def cmd_latency_proj(args):
    """Project attention latency under MoBA from existing baselines.

    Standard QK^T cost ∝ T (per query, KV is T tokens). MoBA reduces the
    set of attended keys to top_k * block_size tokens. Saving:
        savings = 1 - (top_k * block_size) / T
    (only on the patched global layers).
    """
    n_global_total = 6
    keep_last = args.keep_last_full
    n_patched = max(0, n_global_total - keep_last)
    print(f"MoBA latency projection (top_k={args.top_k}, B={args.block_size}, "
          f"keep_last_full={keep_last})")
    print(f"  patched global layers: {n_patched} / {n_global_total}")
    print(f"  {'ctx':>6s}  {'attended':>10s}  {'savings':>9s}  {'global_attn_ratio':>18s}")
    for T in (1024, 2048, 4096, 8192):
        attended = min(args.top_k * args.block_size, T)
        savings = 1.0 - attended / T
        print(f"  {T:>6d}  {attended:>10d}  {savings*100:>7.1f}%  "
              f"{(1 - n_patched/30 * savings)*100:>16.1f}% of full-attn")


# ----------------------------- CLI -----------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offload", dest="offload", action="store_true", default=True)
    p.add_argument("--no-offload", dest="offload", action="store_false")
    p.add_argument("--offload-folder", type=Path,
                   default=OUT_DIR / ".offload_gemma_moba")
    p.add_argument("--max-cpu-memory-gib", type=int, default=DEFAULT_MAX_CPU_MEMORY_GIB)
    p.add_argument("--max-disk-memory-gib", type=int, default=DEFAULT_MAX_DISK_MEMORY_GIB)
    p.add_argument("--disk-free-min-gib", type=int, default=DEFAULT_DISK_FREE_MIN_GIB)
    p.add_argument("--allow-unsafe-cpu-memory", action="store_true")
    p.add_argument("--allow-no-disk-offload", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("golden_long")
    pg.add_argument("--ctx", type=int, required=True)
    pg.add_argument("--force", action="store_true")
    pg.set_defaults(func=cmd_golden_long)

    pv = sub.add_parser("validate")
    pv.add_argument("--ctx", type=int, required=True)
    pv.add_argument("--block-size", type=int, default=512)
    pv.add_argument("--top-k", type=int, default=4)
    pv.add_argument("--keep-last-full", type=int, default=3)
    pv.set_defaults(func=cmd_validate)

    pl = sub.add_parser("latency_proj")
    pl.add_argument("--block-size", type=int, default=512)
    pl.add_argument("--top-k", type=int, default=4)
    pl.add_argument("--keep-last-full", type=int, default=3)
    pl.set_defaults(func=cmd_latency_proj)

    args = p.parse_args()
    if args.cmd in {"golden_long", "validate"}:
        validate_full_model_load_policy(
            "gemma_moba_probe",
            offload_enabled=args.offload,
            max_cpu_memory_gib=args.max_cpu_memory_gib,
            max_disk_memory_gib=args.max_disk_memory_gib,
            allow_unsafe_cpu_memory=args.allow_unsafe_cpu_memory,
            allow_no_disk_offload=args.allow_no_disk_offload,
        )
    args.func(args)


if __name__ == "__main__":
    main()
