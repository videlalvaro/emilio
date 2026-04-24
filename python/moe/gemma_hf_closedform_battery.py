"""gemma_hf_closedform_battery.py — REAP-aware HF factual battery capture.

Builds a low-entropy factual oracle to complement the original open-ended
decode golden. Each prompt is intended to force a short factual answer in the
first 1-2 generated tokens.

Writes:
  python/moe/out/gemma_hf_closedform_battery.npz
  python/moe/out/.gemma_hf_closedform_battery_PASS

Usage (must be .venv313):
  .venv313/bin/python python/moe/gemma_hf_closedform_battery.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

assert "venv313" in sys.executable, (
    f"Wrong interpreter: {sys.executable!r}; need .venv313/bin/python")

import numpy as np
import torch
import transformers

from gemma_closedform_battery_spec import (
    BATTERY,
    HF_OUT_NPZ,
    HF_SENTINEL,
    MIN_REQUIRED_MARGIN,
    N_NEW,
)

MODEL_DIR = Path("models/gemma-4-26b-a4b")
REAP_MASK = Path("python/moe/out/gemma_reap_mask.npz")

FREE_RAM_FLOOR_GB = 12.0


def _atomic_write_npz(target: Path, **arrays):
    tmp_base = target.with_suffix(target.suffix + ".tmp")
    tmp_written = Path(str(tmp_base) + ".npz")
    np.savez(str(tmp_base), **arrays)
    assert tmp_written.exists()
    with open(tmp_written, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_written, target)


def _load_partial_results() -> dict[int, dict]:
    if not HF_OUT_NPZ.exists():
        return {}
    z = np.load(HF_OUT_NPZ, allow_pickle=False)
    results = {}
    for i in range(len(BATTERY)):
        done_key = f"p{i}_done"
        if done_key not in z or not bool(z[done_key]):
            continue
        results[i] = {
            "prompt_ids": np.array(z[f"p{i}_prompt_ids"], dtype=np.int64),
            "top1_ids": np.array(z[f"p{i}_top1_ids"], dtype=np.int64),
            "top1_logits": np.array(z[f"p{i}_top1_logits"], dtype=np.float32),
            "top2_ids": np.array(z[f"p{i}_top2_ids"], dtype=np.int64),
            "top2_logits": np.array(z[f"p{i}_top2_logits"], dtype=np.float32),
            "gen_ids": np.array(z[f"p{i}_gen_ids"], dtype=np.int64),
            "required_min_margin": float(z[f"p{i}_required_min_margin"]),
            "stable": bool(z[f"p{i}_stable"]),
            "wall_s": float(z[f"p{i}_wall_s"]),
        }
    return results


def _write_partial(results: dict[int, dict], transformers_version: str):
    payload = {
        "n_prompts": np.array(len(BATTERY), dtype=np.int64),
        "n_new": np.array(N_NEW, dtype=np.int64),
        "transformers_version": np.array(transformers_version),
        "min_required_margin": np.array(MIN_REQUIRED_MARGIN, dtype=np.float32),
        "completed_prompts": np.array(len(results), dtype=np.int64),
    }
    for i, (key, prompt, required_prefix) in enumerate(BATTERY):
        payload[f"p{i}_key"] = np.array(key)
        payload[f"p{i}_prompt"] = np.array(prompt)
        payload[f"p{i}_required_prefix"] = np.array(required_prefix, dtype=np.int64)
        done = i in results
        payload[f"p{i}_done"] = np.array(done)
        if not done:
            continue
        r = results[i]
        payload[f"p{i}_prompt_ids"] = np.array(r["prompt_ids"], dtype=np.int64)
        payload[f"p{i}_gen_ids"] = np.array(r["gen_ids"], dtype=np.int64)
        payload[f"p{i}_top1_ids"] = np.array(r["top1_ids"], dtype=np.int64)
        payload[f"p{i}_top1_logits"] = np.array(r["top1_logits"], dtype=np.float32)
        payload[f"p{i}_top2_ids"] = np.array(r["top2_ids"], dtype=np.int64)
        payload[f"p{i}_top2_logits"] = np.array(r["top2_logits"], dtype=np.float32)
        payload[f"p{i}_required_min_margin"] = np.array(r["required_min_margin"], dtype=np.float32)
        payload[f"p{i}_stable"] = np.array(r["stable"])
        payload[f"p{i}_wall_s"] = np.array(r["wall_s"], dtype=np.float64)
    _atomic_write_npz(HF_OUT_NPZ, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N prompts from the battery")
    ap.add_argument("--prompt-key", default=None,
                    help="process only the prompt with this stable key")
    ap.add_argument("--force", action="store_true",
                    help="recompute selected prompts even if partial results exist")
    args = ap.parse_args()

    if HF_SENTINEL.exists() and not args.force:
        print(f"sentinel exists: {HF_SENTINEL} (delete to recapture)")
        sys.exit(0)
    if HF_SENTINEL.exists() and args.force:
        print(f"  ignoring existing sentinel due to --force: {HF_SENTINEL}")
    if not MODEL_DIR.exists():
        print(f"FATAL missing: {MODEL_DIR}", file=sys.stderr)
        sys.exit(2)
    if not REAP_MASK.exists():
        print(f"FATAL missing: {REAP_MASK}", file=sys.stderr)
        sys.exit(2)

    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1e9
        print(f"  free RAM: {free_gb:.1f} GB (floor {FREE_RAM_FLOOR_GB})")
        if free_gb < FREE_RAM_FLOOR_GB:
            print(f"FATAL: need >= {FREE_RAM_FLOOR_GB} GB free", file=sys.stderr)
            sys.exit(2)
    except ImportError:
        print("  (psutil not installed; skipping RAM check)")

    keep_idx = np.load(REAP_MASK, allow_pickle=False)["keep_idx"]
    assert keep_idx.shape == (30, 64)

    nth = os.cpu_count() or 1
    torch.set_num_threads(nth)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    print(f"  torch threads = {nth}  transformers={transformers.__version__}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    selected = []
    for i, triple in enumerate(BATTERY):
        key = triple[0]
        if args.prompt_key is not None and key != args.prompt_key:
            continue
        selected.append((i, triple))
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        print("FATAL no prompts selected", file=sys.stderr)
        sys.exit(2)
    print("  selected prompts:", [key for _, (key, _, _) in selected])

    results = {} if args.force else _load_partial_results()

    print("  loading model fp16 CPU (26 GB)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()
    print(f"  load wall: {time.perf_counter()-t0:.1f}s")

    text_model = getattr(model, "model", None) or model
    decoder_layers = None
    for cand in ("layers", "decoder_layers"):
        if hasattr(text_model, cand):
            decoder_layers = getattr(text_model, cand)
            break
        if hasattr(text_model, "language_model"):
            lm = text_model.language_model
            if hasattr(lm, cand):
                decoder_layers = getattr(lm, cand)
                break
        if hasattr(text_model, "text_model"):
            tm = text_model.text_model
            if hasattr(tm, cand):
                decoder_layers = getattr(tm, cand)
                break
    assert decoder_layers is not None and len(decoder_layers) == 30

    layer_keep_masks = []
    for li in range(30):
        m = torch.zeros(128, dtype=torch.bool)
        m[torch.from_numpy(keep_idx[li])] = True
        assert int(m.sum()) == 64
        layer_keep_masks.append(m)

    def make_patched(orig_router, keep_mask):
        not_kept = ~keep_mask
        def patched(hidden_states):
            x = orig_router.norm(hidden_states)
            x = x * orig_router.scale * orig_router.scalar_root_size
            scores = orig_router.proj(x)
            mv = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(not_kept.to(scores.device), mv)
            probs = torch.nn.functional.softmax(scores, dim=-1)
            tw, ti = torch.topk(probs, k=orig_router.config.top_k_experts, dim=-1)
            tw = tw / tw.sum(dim=-1, keepdim=True)
            tw = tw * orig_router.per_expert_scale[ti]
            return probs, tw, ti
        return patched

    for li, dec in enumerate(decoder_layers):
        dec.router.forward = make_patched(dec.router, layer_keep_masks[li])
    print("  patched 30 routers (REAP mask before softmax)")

    for pi, (key, prompt, required_prefix) in selected:
        if pi in results and not args.force:
            print(f"\n  === prompt {pi}: {prompt!r} ===")
            print("    already captured; skipping")
            continue
        print(f"\n  === prompt {pi}: {prompt!r} ===")
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        prompt_ids = input_ids[0].tolist()
        print(f"    prompt ids ({len(prompt_ids)}): {prompt_ids}")

        t0 = time.perf_counter()
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=N_NEW,
            do_sample=False,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
        gen_dt = time.perf_counter() - t0
        full_ids = out.sequences[0].tolist()
        gen_ids = full_ids[len(prompt_ids):]
        top1_ids = []
        top1_logits = []
        top2_ids = []
        top2_logits = []
        for step_logits in out.scores:
            sl = step_logits[0]
            vals, ids2 = torch.topk(sl, k=2)
            top1_ids.append(int(ids2[0].item()))
            top1_logits.append(float(vals[0].item()))
            top2_ids.append(int(ids2[1].item()))
            top2_logits.append(float(vals[1].item()))
        completion = tok.decode(gen_ids, skip_special_tokens=False)
        margins = [a - b for a, b in zip(top1_logits, top2_logits)]
        req_min_margin = min(margins[:required_prefix])
        stable = req_min_margin >= MIN_REQUIRED_MARGIN
        print(f"    gen ids: {gen_ids}")
        print(f"    completion: {completion!r}")
        print(f"    required prefix tokens: {required_prefix}")
        print(f"    required min margin: {req_min_margin:.3f} "
              f"(threshold {MIN_REQUIRED_MARGIN:.3f})  {'STABLE' if stable else 'UNSTABLE'}")
        print(f"    wall: {gen_dt:.1f}s ({gen_dt/N_NEW:.1f} s/tok)")
        results[pi] = {
            "prompt_ids": np.array(prompt_ids, dtype=np.int64),
            "gen_ids": np.array(gen_ids, dtype=np.int64),
            "top1_ids": np.array(top1_ids, dtype=np.int64),
            "top1_logits": np.array(top1_logits, dtype=np.float32),
            "top2_ids": np.array(top2_ids, dtype=np.int64),
            "top2_logits": np.array(top2_logits, dtype=np.float32),
            "required_min_margin": req_min_margin,
            "stable": stable,
            "wall_s": gen_dt,
        }
        _write_partial(results, transformers.__version__)
        print(f"    checkpointed → {HF_OUT_NPZ}")

    if args.prompt_key is not None or args.limit is not None:
        print("\n  partial capture complete; sentinel not written in smoke/select mode")
        return

    unstable = []
    incomplete = []
    for i, (key, _, _) in enumerate(BATTERY):
        if i not in results:
            incomplete.append(key)
            continue
        if not results[i]["stable"]:
            unstable.append(key)

    if incomplete or unstable:
        if HF_SENTINEL.exists():
            HF_SENTINEL.unlink()
        print(f"\n  wrote {HF_OUT_NPZ}")
        if incomplete:
            print(f"  INCOMPLETE prompts: {incomplete}")
        if unstable:
            print(f"  UNSTABLE prompts (required-prefix margin < {MIN_REQUIRED_MARGIN}): {unstable}")
        print("  sentinel not written")
        sys.exit(1)

    HF_SENTINEL.write_text(
        "n_prompts={} n_new={} transformers={} min_required_margin={}\n".format(
            len(BATTERY), N_NEW, transformers.__version__, MIN_REQUIRED_MARGIN))
    print(f"\n  wrote {HF_OUT_NPZ}")
    print(f"  sentinel: {HF_SENTINEL}")


if __name__ == "__main__":
    main()