"""Screen candidate factual prompts by expected answer-prefix stability.

This utility is intentionally separate from the frozen battery capture. It is
used to select prompts whose expected answer token(s) have a healthy HF margin
before those prompts are admitted into the closed-form regression battery.

Run only with gatekeeper approval because it loads the full HF Gemma-4 model.
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

from hf_full_model_safety import (
    DEFAULT_DISK_FREE_MIN_GIB,
    DEFAULT_MAX_CPU_MEMORY_GIB,
    DEFAULT_MAX_DISK_MEMORY_GIB,
    prepare_model_load_kwargs,
    require_disk_free,
    validate_full_model_load_policy,
)

MODEL_DIR = Path("models/gemma-4-26b-a4b")
REAP_MASK = Path("python/moe/out/gemma_reap_mask.npz")
OUT_DIR = Path("python/moe/out")
MIN_MARGIN = 1.0
N_NEW = 4

# (key, prompt, expected_prefix_text)
CANDIDATES = (
    ("japan_capital", "The capital of Japan is", " Tokyo"),
    ("italy_capital", "The capital of Italy is", " Rome"),
    ("spain_capital", "The capital of Spain is", " Madrid"),
    ("largest_planet", "The largest planet in the Solar System is", " Jupiter"),
    ("gold_symbol", "The chemical symbol for gold is", " Au"),
    ("silver_symbol", "The chemical symbol for silver is", " Ag"),
    ("iron_symbol", "The chemical symbol for iron is", " Fe"),
    ("copper_symbol", "The chemical symbol for copper is", " Cu"),
    ("helium_symbol", "The chemical symbol for helium is", " He"),
    ("two_plus_two", "2 + 2 =", " 4"),
    ("sqrt_nine", "The square root of 9 is", " 3"),
    ("roman_ten", "The Roman numeral for ten is", " X"),
    ("alphabet_first", "The first letter of the English alphabet is", " A"),
    ("largest_ocean", "The largest ocean on Earth is the", " Pacific"),
)


def _resolve_decoder_layers(model):
    text_model = getattr(model, "model", None) or model
    for cand in ("layers", "decoder_layers"):
        if hasattr(text_model, cand):
            layers = getattr(text_model, cand)
            if len(layers) == 30:
                return layers
        if hasattr(text_model, "language_model"):
            lm = text_model.language_model
            if hasattr(lm, cand):
                layers = getattr(lm, cand)
                if len(layers) == 30:
                    return layers
        if hasattr(text_model, "text_model"):
            tm = text_model.text_model
            if hasattr(tm, cand):
                layers = getattr(tm, cand)
                if len(layers) == 30:
                    return layers
    raise RuntimeError("could not resolve 30 decoder layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=None,
                    help="screen only these stable candidate keys")
    ap.add_argument("--limit", type=int, default=None,
                    help="screen only the first N candidates after key filtering")
    ap.add_argument("--start-at-key", default=None,
                    help="resume screening from this candidate key onward")
    ap.add_argument("--stop-after-load", action="store_true",
                    help="load and patch the model, then exit without screening prompts")
    ap.add_argument("--offload", dest="offload", action="store_true", default=True,
                    help="enable disk offload for the full HF model load (default: on)")
    ap.add_argument("--no-offload", dest="offload", action="store_false",
                    help="disable disk offload; requires explicit unsafe override")
    ap.add_argument("--offload-folder", type=Path,
                    default=OUT_DIR / ".offload_closedform_screen")
    ap.add_argument("--max-cpu-memory-gib", type=int, default=DEFAULT_MAX_CPU_MEMORY_GIB)
    ap.add_argument("--max-disk-memory-gib", type=int, default=DEFAULT_MAX_DISK_MEMORY_GIB)
    ap.add_argument("--disk-free-min-gib", type=int, default=DEFAULT_DISK_FREE_MIN_GIB)
    ap.add_argument("--allow-unsafe-cpu-memory", action="store_true")
    ap.add_argument("--allow-no-disk-offload", action="store_true")
    args = ap.parse_args()

    validate_full_model_load_policy(
        "gemma_hf_closedform_screen",
        offload_enabled=args.offload,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        max_disk_memory_gib=args.max_disk_memory_gib,
        allow_unsafe_cpu_memory=args.allow_unsafe_cpu_memory,
        allow_no_disk_offload=args.allow_no_disk_offload,
    )

    if not MODEL_DIR.exists():
        print(f"FATAL missing: {MODEL_DIR}", file=sys.stderr)
        sys.exit(2)
    if not REAP_MASK.exists():
        print(f"FATAL missing: {REAP_MASK}", file=sys.stderr)
        sys.exit(2)

    keep_idx = np.load(REAP_MASK, allow_pickle=False)["keep_idx"]
    assert keep_idx.shape == (30, 64)

    nth = os.cpu_count() or 1
    torch.set_num_threads(nth)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    print(f"  torch threads = {nth}  transformers={transformers.__version__}")
    disk_paths = [MODEL_DIR, OUT_DIR]
    if args.offload:
        disk_paths.append(args.offload_folder)
    require_disk_free(disk_paths, args.disk_free_min_gib)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print("  loading model fp16 with guarded offload policy...")
    print(
        f"  memory policy: offload={args.offload} cpu={args.max_cpu_memory_gib}GiB "
        f"disk={args.max_disk_memory_gib}GiB"
    )
    t0 = time.perf_counter()
    model_kwargs, actual_offload_folder = prepare_model_load_kwargs(
        torch_dtype=torch.float16,
        offload_enabled=args.offload,
        offload_folder=args.offload_folder,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        max_disk_memory_gib=args.max_disk_memory_gib,
        local_files_only=True,
    )
    if actual_offload_folder is not None:
        print(f"  offload folder: {actual_offload_folder}")
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), **model_kwargs).eval()
    print(f"  load wall: {time.perf_counter()-t0:.1f}s")

    decoder_layers = _resolve_decoder_layers(model)
    layer_keep_masks = []
    for li in range(30):
        m = torch.zeros(128, dtype=torch.bool)
        m[torch.from_numpy(keep_idx[li])] = True
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

    selected = list(CANDIDATES)
    if args.keys:
        wanted = set(args.keys)
        selected = [triple for triple in selected if triple[0] in wanted]
    if args.start_at_key is not None:
        seen = False
        trimmed = []
        for triple in selected:
            if triple[0] == args.start_at_key:
                seen = True
            if seen:
                trimmed.append(triple)
        selected = trimmed
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        print("FATAL no candidates selected", file=sys.stderr)
        sys.exit(2)
    print("  selected candidates:", [key for key, _, _ in selected])
    if args.stop_after_load:
        print("  stop-after-load requested; exiting before screening")
        return

    print(f"\n=== Screening candidates (min margin {MIN_MARGIN:.3f}) ===")
    stable = []
    for key, prompt, expected_text in selected:
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        expected_ids = tok(expected_text, add_special_tokens=False)["input_ids"]
        print(f"\n--- {key}: {prompt!r}")
        print(f"  expected prefix text: {expected_text!r} ids={expected_ids}")

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
        gen_ids = full_ids[len(input_ids[0]):]
        gen_text = tok.decode(gen_ids, skip_special_tokens=False)
        print(f"  gen ids: {gen_ids}")
        print(f"  completion: {gen_text!r}")
        ok = True
        min_margin = None
        for step_i, exp_id in enumerate(expected_ids):
            if step_i >= len(out.scores):
                ok = False
                print(f"  step {step_i}: missing score row")
                break
            sl = out.scores[step_i][0]
            vals, ids2 = torch.topk(sl, k=2)
            top1_id = int(ids2[0].item())
            top2_id = int(ids2[1].item())
            top1_logit = float(vals[0].item())
            top2_logit = float(vals[1].item())
            exp_logit = float(sl[exp_id].item())
            margin = exp_logit - top2_logit if top1_id == exp_id else exp_logit - top1_logit
            if min_margin is None or margin < min_margin:
                min_margin = margin
            step_ok = (top1_id == exp_id) and (top1_logit - top2_logit >= MIN_MARGIN)
            ok = ok and step_ok
            print(
                f"  step {step_i}: expected={exp_id} {tok.decode([exp_id], skip_special_tokens=False)!r:14s} "
                f"top1={top1_id} {tok.decode([top1_id], skip_special_tokens=False)!r:14s} {top1_logit:7.3f}  "
                f"top2={top2_id} {tok.decode([top2_id], skip_special_tokens=False)!r:14s} {top2_logit:7.3f}  "
                f"margin={top1_logit-top2_logit if top1_id == exp_id else margin:7.3f}  "
                f"{'OK' if step_ok else 'NO'}"
            )
        print(f"  wall: {gen_dt:.1f}s ({gen_dt/N_NEW:.1f} s/tok)")
        print(f"  verdict: {'STABLE' if ok else 'UNSTABLE'}")
        if ok:
            stable.append((key, expected_text, min_margin))

    print("\n=== Stable candidates ===")
    if not stable:
        print("  none")
    for key, expected_text, min_margin in stable:
        print(f"  {key}: expected={expected_text!r} min_margin={min_margin:.3f}")


if __name__ == "__main__":
    main()