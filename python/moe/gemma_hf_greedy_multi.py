"""gemma_hf_greedy_multi.py — multi-prompt REAP-aware HF greedy generation.

For each prompt, greedy-generates N tokens with the HF Gemma-4-26B-A4B model
(fp16, CPU) under REAP routing (non-keep_idx experts masked to -inf before
softmax, identical to gemma_hf_logits_capture_reap.py). Saves per-prompt
token IDs + per-step top-1 logits for direct apples-to-apples comparison
against the CoreML INT4 chain (gemma_t414_generate_multi.py).

Writes:
  python/moe/out/gemma_hf_greedy_multi.npz
  python/moe/out/.gemma_hf_greedy_multi_PASS

Usage (must be .venv313):
  .venv313/bin/python python/moe/gemma_hf_greedy_multi.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

assert "venv313" in sys.executable, (
    f"Wrong interpreter: {sys.executable!r}; need .venv313/bin/python")

import numpy as np
import torch
import transformers

PROMPTS = [
    "The capital of France is",                     # T4.1.4 regression
    "2 + 2 =",                                       # easy arithmetic
    "The first president of the United States was", # factual recall
    "The Riemann hypothesis states that",            # hard continuation
]
N_NEW = 8

MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT_DIR = Path("python/moe/out")
OUT_NPZ = OUT_DIR / "gemma_hf_greedy_multi.npz"
SENTINEL = OUT_DIR / ".gemma_hf_greedy_multi_PASS"
REAP_MASK = OUT_DIR / "gemma_reap_mask.npz"

FREE_RAM_FLOOR_GB = 12.0   # gatekeeper-required


def _atomic_write_npz(target: Path, **arrays):
    tmp_base = target.with_suffix(target.suffix + ".tmp")
    tmp_written = Path(str(tmp_base) + ".npz")
    np.savez(str(tmp_base), **arrays)
    assert tmp_written.exists()
    with open(tmp_written, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_written, target)


def main():
    if SENTINEL.exists():
        print(f"sentinel exists: {SENTINEL} (delete to recapture)"); sys.exit(0)
    if not MODEL_DIR.exists():
        print(f"FATAL missing: {MODEL_DIR}", file=sys.stderr); sys.exit(2)
    if not REAP_MASK.exists():
        print(f"FATAL missing: {REAP_MASK}", file=sys.stderr); sys.exit(2)

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
    torch.set_num_threads(nth); torch.set_grad_enabled(False); torch.manual_seed(0)
    print(f"  torch threads = {nth}  transformers={transformers.__version__}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"  loading model fp16 CPU (26 GB)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()
    print(f"  load wall: {time.perf_counter()-t0:.1f}s")

    # Robust resolver (copied from gemma_hf_logits_capture_reap.py).
    text_model = getattr(model, "model", None) or model
    decoder_layers = None
    for cand in ("layers", "decoder_layers"):
        if hasattr(text_model, cand):
            decoder_layers = getattr(text_model, cand); break
        if hasattr(text_model, "language_model"):
            lm = text_model.language_model
            if hasattr(lm, cand):
                decoder_layers = getattr(lm, cand); break
        if hasattr(text_model, "text_model"):
            tm = text_model.text_model
            if hasattr(tm, cand):
                decoder_layers = getattr(tm, cand); break
    assert decoder_layers is not None and len(decoder_layers) == 30, \
        f"could not resolve 30 decoder layers; got {decoder_layers!r}"

    layer_keep_masks = []
    for li in range(30):
        m = torch.zeros(128, dtype=torch.bool)
        m[torch.from_numpy(keep_idx[li])] = True
        assert int(m.sum()) == 64
        layer_keep_masks.append(m)

    def make_patched(orig_router, keep_mask, layer_idx):
        not_kept = ~keep_mask
        first = {"flag": False}
        def patched(hidden_states):
            x = orig_router.norm(hidden_states)
            x = x * orig_router.scale * orig_router.scalar_root_size
            scores = orig_router.proj(x)
            mv = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(not_kept.to(scores.device), mv)
            if not first["flag"]:
                assert (scores[..., not_kept] <= mv + 1).all(), \
                    f"L{layer_idx}: mask not applied"
                first["flag"] = True
            probs = torch.nn.functional.softmax(scores, dim=-1)
            tw, ti = torch.topk(probs, k=orig_router.config.top_k_experts, dim=-1)
            tw = tw / tw.sum(dim=-1, keepdim=True)
            tw = tw * orig_router.per_expert_scale[ti]
            return probs, tw, ti
        return patched

    for li, dec in enumerate(decoder_layers):
        assert hasattr(dec, "router"), f"L{li} no .router"
        dec.router.forward = make_patched(dec.router, layer_keep_masks[li], li)
    print(f"  patched 30 routers (logits → mask → softmax → top-k; HF convention)")

    # --- per-prompt greedy generation -------------------------------
    results: list[dict] = []
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  === prompt {pi}: {prompt!r} ===")
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        prompt_ids = input_ids[0].tolist()
        print(f"    prompt ids ({len(prompt_ids)}): {prompt_ids}")

        # Single-shot with HF cache.
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
        # out.scores is tuple of (1, vocab) per generated step
        top1_logits = []
        for step_logits in out.scores:
            sl = step_logits[0]                      # (vocab,)
            tid = int(torch.argmax(sl).item())
            top1_logits.append((tid, float(sl[tid].item())))
        completion = tok.decode(gen_ids, skip_special_tokens=False)
        print(f"    gen ids: {gen_ids}")
        print(f"    completion: {completion!r}")
        print(f"    wall: {gen_dt:.1f}s ({gen_dt/N_NEW:.1f} s/tok)")
        results.append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "gen_ids": gen_ids,
            "top1_logits": top1_logits,
            "wall_s": gen_dt,
        })

    # --- save -------------------------------------------------------
    payload = {
        "n_prompts": np.array(len(PROMPTS), dtype=np.int64),
        "n_new": np.array(N_NEW, dtype=np.int64),
        "transformers_version": np.array(transformers.__version__),
    }
    for i, r in enumerate(results):
        payload[f"p{i}_prompt"] = np.array(r["prompt"])
        payload[f"p{i}_prompt_ids"] = np.array(r["prompt_ids"], dtype=np.int64)
        payload[f"p{i}_gen_ids"] = np.array(r["gen_ids"], dtype=np.int64)
        payload[f"p{i}_top1_logits"] = np.array(
            [v for _, v in r["top1_logits"]], dtype=np.float32)
        payload[f"p{i}_wall_s"] = np.array(r["wall_s"], dtype=np.float64)

    _atomic_write_npz(OUT_NPZ, **payload)
    SENTINEL.write_text(
        "n_prompts={} n_new={} transformers={}\n".format(
            len(PROMPTS), N_NEW, transformers.__version__))
    print(f"\n  wrote {OUT_NPZ}")
    print(f"  sentinel: {SENTINEL}")


if __name__ == "__main__":
    main()
