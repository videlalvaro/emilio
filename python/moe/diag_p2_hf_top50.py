"""diag_p2_hf_top50.py — capture HF top-50 logits at prompt-2 divergence step.

Loads gemma-4-26b-a4b fp16 with REAP router patch, runs prompt
"The first president of the United States was" at T=9, dumps top-50 token IDs
+ logits at the LAST prompt position (the position whose argmax becomes the
first generated token; HF said " George", CML said " a").

Output: python/moe/out/diag_p2_hf_top50.npz
Usage:  .venv313/bin/python python/moe/diag_p2_hf_top50.py
"""
from __future__ import annotations
import argparse
import os, sys, time
from pathlib import Path
assert "venv313" in sys.executable
import numpy as np, torch, transformers

from hf_full_model_safety import (
    DEFAULT_DISK_FREE_MIN_GIB,
    DEFAULT_MAX_CPU_MEMORY_GIB,
    DEFAULT_MAX_DISK_MEMORY_GIB,
    prepare_model_load_kwargs,
    require_disk_free,
    validate_full_model_load_policy,
)

PROMPT = "The first president of the United States was"
MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT = Path("python/moe/out/diag_p2_hf_top50.npz")
REAP = Path("python/moe/out/gemma_reap_mask.npz")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offload", dest="offload", action="store_true", default=True)
    ap.add_argument("--no-offload", dest="offload", action="store_false")
    ap.add_argument("--offload-folder", type=Path,
                    default=OUT.parent / ".offload_diag_p2_top50")
    ap.add_argument("--max-cpu-memory-gib", type=int, default=DEFAULT_MAX_CPU_MEMORY_GIB)
    ap.add_argument("--max-disk-memory-gib", type=int, default=DEFAULT_MAX_DISK_MEMORY_GIB)
    ap.add_argument("--disk-free-min-gib", type=int, default=DEFAULT_DISK_FREE_MIN_GIB)
    ap.add_argument("--allow-unsafe-cpu-memory", action="store_true")
    ap.add_argument("--allow-no-disk-offload", action="store_true")
    args = ap.parse_args()

    validate_full_model_load_policy(
        "diag_p2_hf_top50",
        offload_enabled=args.offload,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        max_disk_memory_gib=args.max_disk_memory_gib,
        allow_unsafe_cpu_memory=args.allow_unsafe_cpu_memory,
        allow_no_disk_offload=args.allow_no_disk_offload,
    )

    keep_idx = np.load(REAP, allow_pickle=False)["keep_idx"]
    torch.set_num_threads(os.cpu_count() or 1); torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    disk_paths = [MODEL_DIR, OUT.parent]
    if args.offload:
        disk_paths.append(args.offload_folder)
    require_disk_free(disk_paths, args.disk_free_min_gib)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    print("loading model..."); t0 = time.perf_counter()
    print(
        f"  memory policy: offload={args.offload} cpu={args.max_cpu_memory_gib}GiB "
        f"disk={args.max_disk_memory_gib}GiB"
    )
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
    print(f"  load {time.perf_counter()-t0:.1f}s")

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
    assert decoder_layers is not None and len(decoder_layers) == 30
    masks = []
    for li in range(30):
        m = torch.zeros(128, dtype=torch.bool)
        m[torch.from_numpy(keep_idx[li])] = True
        masks.append(m)
    def make_patched(orig, keep, li):
        not_kept = ~keep
        def patched(h):
            x = orig.norm(h); x = x * orig.scale * orig.scalar_root_size
            s = orig.proj(x); mv = torch.finfo(s.dtype).min
            s = s.masked_fill(not_kept, mv)
            p = torch.nn.functional.softmax(s, dim=-1)
            tw, ti = torch.topk(p, k=orig.config.top_k_experts, dim=-1)
            tw = tw / tw.sum(-1, keepdim=True)
            tw = tw * orig.per_expert_scale[ti]
            return p, tw, ti
        return patched
    for li, dec in enumerate(decoder_layers):
        dec.router.forward = make_patched(dec.router, masks[li], li)

    enc = tok(PROMPT, return_tensors="pt")
    ids = enc["input_ids"]
    print(f"prompt ids ({ids.shape[1]}): {ids[0].tolist()}")
    print("forward..."); t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=False)
    print(f"  fwd {time.perf_counter()-t0:.1f}s")
    last = out.logits[0, -1].float().cpu().numpy()  # vocab,
    top50 = np.argsort(-last)[:50]
    print("HF top-50 at last prompt position:")
    for r, tid in enumerate(top50):
        print(f"  [{r:2d}] id={int(tid):7d} L={last[tid]:7.3f}  "
              f"{tok.decode([int(tid)], skip_special_tokens=False)!r}")
    np.savez(str(OUT),
             prompt=np.array(PROMPT),
             input_ids=ids[0].numpy(),
             last_pos_logits=last.astype(np.float32),
             top50_ids=top50.astype(np.int64),
             top50_logits=last[top50].astype(np.float32))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
