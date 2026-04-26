"""diag_p2_hf_layers.py — capture HF hidden states at every decoder layer.

Same setup as diag_p2_hf_top50.py but with output_hidden_states=True so we can
diff against CML at shard boundaries (after layers 15, 22, 30) and per-layer.

Output: python/moe/out/diag_p2_hf_layers.npz with
  hidden_norms[T, L+1]                     ‖h_pos‖₂ at each layer (incl. embed)
  hidden_last_pos_per_layer[L+1, D]        h at last prompt pos, every layer
  input_ids[T]
"""
from __future__ import annotations
import argparse
import os, sys, time
from pathlib import Path
assert "venv313" in sys.executable
import numpy as np, torch

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
OUT = Path("python/moe/out/diag_p2_hf_layers.npz")
REAP = Path("python/moe/out/gemma_reap_mask.npz")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offload", dest="offload", action="store_true", default=True)
    ap.add_argument("--no-offload", dest="offload", action="store_false")
    ap.add_argument("--offload-folder", type=Path,
                    default=OUT.parent / ".offload_diag_p2_layers")
    ap.add_argument("--max-cpu-memory-gib", type=int, default=DEFAULT_MAX_CPU_MEMORY_GIB)
    ap.add_argument("--max-disk-memory-gib", type=int, default=DEFAULT_MAX_DISK_MEMORY_GIB)
    ap.add_argument("--disk-free-min-gib", type=int, default=DEFAULT_DISK_FREE_MIN_GIB)
    ap.add_argument("--allow-unsafe-cpu-memory", action="store_true")
    ap.add_argument("--allow-no-disk-offload", action="store_true")
    args = ap.parse_args()

    validate_full_model_load_policy(
        "diag_p2_hf_layers",
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
    def make_patched(orig, keep):
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
        dec.router.forward = make_patched(dec.router, masks[li])

    enc = tok(PROMPT, return_tensors="pt")
    ids = enc["input_ids"]
    T = int(ids.shape[1])
    print(f"prompt ids ({T}): {ids[0].tolist()}")
    print("forward (with hidden states)..."); t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    print(f"  fwd {time.perf_counter()-t0:.1f}s")
    hs = out.hidden_states  # tuple len 31: embed + 30 layer outputs
    L1 = len(hs); D = hs[0].shape[-1]
    print(f"  hidden_states len={L1}  D={D}")

    norms = np.zeros((T, L1), dtype=np.float32)
    last_per_layer = np.zeros((L1, D), dtype=np.float32)
    for li, h in enumerate(hs):
        h_np = h[0].float().cpu().numpy()  # (T, D)
        for p in range(T):
            norms[p, li] = float(np.linalg.norm(h_np[p]))
        last_per_layer[li] = h_np[-1]
    print("‖h_last‖ per layer (boundary): ",
          [f"L{li}:{norms[-1, li]:.1f}" for li in (0, 1, 7, 15, 22, 29, 30)])
    np.savez(str(OUT),
             prompt=np.array(PROMPT),
             input_ids=ids[0].numpy(),
             hidden_norms=norms,
             hidden_last_pos_per_layer=last_per_layer)
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
