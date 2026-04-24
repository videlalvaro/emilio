"""gemma_ref.py — produce golden logits from the reference HF Gemma-4 model.

We need a fixed reference output that any future ANE/optimized
implementation must match (logit cosine similarity > 0.999, top-k overlap,
sampled output matches).

Saves to python/moe/out/gemma_golden.npz:
  prompt_ids        : int32 (T,)
  logits_full       : fp32 (T, vocab)         logits at every position
  next_token_ids    : int32 (N_GEN,)          greedy continuation
  next_token_logits : fp32 (N_GEN, vocab)     logits used for each step

Run:  .venv313/bin/python python/moe/gemma_ref.py
"""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT_PATH  = Path("python/moe/out/gemma_golden.npz")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PROMPT = "What is the capital of France?"
N_GEN  = 16


def main():
    print("loading tokenizer + model (bf16, CPU)...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  model class: {type(model).__name__}")

    ids = tok(PROMPT, return_tensors="pt").input_ids
    print(f"prompt: {PROMPT!r}  → {ids.shape[1]} tokens: {ids[0].tolist()}")

    with torch.no_grad():
        # Full prompt forward (logits at every position)
        out = model(ids, use_cache=True)
        logits_full = out.logits[0].float().cpu().numpy()  # (T, vocab)
        past = out.past_key_values
        print(f"prompt logits: shape={logits_full.shape}, "
              f"argmax(last)={int(logits_full[-1].argmax())} "
              f"-> {tok.decode([int(logits_full[-1].argmax())])!r}")

        # Greedy decode N_GEN tokens, recording logits for each step
        gen_ids, gen_logits = [], []
        cur = torch.tensor([[int(logits_full[-1].argmax())]], dtype=torch.long)
        for i in range(N_GEN):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            l = out.logits[0, -1].float().cpu().numpy()
            gen_logits.append(l)
            gen_ids.append(int(cur.item()))
            nxt = int(l.argmax())
            cur = torch.tensor([[nxt]], dtype=torch.long)

    gen_ids = np.array(gen_ids, dtype=np.int32)
    gen_logits = np.stack(gen_logits, axis=0).astype(np.float32)
    print(f"\ngreedy continuation: {tok.decode(gen_ids.tolist())!r}")
    print(f"token ids: {gen_ids.tolist()}")

    np.savez(
        OUT_PATH,
        prompt=PROMPT,
        prompt_ids=ids[0].numpy().astype(np.int32),
        logits_full=logits_full,
        next_token_ids=gen_ids,
        next_token_logits=gen_logits,
    )
    print(f"\nsaved → {OUT_PATH}")
    print(f"  size = {OUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
