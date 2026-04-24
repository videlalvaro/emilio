"""diff HF vs CoreML multi-prompt greedy outputs."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import tokenizers

tok = tokenizers.Tokenizer.from_file('models/gemma-4-26b-a4b/tokenizer.json')
hf = np.load('python/moe/out/gemma_hf_greedy_multi.npz', allow_pickle=False)
cm = np.load('python/moe/out/gemma_t414_generate_multi.npz', allow_pickle=False)
N = int(hf['n_prompts']); K = int(hf['n_new'])

total_match = 0; total = 0; prefix_lens = []
for i in range(N):
    prompt = str(hf[f'p{i}_prompt'])
    h_ids = list(map(int, hf[f'p{i}_gen_ids']))
    c_ids = list(map(int, cm[f'p{i}_gen_ids']))
    h_logits = hf[f'p{i}_top1_logits']
    c_logits = cm[f'p{i}_top1_logits']
    matches = [h_ids[k] == c_ids[k] for k in range(K)]
    n_match = sum(matches)
    plen = 0
    for m in matches:
        if m: plen += 1
        else: break
    prefix_lens.append(plen)
    total_match += n_match; total += K
    print(f"--- prompt {i}: {prompt!r} ---")
    print(f"  HF  text: {tok.decode(h_ids, skip_special_tokens=False)!r}")
    print(f"  CML text: {tok.decode(c_ids, skip_special_tokens=False)!r}")
    print(f"  match {n_match}/{K}, common prefix {plen}, per-step:")
    for k in range(K):
        h_id = h_ids[k]; c_id = c_ids[k]
        h_str = tok.decode([h_id], skip_special_tokens=False)
        c_str = tok.decode([c_id], skip_special_tokens=False)
        flag = "OK" if h_id == c_id else "XX"
        print(f"    [{k}] {flag}  HF={h_id:7d} ({h_str!r:18s} L={h_logits[k]:7.3f})"
              f"   CML={c_id:7d} ({c_str!r:18s} L={c_logits[k]:7.3f})")
print(f"\n=== TOTAL token agreement: {total_match}/{total} "
      f"= {100*total_match/total:.1f}%   prefix lens {prefix_lens} ===")
