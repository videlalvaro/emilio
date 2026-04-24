"""gemma_t414_generate_multi.py — CoreML INT4 chain greedy on multiple prompts.

Counterpart to gemma_hf_greedy_multi.py. Same prompts, same N_NEW, same
greedy decoding. Saves per-prompt token IDs + per-step top-1 logits.

Run:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_t414_generate_multi.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct
import tokenizers

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import D_MODEL, SLD_D_HEAD, GLB_ROT_DIM  # noqa: E402
from gemma_t414_generate import (  # noqa: E402
    SHARDS, SHARD_PATHS, LOGIT_HEAD_NPZ, TOKENIZER_JSON, MAX_CTX,
    _rope_for_pos, _attn_mask_for_pos, _write_mask_for_pos,
    _final_norm_softcap_logits,
)

PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "The first president of the United States was",
    "The Riemann hypothesis states that",
]
N_NEW = 8

OUT_DIR = Path("python/moe/out")
OUT_NPZ = OUT_DIR / "gemma_t414_generate_multi.npz"


def _atomic_write_npz(target: Path, **arrays):
    tmp_base = target.with_suffix(target.suffix + ".tmp")
    tmp_written = Path(str(tmp_base) + ".npz")
    np.savez(str(tmp_base), **arrays)
    assert tmp_written.exists()
    with open(tmp_written, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_written, target)


def main():
    for p in [LOGIT_HEAD_NPZ, TOKENIZER_JSON] + SHARD_PATHS:
        if not p.exists():
            print(f"FATAL missing: {p}", file=sys.stderr); sys.exit(2)

    head = np.load(LOGIT_HEAD_NPZ, allow_pickle=False)
    embed = head["embed_weight"]; gamma = head["final_norm_gamma"]
    eps = float(head["rms_norm_eps"]); softcap = float(head["softcap"])
    embed_scale = float(np.sqrt(D_MODEL))
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))

    print("=== T4.1.4 multi-prompt greedy (CoreML INT4 chain) ===")
    print(f"  loading {len(SHARD_PATHS)} shards...")
    shard_models = []
    for (a, b), pth in zip(SHARDS, SHARD_PATHS):
        m = ct.models.CompiledMLModel(str(pth),
                                       compute_units=ct.ComputeUnit.CPU_AND_NE)
        shard_models.append(m)
        print(f"    shard[{a},{b}] loaded")

    results: list[dict] = []
    for pi, prompt in enumerate(PROMPTS):
        print(f"\n  === prompt {pi}: {prompt!r} ===")
        # Fresh state per prompt (no cross-prompt KV bleed).
        shard_states = [m.make_state() for m in shard_models]

        prompt_ids = tok.encode(prompt).ids
        # Gemma uses <bos>; tokenizers default may not prepend. Match HF.
        if not prompt_ids or prompt_ids[0] != 2:
            prompt_ids = [2] + prompt_ids
        print(f"    prompt ids ({len(prompt_ids)}): {prompt_ids}")

        def step(token_id: int, pos: int):
            x = (embed[token_id].astype(np.float32) * embed_scale).astype(np.float16)
            x = x.reshape(1, 1, D_MODEL)
            cos_s, sin_s = _rope_for_pos(10_000.0, SLD_D_HEAD, pos)
            cos_g, sin_g = _rope_for_pos(1_000_000.0, GLB_ROT_DIM, pos)
            amask = _attn_mask_for_pos(pos); wmask = _write_mask_for_pos(pos)
            cur = x
            for m, st in zip(shard_models, shard_states):
                out = m.predict(dict(x=cur, cos_s=cos_s, sin_s=sin_s,
                                     cos_g=cos_g, sin_g=sin_g,
                                     attn_mask=amask, kv_write_mask=wmask),
                                state=st)
                cur = np.asarray(out["hidden"]).astype(np.float16).reshape(1, 1, D_MODEL)
                assert np.all(np.isfinite(cur))
            return _final_norm_softcap_logits(cur, gamma, eps, embed, softcap)

        t0 = time.perf_counter()
        last_logits = None
        for pos, tid in enumerate(prompt_ids):
            last_logits = step(tid, pos)
        prime_dt = time.perf_counter() - t0
        print(f"    prime: {prime_dt:.1f}s ({prime_dt/len(prompt_ids):.2f} s/tok)")

        gen_ids: list[int] = []
        top1_logits: list[tuple[int, float]] = []
        pos = len(prompt_ids)
        cur_logits = last_logits
        t0 = time.perf_counter()
        for _ in range(N_NEW):
            tid = int(np.argmax(cur_logits))
            top1_logits.append((tid, float(cur_logits[tid])))
            gen_ids.append(tid)
            cur_logits = step(tid, pos); pos += 1
        gen_dt = time.perf_counter() - t0
        completion = tok.decode(gen_ids, skip_special_tokens=False)
        print(f"    gen ids: {gen_ids}")
        print(f"    completion: {completion!r}")
        print(f"    gen wall: {gen_dt:.1f}s ({gen_dt/N_NEW:.2f} s/tok)")

        results.append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "gen_ids": gen_ids,
            "top1_logits": top1_logits,
            "prime_s": prime_dt,
            "gen_s": gen_dt,
        })

    payload = {
        "n_prompts": np.array(len(PROMPTS), dtype=np.int64),
        "n_new": np.array(N_NEW, dtype=np.int64),
    }
    for i, r in enumerate(results):
        payload[f"p{i}_prompt"] = np.array(r["prompt"])
        payload[f"p{i}_prompt_ids"] = np.array(r["prompt_ids"], dtype=np.int64)
        payload[f"p{i}_gen_ids"] = np.array(r["gen_ids"], dtype=np.int64)
        payload[f"p{i}_top1_logits"] = np.array(
            [v for _, v in r["top1_logits"]], dtype=np.float32)
        payload[f"p{i}_prime_s"] = np.array(r["prime_s"], dtype=np.float64)
        payload[f"p{i}_gen_s"] = np.array(r["gen_s"], dtype=np.float64)
    _atomic_write_npz(OUT_NPZ, **payload)
    print(f"\n  wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
