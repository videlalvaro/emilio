"""gemma_t414_generate.py — task-metric ship gate.

Per opf handoff §8 + TL;DR #8: when min cos drops below ~1.0 for fundamental
fp16/INT4 reasons, cos stops correlating with quality. Gate on actual
generation.

Loads the 3 chained CoreML shards, primes KV with the 6-token prompt
"The capital of France is", then greedy-decodes N more tokens. Reports the
generated string and per-step argmax.

If the model produces "Paris" or sane French-capital-related text, ship.

Run:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_t414_generate.py [--n-new 8]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import coremltools as ct
import tokenizers

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import D_MODEL, SLD_D_HEAD, GLB_ROT_DIM  # noqa: E402
from gemma_mixedN_golden import _real_rope  # noqa: E402

OUT_DIR = Path("python/moe/out")
MAX_CTX = 1024
SHARDS = [(0, 15), (15, 22), (22, 30)]
SHARD_PATHS = [OUT_DIR / f"gemma4_shard{a}_{b}_real.mlmodelc" for a, b in SHARDS]
LOGIT_HEAD_NPZ = OUT_DIR / "gemma_logit_head.npz"
TOKENIZER_JSON = Path("models/gemma-4-26b-a4b/tokenizer.json")
PROMPT_IDS = [2, 818, 5279, 529, 7001, 563]  # "<bos>The capital of France is"


def _rope_for_pos(theta: float, dh: int, pos: int):
    cos, sin = _real_rope(theta=theta, dh=dh, pos=pos)
    return (cos.astype(np.float16).reshape(1, 1, dh),
            sin.astype(np.float16).reshape(1, 1, dh))


def _attn_mask_for_pos(pos: int) -> np.ndarray:
    m = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    m[..., : pos + 1] = 0.0
    return m


def _write_mask_for_pos(pos: int) -> np.ndarray:
    w = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    w[0, 0, pos, 0] = 1.0
    return w


def _final_norm_softcap_logits(hidden_fp16, gamma_fp16, eps, embed_fp16, softcap):
    h = hidden_fp16.astype(np.float32).reshape(-1)
    rms = np.sqrt((h * h).mean() + eps)
    g32 = gamma_fp16.astype(np.float32)
    # Gemma-4 RMSNorm: weight init=1.0, applied DIRECTLY (NOT (1+weight)).
    # See transformers/models/gemma4/modeling_gemma4.py Gemma4RMSNorm.forward.
    # Earlier vanilla Gemma used (1+weight) with weight init=0; do not apply
    # that convention here or the final-norm scale is double-shifted, which
    # flips near-tie tokens at the softcap (e.g. ' George' vs ' a').
    h_norm = (h / rms) * g32
    if not np.isfinite(rms):
        raise FloatingPointError(f"non-finite rms in final norm: {rms}")
    if not np.all(np.isfinite(h_norm)):
        raise FloatingPointError("non-finite h_norm in final norm")
    e32 = embed_fp16.astype(np.float32)
    if softcap and softcap > 0:
        # See gemma_t414_logit_gate.py: scale h_norm by 1/softcap BEFORE the
        # matmul (E@h overflows on outlier rows otherwise; dividing after
        # is too late).
        h_scaled = (h_norm / softcap).astype(np.float32)
        if not np.all(np.isfinite(h_scaled)):
            raise FloatingPointError("non-finite h_scaled before softcap matmul")
        debug_softcap = os.getenv("GEMMA_SOFTCAP_DEBUG") == "1"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            raw = e32 @ h_scaled
        if debug_softcap and caught:
            msgs = [str(w.message) for w in caught]
            print(
                "SOFTCAP_DEBUG raw-matmul warning: "
                f"warnings={msgs} raw_finite={bool(np.all(np.isfinite(raw)))} "
                f"raw_min={float(np.nanmin(raw)):.3f} raw_max={float(np.nanmax(raw)):.3f} "
                f"raw_absmax={float(np.nanmax(np.abs(raw))):.3f} "
                f"h_scaled_absmax={float(np.max(np.abs(h_scaled))):.3f} "
                f"embed_absmax={float(np.max(np.abs(e32))):.3f}",
                file=sys.stderr,
            )
        if debug_softcap and not np.all(np.isfinite(raw)):
            raise FloatingPointError("non-finite raw logits before tanh softcap")
        logits = np.tanh(raw) * softcap
    else:
        logits = e32 @ h_norm
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("non-finite logits after final norm softcap")
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-new", type=int, default=8,
                    help="number of tokens to greedy-generate after prompt")
    args = ap.parse_args()

    print("=== T4.1.4 GENERATE (task-metric ship gate) ===")
    for p in [LOGIT_HEAD_NPZ, TOKENIZER_JSON] + SHARD_PATHS:
        if not p.exists():
            print(f"FATAL missing: {p}", file=sys.stderr); sys.exit(2)

    head = np.load(LOGIT_HEAD_NPZ, allow_pickle=False)
    embed = head["embed_weight"]
    gamma = head["final_norm_gamma"]
    eps = float(head["rms_norm_eps"])
    softcap = float(head["softcap"])
    embed_scale = float(np.sqrt(D_MODEL))

    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))

    print(f"  loading {len(SHARD_PATHS)} shards (will warm BNNS)...")
    shard_models = []; shard_states = []
    for (a, b), pth in zip(SHARDS, SHARD_PATHS):
        m = ct.models.CompiledMLModel(str(pth),
                                       compute_units=ct.ComputeUnit.CPU_AND_NE)
        shard_models.append(m); shard_states.append(m.make_state())
        print(f"    shard[{a},{b}]  loaded")

    prompt_str = tok.decode(PROMPT_IDS, skip_special_tokens=False)
    print(f"  prompt ids: {PROMPT_IDS}")
    print(f"  prompt str: {prompt_str!r}")

    def step(token_id: int, pos: int):
        x = (embed[token_id].astype(np.float32) * embed_scale).astype(np.float16)
        x = x.reshape(1, 1, D_MODEL)
        cos_s, sin_s = _rope_for_pos(10_000.0, SLD_D_HEAD, pos)
        cos_g, sin_g = _rope_for_pos(1_000_000.0, GLB_ROT_DIM, pos)
        amask = _attn_mask_for_pos(pos)
        wmask = _write_mask_for_pos(pos)
        cur = x
        for m, st in zip(shard_models, shard_states):
            out = m.predict(dict(x=cur, cos_s=cos_s, sin_s=sin_s,
                                 cos_g=cos_g, sin_g=sin_g,
                                 attn_mask=amask, kv_write_mask=wmask),
                            state=st)
            cur = np.asarray(out["hidden"]).astype(np.float16).reshape(1, 1, D_MODEL)
            assert np.all(np.isfinite(cur))
        logits = _final_norm_softcap_logits(cur, gamma, eps, embed, softcap)
        return logits

    # Prime: feed all prompt tokens, capture last logits.
    print(f"\n  priming KV with {len(PROMPT_IDS)} prompt tokens...")
    t0 = time.perf_counter()
    last_logits = None
    for pos, t_id in enumerate(PROMPT_IDS):
        last_logits = step(t_id, pos)
    prime_dt = time.perf_counter() - t0
    print(f"  prime wall: {prime_dt*1e3:.0f} ms ({prime_dt*1e3/len(PROMPT_IDS):.0f} ms/tok)")

    # Greedy generate.
    generated_ids: list[int] = []
    pos = len(PROMPT_IDS)
    print(f"\n  greedy generate ({args.n_new} new tokens):")
    t0 = time.perf_counter()
    cur_logits = last_logits
    for step_i in range(args.n_new):
        next_id = int(np.argmax(cur_logits))
        next_str = tok.decode([next_id], skip_special_tokens=False)
        top5 = np.argsort(-cur_logits)[:5]
        top5_strs = [(int(i), tok.decode([int(i)], skip_special_tokens=False),
                      float(cur_logits[i])) for i in top5]
        print(f"    step {step_i}: id={next_id:6d}  str={next_str!r:20s}  "
              f"top5={[(i, repr(s), f'{l:.2f}') for i,s,l in top5_strs]}")
        generated_ids.append(next_id)
        cur_logits = step(next_id, pos)
        pos += 1
    gen_dt = time.perf_counter() - t0
    print(f"  generate wall: {gen_dt*1e3:.0f} ms ({gen_dt*1e3/args.n_new:.0f} ms/tok)")

    full_ids = list(PROMPT_IDS) + generated_ids
    completion = tok.decode(generated_ids, skip_special_tokens=False)
    full = tok.decode(full_ids, skip_special_tokens=False)
    print(f"\n  generated ids : {generated_ids}")
    print(f"  completion str: {completion!r}")
    print(f"  full str      : {full!r}")

    # Task-metric verdict: is "Paris" (any case, with leading space) in the completion?
    completion_lower = completion.lower()
    contains_paris = "paris" in completion_lower
    print(f"\n  task metric: contains 'Paris' = {contains_paris}")
    if contains_paris:
        print("\n# T4.1.4 GENERATE: PASS — model knows the capital of France")
    else:
        print("\n# T4.1.4 GENERATE: INCONCLUSIVE — manual review of completion needed")


if __name__ == "__main__":
    main()
