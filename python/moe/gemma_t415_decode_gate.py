"""gemma_t415_decode_gate.py — T4.1.5 ship gate.

Closes the Python-side pipeline: 16-token greedy decode against the saved
HF reference in `python/moe/out/gemma_golden.npz`. Verifies the KV-state
read/write path stays correct over a longer decode horizon than T4.1.4
(which only checked next_token logits).

Golden file schema (already present):
    prompt           : "What is the capital of France?"
    prompt_ids       : (8,)  int32   — including <bos>
    logits_full      : (8, V) f32    — HF teacher-forced over the prompt
    next_token_ids   : (16,) int32   — HF greedy continuation
    next_token_logits: (16,V) f32    — HF logits at each greedy step

Procedure:
  1. Prime KV by feeding the 8 prompt tokens through the 3 chained shards.
  2. Greedy decode 16 tokens (CML argmax fed back as the next position).
  3. Compare CML generated_ids vs golden.next_token_ids elementwise.

Pass condition (paper-quality but realistic given INT4 + RMSNorm bake bug):
    exact_match >= 12 / 16   (75% byte-identity vs HF greedy)
    AND prefix-match >= 4    (first 4 tokens identical — captures ' Paris.')

Hard pass (sentinel `.gemma_t415_decode_gate_PASS`) is written only when
both gates clear. The first-divergence index is always reported so we can
see whether drift is at position 5 (early — bad) vs position 12 (late —
expected from compounding INT4 noise).

Run:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_t415_decode_gate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct
import tokenizers

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import D_MODEL, SLD_D_HEAD, GLB_ROT_DIM  # noqa: E402
from gemma_t414_generate import (  # noqa: E402
    SHARDS, SHARD_PATHS, LOGIT_HEAD_NPZ, TOKENIZER_JSON,
    _rope_for_pos, _attn_mask_for_pos, _write_mask_for_pos,
    _final_norm_softcap_logits,
)

OUT_DIR = Path("python/moe/out")
GOLDEN_NPZ = OUT_DIR / "gemma_golden.npz"
SENTINEL = OUT_DIR / ".gemma_t415_decode_gate_PASS"

PASS_EXACT = 12  # of 16
PASS_PREFIX = 4


def main() -> int:
    print("=== T4.1.5 DECODE GATE (16-token greedy vs HF golden) ===")
    for p in [LOGIT_HEAD_NPZ, TOKENIZER_JSON, GOLDEN_NPZ] + SHARD_PATHS:
        if not p.exists():
            print(f"FATAL missing: {p}", file=sys.stderr)
            return 2

    g = np.load(GOLDEN_NPZ, allow_pickle=False)
    prompt_ids = [int(x) for x in g["prompt_ids"]]
    golden_next = [int(x) for x in g["next_token_ids"]]
    golden_step_logits = g["next_token_logits"]
    n_new = len(golden_next)
    print(f"  prompt ids ({len(prompt_ids)}): {prompt_ids}")
    print(f"  golden next ({n_new}): {golden_next}")

    head = np.load(LOGIT_HEAD_NPZ, allow_pickle=False)
    embed = head["embed_weight"]
    gamma = head["final_norm_gamma"]
    eps = float(head["rms_norm_eps"])
    softcap = float(head["softcap"])
    embed_scale = float(np.sqrt(D_MODEL))

    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))
    print(f"  prompt str   : {tok.decode(prompt_ids, skip_special_tokens=False)!r}")
    print(f"  golden str   : {tok.decode(golden_next, skip_special_tokens=False)!r}")

    print(f"\n  loading {len(SHARD_PATHS)} shards...")
    shard_models = []
    shard_states = []
    for (a, b), pth in zip(SHARDS, SHARD_PATHS):
        m = ct.models.CompiledMLModel(str(pth),
                                       compute_units=ct.ComputeUnit.CPU_AND_NE)
        shard_models.append(m)
        shard_states.append(m.make_state())
        print(f"    shard[{a},{b}] loaded")

    def step(token_id: int, pos: int):
        x = (embed[token_id].astype(np.float32) * embed_scale).astype(np.float16)
        x = x.reshape(1, 1, D_MODEL)
        cos_s, sin_s = _rope_for_pos(10_000.0, SLD_D_HEAD, pos)
        cos_g, sin_g = _rope_for_pos(1_000_000.0, GLB_ROT_DIM, pos)
        amask = _attn_mask_for_pos(pos)
        wmask = _write_mask_for_pos(pos)
        cur = x
        for s_idx, (m, st) in enumerate(zip(shard_models, shard_states)):
            out = m.predict(dict(x=cur, cos_s=cos_s, sin_s=sin_s,
                                 cos_g=cos_g, sin_g=sin_g,
                                 attn_mask=amask, kv_write_mask=wmask),
                            state=st)
            cur = np.asarray(out["hidden"]).astype(np.float16).reshape(1, 1, D_MODEL)
            if not np.all(np.isfinite(cur)):
                print(f"FATAL non-finite hidden at pos={pos} shard_idx={s_idx}",
                      file=sys.stderr)
                return None
        return _final_norm_softcap_logits(cur, gamma, eps, embed, softcap)

    print(f"\n  priming KV with {len(prompt_ids)} prompt tokens...")
    t0 = time.perf_counter()
    last_logits = None
    for pos, t_id in enumerate(prompt_ids):
        last_logits = step(t_id, pos)
        if last_logits is None:
            return 3
    prime_dt = time.perf_counter() - t0
    print(f"  prime wall: {prime_dt*1e3:.0f} ms "
          f"({prime_dt*1e3/len(prompt_ids):.0f} ms/tok)")

    print(f"\n  greedy decode {n_new} tokens, comparing to golden:")
    generated: list[int] = []
    pos = len(prompt_ids)
    cur_logits = last_logits
    matches = []
    first_div = -1
    t0 = time.perf_counter()
    for i in range(n_new):
        cml_id = int(np.argmax(cur_logits))
        hf_id = golden_next[i]
        ok = (cml_id == hf_id)
        matches.append(ok)
        if not ok and first_div < 0:
            first_div = i
        cml_str = tok.decode([cml_id], skip_special_tokens=False)
        hf_str = tok.decode([hf_id], skip_special_tokens=False)
        cml_top2 = np.argsort(-cur_logits)[:2]
        cml_margin = float(cur_logits[cml_top2[0]] - cur_logits[cml_top2[1]])
        mark = "✓" if ok else "✗"
        print(f"    [{i:2d}] {mark} cml={cml_id:6d} {cml_str!r:18s}  "
              f"hf={hf_id:6d} {hf_str!r:18s}  cml_margin={cml_margin:6.3f}")
        if i < n_new - 1:
            hf_row = golden_step_logits[i]
            hf_top2 = np.argsort(-hf_row)[:2]
            hf_margin = float(hf_row[hf_top2[0]] - hf_row[hf_top2[1]])
            hf_pred = int(hf_top2[0])
            hf_pred_str = tok.decode([hf_pred], skip_special_tokens=False)
            print(f"           next-hf-pred={hf_pred:6d} {hf_pred_str!r:18s}  "
                  f"hf_margin={hf_margin:6.3f}")
        generated.append(cml_id)
        try:
            cur_logits = step(cml_id, pos)
        except FloatingPointError as exc:
            print(f"FATAL final-logit computation failed at decode step {i}, pos={pos}: {exc}",
                  file=sys.stderr)
            return 4
        if cur_logits is None:
            print("  aborting decode early due to non-finite hidden")
            break
        pos += 1
    gen_dt = time.perf_counter() - t0
    print(f"  decode wall: {gen_dt*1e3:.0f} ms "
          f"({gen_dt*1e3/n_new:.0f} ms/tok)")

    n_match = sum(matches)
    prefix = 0
    for ok in matches:
        if ok:
            prefix += 1
        else:
            break

    cml_str = tok.decode(generated, skip_special_tokens=False)
    hf_str = tok.decode(golden_next, skip_special_tokens=False)

    print(f"\n  exact match : {n_match}/{n_new}  (gate >= {PASS_EXACT})")
    print(f"  prefix      : {prefix}           (gate >= {PASS_PREFIX})")
    print(f"  first diverg: {first_div if first_div >= 0 else 'none'}")
    print(f"  cml string  : {cml_str!r}")
    print(f"  hf  string  : {hf_str!r}")

    passed = (n_match >= PASS_EXACT) and (prefix >= PASS_PREFIX)
    if passed:
        SENTINEL.write_text(
            f"PASS exact={n_match}/{n_new} prefix={prefix} "
            f"first_div={first_div}\n"
        )
        print(f"\n# T4.1.5 DECODE GATE: PASS — sentinel {SENTINEL} written")
        return 0
    else:
        if SENTINEL.exists():
            SENTINEL.unlink()
        print(f"\n# T4.1.5 DECODE GATE: FAIL — exact={n_match}/{n_new} "
              f"prefix={prefix}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
