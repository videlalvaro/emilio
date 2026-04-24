"""gemma_t415_closedform_battery_gate.py — strict factual battery gate.

Loads the HF closed-form battery and checks whether the CoreML INT4 chain
matches the required answer prefix for each prompt. This does not replace the
open-ended T4.1.5 gate; it complements it with a low-entropy factual oracle.

Usage:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_t415_closedform_battery_gate.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import tokenizers

sys.path.insert(0, str(Path(__file__).parent))
from gemma_closedform_battery_spec import (  # noqa: E402
    BATTERY,
    GATE_SENTINEL,
    HF_OUT_NPZ,
    HF_SENTINEL,
    N_NEW,
    TOTAL_REQUIRED_PREFIX,
)
from gemma_to_ane import D_MODEL, GLB_ROT_DIM, SLD_D_HEAD  # noqa: E402
from gemma_t414_generate import (  # noqa: E402
    LOGIT_HEAD_NPZ,
    SHARDS,
    SHARD_PATHS,
    TOKENIZER_JSON,
    _attn_mask_for_pos,
    _final_norm_softcap_logits,
    _rope_for_pos,
    _write_mask_for_pos,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-key", default=None,
                    help="run only one prompt from the closed-form battery")
    args = ap.parse_args()

    print("=== T4.1.5 CLOSED-FORM FACTUAL BATTERY ===")
    for p in [LOGIT_HEAD_NPZ, TOKENIZER_JSON, HF_OUT_NPZ, HF_SENTINEL] + SHARD_PATHS:
        if not p.exists():
            print(f"FATAL missing: {p}", file=sys.stderr)
            return 2

    hf = np.load(HF_OUT_NPZ, allow_pickle=False)
    if int(hf["n_prompts"]) != len(BATTERY) or int(hf["n_new"]) != N_NEW:
        print("FATAL battery spec does not match saved HF oracle", file=sys.stderr)
        return 2

    head = np.load(LOGIT_HEAD_NPZ, allow_pickle=False)
    embed = head["embed_weight"]
    gamma = head["final_norm_gamma"]
    eps = float(head["rms_norm_eps"])
    softcap = float(head["softcap"])
    embed_scale = float(np.sqrt(D_MODEL))
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))

    print(f"  loading {len(SHARD_PATHS)} shards...")
    shard_models = []
    for (a, b), pth in zip(SHARDS, SHARD_PATHS):
        m = ct.models.CompiledMLModel(str(pth),
                                      compute_units=ct.ComputeUnit.CPU_AND_NE)
        shard_models.append(m)
        print(f"    shard[{a},{b}] loaded")

    total_prefix_match = 0
    passed_prompts = 0
    results = []

    selected = []
    for triple in BATTERY:
        if args.prompt_key is not None and triple[0] != args.prompt_key:
            continue
        selected.append(triple)
    if not selected:
        print("FATAL no prompts selected", file=sys.stderr)
        return 2

    for i, (key, prompt, required_prefix) in enumerate(BATTERY):
        if (key, prompt, required_prefix) not in selected:
            continue
        print(f"\n  === prompt {i} / {key}: {prompt!r} ===")
        prompt_ids = [int(x) for x in hf[f"p{i}_prompt_ids"]]
        hf_ids = [int(x) for x in hf[f"p{i}_gen_ids"]]
        assert prompt == str(hf[f"p{i}_prompt"])
        assert required_prefix == int(hf[f"p{i}_required_prefix"])
        assert bool(hf[f"p{i}_done"])
        assert bool(hf[f"p{i}_stable"])

        shard_states = [m.make_state() for m in shard_models]

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

        t0 = time.perf_counter()
        last_logits = None
        for pos, tid in enumerate(prompt_ids):
            try:
                last_logits = step(tid, pos)
            except FloatingPointError as exc:
                print(f"FATAL final-logit computation failed during prime at pos={pos}: {exc}",
                      file=sys.stderr)
                return 4
            if last_logits is None:
                return 3
        prime_dt = time.perf_counter() - t0
        print(f"    prime: {prime_dt:.1f}s ({prime_dt/len(prompt_ids):.2f} s/tok)")

        cml_ids = []
        pos = len(prompt_ids)
        cur_logits = last_logits
        t0 = time.perf_counter()
        for step_i in range(N_NEW):
            tid = int(np.argmax(cur_logits))
            cml_ids.append(tid)
            try:
                cur_logits = step(tid, pos)
            except FloatingPointError as exc:
                print(f"FATAL final-logit computation failed at decode step {step_i}, pos={pos}: {exc}",
                      file=sys.stderr)
                return 4
            if cur_logits is None:
                return 3
            pos += 1
        gen_dt = time.perf_counter() - t0

        matches = [c == h for c, h in zip(cml_ids, hf_ids)]
        prefix = 0
        for ok in matches:
            if ok:
                prefix += 1
            else:
                break
        pass_prompt = prefix >= required_prefix
        total_prefix_match += min(prefix, required_prefix)
        passed_prompts += int(pass_prompt)

        hf_text = tok.decode(hf_ids, skip_special_tokens=False)
        cml_text = tok.decode(cml_ids, skip_special_tokens=False)
        print(f"    hf : {hf_text!r}")
        print(f"    cml: {cml_text!r}")
        print(f"    match={sum(matches)}/{N_NEW}  prefix={prefix}  required={required_prefix}  "
              f"{'PASS' if pass_prompt else 'FAIL'}")
        results.append((key, prefix, required_prefix, pass_prompt, gen_dt))

    required_total = sum(req for _, _, req in selected)
    passed = (total_prefix_match == required_total) and (passed_prompts == len(selected))
    print(f"\n  prompts passed      : {passed_prompts}/{len(selected)}")
    print(f"  required prefixes   : {total_prefix_match}/{required_total}")
    print(f"  gate condition      : all prompts meet required prefix")

    if args.prompt_key is not None:
        print("\n  select mode: sentinel unchanged")
        return 0 if passed else 1

    if passed:
        GATE_SENTINEL.write_text(
            f"PASS prompts={passed_prompts}/{len(selected)} "
            f"required_prefix={total_prefix_match}/{required_total}\n"
        )
        print(f"\n# CLOSED-FORM BATTERY: PASS — sentinel {GATE_SENTINEL} written")
        return 0

    if GATE_SENTINEL.exists():
        GATE_SENTINEL.unlink()
        print(f"\n# CLOSED-FORM BATTERY: FAIL — prompts={passed_prompts}/{len(selected)} "
            f"required_prefix={total_prefix_match}/{required_total}")
    return 1


if __name__ == "__main__":
    sys.exit(main())