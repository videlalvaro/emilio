"""diag_p2_cml_top50.py — capture CoreML INT4 chain top-50 logits at the same
position as diag_p2_hf_top50.py (last prompt position of "The first president
of the United States was").

Output: python/moe/out/diag_p2_cml_top50.npz
Usage:  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
            python/moe/diag_p2_cml_top50.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, coremltools as ct, tokenizers
sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import D_MODEL, SLD_D_HEAD, GLB_ROT_DIM
from gemma_t414_generate import (
    SHARDS, SHARD_PATHS, LOGIT_HEAD_NPZ, TOKENIZER_JSON,
    _rope_for_pos, _attn_mask_for_pos, _write_mask_for_pos,
    _final_norm_softcap_logits,
)

PROMPT = "The first president of the United States was"
OUT = Path("python/moe/out/diag_p2_cml_top50.npz")

def main():
    head = np.load(LOGIT_HEAD_NPZ, allow_pickle=False)
    embed = head["embed_weight"]; gamma = head["final_norm_gamma"]
    eps = float(head["rms_norm_eps"]); softcap = float(head["softcap"])
    embed_scale = float(np.sqrt(D_MODEL))
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))

    print("loading shards...")
    shard_models = [ct.models.CompiledMLModel(str(p),
                        compute_units=ct.ComputeUnit.CPU_AND_NE) for p in SHARD_PATHS]
    shard_states = [m.make_state() for m in shard_models]

    ids = tok.encode(PROMPT).ids
    if ids[0] != 2: ids = [2] + ids
    print(f"prompt ids ({len(ids)}): {ids}")

    last_logits = None
    t0 = time.perf_counter()
    for pos, tid in enumerate(ids):
        x = (embed[tid].astype(np.float32) * embed_scale).astype(np.float16)
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
        last_logits = _final_norm_softcap_logits(cur, gamma, eps, embed, softcap)
    print(f"prime wall {time.perf_counter()-t0:.1f}s")

    top50 = np.argsort(-last_logits)[:50]
    print("CML top-50 at last prompt position:")
    for r, tid in enumerate(top50):
        print(f"  [{r:2d}] id={int(tid):7d} L={last_logits[tid]:7.3f}  "
              f"{tok.decode([int(tid)], skip_special_tokens=False)!r}")
    GEORGE = 9142
    rk = int(np.where(np.argsort(-last_logits) == GEORGE)[0][0])
    print(f"\n  ' George' (id={GEORGE}) rank in CML: {rk}  "
          f"L={last_logits[GEORGE]:.3f}  vs CML top1 L={last_logits[top50[0]]:.3f}  "
          f"gap={last_logits[top50[0]]-last_logits[GEORGE]:.3f}")
    np.savez(str(OUT),
             prompt=np.array(PROMPT),
             input_ids=np.array(ids, dtype=np.int64),
             last_pos_logits=last_logits.astype(np.float32),
             top50_ids=top50.astype(np.int64),
             top50_logits=last_logits[top50].astype(np.float32),
             george_rank=np.array(rk, dtype=np.int64),
             george_logit=np.array(last_logits[GEORGE], dtype=np.float32))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
