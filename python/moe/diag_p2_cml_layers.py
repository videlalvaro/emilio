"""diag_p2_cml_layers.py — capture CML hidden states at each shard boundary.

Same prompt as diag_p2_hf_layers.py. Output schema mirrors HF capture so that
diff_p2_layers.py can compute per-layer drift metrics.

Output: python/moe/out/diag_p2_cml_layers.npz with
  hidden_norms[T, 4]                    ‖h_pos‖₂ at: embed, post-shard0(L15),
                                                       post-shard1(L22),
                                                       post-shard2(L30)
  hidden_last_pos_per_boundary[4, D]    h at last prompt pos at each boundary
  input_ids[T]
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
)

PROMPT = "The first president of the United States was"
OUT = Path("python/moe/out/diag_p2_cml_layers.npz")

def main():
    head = np.load(LOGIT_HEAD_NPZ, allow_pickle=False)
    embed = head["embed_weight"]
    embed_scale = float(np.sqrt(D_MODEL))
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))

    print("loading shards...")
    shard_models = [ct.models.CompiledMLModel(str(p),
                        compute_units=ct.ComputeUnit.CPU_AND_NE) for p in SHARD_PATHS]
    shard_states = [m.make_state() for m in shard_models]

    ids = tok.encode(PROMPT).ids
    if ids[0] != 2: ids = [2] + ids
    T = len(ids)
    print(f"prompt ids ({T}): {ids}")

    B = 1 + len(SHARDS)  # embed + after each of 3 shards
    norms = np.zeros((T, B), dtype=np.float32)
    last_per_b = np.zeros((B, D_MODEL), dtype=np.float32)

    t0 = time.perf_counter()
    for pos, tid in enumerate(ids):
        emb_vec = (embed[tid].astype(np.float32) * embed_scale).astype(np.float32)
        norms[pos, 0] = float(np.linalg.norm(emb_vec))
        if pos == T - 1:
            last_per_b[0] = emb_vec.copy()

        x = emb_vec.astype(np.float16).reshape(1, 1, D_MODEL)
        cos_s, sin_s = _rope_for_pos(10_000.0, SLD_D_HEAD, pos)
        cos_g, sin_g = _rope_for_pos(1_000_000.0, GLB_ROT_DIM, pos)
        amask = _attn_mask_for_pos(pos); wmask = _write_mask_for_pos(pos)
        cur = x
        for si, (m, st) in enumerate(zip(shard_models, shard_states)):
            out = m.predict(dict(x=cur, cos_s=cos_s, sin_s=sin_s,
                                 cos_g=cos_g, sin_g=sin_g,
                                 attn_mask=amask, kv_write_mask=wmask),
                            state=st)
            cur = np.asarray(out["hidden"]).astype(np.float16).reshape(1, 1, D_MODEL)
            v32 = cur.astype(np.float32).reshape(D_MODEL)
            norms[pos, si + 1] = float(np.linalg.norm(v32))
            if pos == T - 1:
                last_per_b[si + 1] = v32.copy()
    print(f"prime wall {time.perf_counter()-t0:.1f}s")

    print("‖h_last‖ per boundary: ",
          [f"B{b}:{norms[-1, b]:.1f}" for b in range(B)])
    np.savez(str(OUT),
             prompt=np.array(PROMPT),
             input_ids=np.array(ids, dtype=np.int64),
             hidden_norms=norms,
             hidden_last_pos_per_boundary=last_per_b,
             boundary_layers=np.array([0, 15, 22, 30], dtype=np.int64))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
