"""diag_t414_per_shard.py — Path A: localize where T4.1.4 cos collapses.

Same prompt, same shards, same logit head as gemma_t414_logit_gate.py — but
records ‖hidden‖ AFTER each shard at each position. Pattern interpretation:

  monotone-with-depth growth/shrink across shards   → quant noise compounds
  one shard delivers wildly different norm than rest → that shard is broken
  pos=1 norm ≪ pos=0 norm at shard 0                → attn output dies first
  pos=1 hidden ≈ pos=0 hidden after shard k          → layers k.. are no-op

Reads gemma_hf_golden_logits.npz to compare per-pos cos against HF.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))
from gemma_mixedN_golden import _real_rope  # noqa: E402

OUT_DIR = Path("python/moe/out")
LOGIT_HEAD_NPZ = OUT_DIR / "gemma_logit_head.npz"
HF_GOLDEN = OUT_DIR / "gemma_hf_golden_logits.npz"
SHARDS = [(0, 15), (15, 22), (22, 30)]
SHARD_PATHS = [OUT_DIR / f"gemma4_shard{a}_{b}_real.mlmodelc" for a, b in SHARDS]
D_MODEL = 2816
SLD_DH = 256
GLB_ROT = 128
MAX_CTX = 1024


def _rope(theta, dh, pos):
    cs, sn = _real_rope(theta=theta, dh=dh, pos=pos)
    return (cs.astype(np.float16).reshape(1, 1, dh),
            sn.astype(np.float16).reshape(1, 1, dh))


def _amask(pos):
    m = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    m[..., : pos + 1] = 0.0
    return m


def _wmask(pos):
    w = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    w[0, 0, pos, 0] = 1.0
    return w


def _norm(a):
    return float(np.linalg.norm(a.astype(np.float64).ravel()))


def _cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _logits(h_fp16, gamma, eps, embed, softcap):
    h = h_fp16.astype(np.float32).reshape(-1)
    rms = np.sqrt((h * h).mean() + eps)
    h_norm = (h / rms) * gamma.astype(np.float32)  # Gemma-4 RMSNorm: direct
    out = embed.astype(np.float32) @ h_norm
    if softcap > 0:
        out = np.tanh(out / softcap) * softcap
    return out


def main():
    print("=== Path A: per-shard diagnostic for T4.1.4 ===")
    head = np.load(LOGIT_HEAD_NPZ)
    embed = head["embed_weight"]; gamma = head["final_norm_gamma"]
    eps = float(head["rms_norm_eps"]); softcap = float(head["softcap"])

    gold = np.load(HF_GOLDEN)
    hf_logits = gold["logits"].astype(np.float32)
    input_ids = gold["input_ids"].astype(np.int64)
    T = int(input_ids.shape[0])
    print(f"  prompt T={T}  ids={input_ids.tolist()}")

    print("  loading 3 shards...")
    models = []; states = []
    for (a, b), pth in zip(SHARDS, SHARD_PATHS):
        m = ct.models.CompiledMLModel(str(pth),
                                       compute_units=ct.ComputeUnit.CPU_AND_NE)
        models.append(m); states.append(m.make_state())

    embed_scale = float(np.sqrt(D_MODEL))
    # hidden_after_shard[shard_idx][pos] -> fp32 D_MODEL vector
    H = [[None] * T for _ in range(len(SHARDS))]
    EMB = [None] * T

    for pos in range(T):
        tok = int(input_ids[pos])
        x = (embed[tok].astype(np.float32) * embed_scale).astype(np.float16)
        x = x.reshape(1, 1, D_MODEL)
        EMB[pos] = (embed[tok].astype(np.float32) * embed_scale).astype(np.float32).copy()
        cs_s, sn_s = _rope(10_000.0, SLD_DH, pos)
        cs_g, sn_g = _rope(1_000_000.0, GLB_ROT, pos)
        base = dict(cos_s=cs_s, sin_s=sn_s, cos_g=cs_g, sin_g=sn_g,
                    attn_mask=_amask(pos), kv_write_mask=_wmask(pos))
        cur = x
        for si, (m, st) in enumerate(zip(models, states)):
            t0 = time.perf_counter()
            out = m.predict(dict(base, x=cur), state=st)
            dt = time.perf_counter() - t0
            cur = np.asarray(out["hidden"]).astype(np.float16).reshape(1, 1, D_MODEL)
            H[si][pos] = cur.astype(np.float32).reshape(D_MODEL).copy()
            if not np.all(np.isfinite(cur)):
                print(f"FATAL non-finite pos={pos} shard={si}", file=sys.stderr)
                sys.exit(2)
        # final logit head
        logits = _logits(cur, gamma, eps, embed, softcap)
        c = _cos(logits, hf_logits[pos])
        top1 = int(np.argmax(logits))
        hf_top1 = int(np.argmax(hf_logits[pos]))
        print(f"  pos={pos} tok={tok:6d}  cos_logits={c:+.4f}  "
              f"cml_top1={top1}  hf_top1={hf_top1}")

    print()
    print("  ---- per-shard ‖hidden‖ table (rows=pos, cols=after shard) ----")
    print("  pos | embed‖x‖ | shd0[0:15] | shd1[15:22] | shd2[22:30]")
    for pos in range(T):
        row = [_norm(EMB[pos])] + [_norm(H[si][pos]) for si in range(len(SHARDS))]
        print(f"  {pos:>3} | {row[0]:>8.2f} | {row[1]:>10.2f} | {row[2]:>11.2f} | {row[3]:>11.2f}")

    print()
    print("  ---- cos(pos=k vs pos=0) at each shard output ----")
    print("  shard | pos1     pos2     pos3     pos4     pos5")
    for si in range(len(SHARDS)):
        cs = [_cos(H[si][0], H[si][p]) for p in range(1, T)]
        s = "  ".join(f"{c:+.3f}" for c in cs)
        print(f"  shd{si}  | {s}")

    print()
    print("  ---- delta growth: ‖H[s][p] - H[s-1][p]‖ / ‖H[s-1][p]‖ (shard contribution) ----")
    print("  pos | shd0_vs_emb  shd1_vs_shd0  shd2_vs_shd1")
    for p in range(T):
        d0 = np.linalg.norm(H[0][p] - EMB[p]) / (_norm(EMB[p]) + 1e-9)
        d1 = np.linalg.norm(H[1][p] - H[0][p]) / (_norm(H[0][p]) + 1e-9)
        d2 = np.linalg.norm(H[2][p] - H[1][p]) / (_norm(H[1][p]) + 1e-9)
        print(f"  {p:>3} | {d0:>10.3f}   {d1:>10.3f}   {d2:>10.3f}")

    print()
    print("  ---- residual fraction: cos(H[s][p], EMB[p]) (high=layers acted as identity) ----")
    print("  pos | shd0    shd1    shd2")
    for p in range(T):
        c0 = _cos(H[0][p], EMB[p])
        c1 = _cos(H[1][p], EMB[p])
        c2 = _cos(H[2][p], EMB[p])
        print(f"  {p:>3} | {c0:+.3f}  {c1:+.3f}  {c2:+.3f}")


if __name__ == "__main__":
    main()
