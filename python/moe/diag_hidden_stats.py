"""diag_hidden_stats.py — probe per-position hidden statistics to localize
the fp16 numerical blow-up at pos 0 and pos 2.

Re-runs the streaming PyTorch fp16 forward but instead of just printing cos
at the end, dumps stats of the final-layer hidden for each position:
  ‖h‖, |h|.max, |h|.min(nonzero), # of |h|>=fp16_max/2, # of NaN/Inf,
  RMS = sqrt(mean(h^2)), and what the post-norm hidden looks like.

Goal: find which position's hidden goes pathological and where the matmul
divide-by-zero/overflow comes from.
"""
from __future__ import annotations
import sys, time, gc
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    _load_layer_weights, _layer_types_from_config,
    _state_shape, GemmaSlidingLayer, GemmaGlobalLayer,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM,
)
from gemma_mixedN_golden import _real_rope  # noqa: E402

OUT_DIR = Path("python/moe/out")
MAX_CTX = 1024
N = 30
FP16_MAX = 65504.0


def _rope(theta, dh, pos):
    cos, sin = _real_rope(theta=theta, dh=dh, pos=pos)
    return (torch.from_numpy(cos.astype(np.float16).reshape(1, 1, dh)),
            torch.from_numpy(sin.astype(np.float16).reshape(1, 1, dh)))


def _amask(pos):
    m = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    m[..., : pos + 1] = 0.0
    return torch.from_numpy(m)


def _wmask(pos):
    w = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    w[0, 0, pos, 0] = 1.0
    return torch.from_numpy(w)


def _stats(name, h_np):
    h = h_np.reshape(-1).astype(np.float64)
    nan = int(np.isnan(h).sum())
    inf = int(np.isinf(h).sum())
    h_finite = h[np.isfinite(h)]
    nrm = float(np.linalg.norm(h_finite))
    amax = float(np.max(np.abs(h_finite))) if h_finite.size else 0.0
    rms = float(np.sqrt((h_finite ** 2).mean())) if h_finite.size else 0.0
    n_huge = int((np.abs(h_finite) >= FP16_MAX / 2).sum())
    nz = h_finite[h_finite != 0]
    amin = float(np.min(np.abs(nz))) if nz.size else 0.0
    print(f"  {name}: ‖h‖={nrm:.3e} max|h|={amax:.3e} rms={rms:.3e} "
          f"min|h|nz={amin:.3e} huge(≥fp16max/2)={n_huge} nan={nan} inf={inf}")


def main():
    print("=== diag_hidden_stats: per-position hidden audit ===")
    head = np.load(OUT_DIR / "gemma_logit_head.npz", allow_pickle=False)
    embed = head["embed_weight"]
    gamma = head["final_norm_gamma"]
    eps = float(head["rms_norm_eps"])
    softcap = float(head["softcap"])
    gold = np.load(OUT_DIR / "gemma_hf_golden_logits.npz", allow_pickle=False)
    hf_logits = gold["logits"].astype(np.float32)
    input_ids = gold["input_ids"].astype(np.int64)
    T = int(input_ids.shape[0])
    embed_scale = float(np.sqrt(D_MODEL))

    layer_types = _layer_types_from_config(N)

    # KV caches
    kv = []
    for i, t in enumerate(layer_types):
        shp = _state_shape(t == "global", MAX_CTX)
        kv.append((torch.zeros(*shp, dtype=torch.float16),
                   torch.zeros(*shp, dtype=torch.float16)))

    # Init hiddens with embedded tokens.
    hiddens = []
    for pos in range(T):
        tok = int(input_ids[pos])
        x_np = (embed[tok].astype(np.float32) * embed_scale).astype(np.float16)
        hiddens.append(torch.from_numpy(x_np.reshape(1, 1, D_MODEL)))

    print("\n--- after embedding (input to L0) ---")
    for pos in range(T):
        _stats(f"pos={pos} tok={int(input_ids[pos]):6d}",
               hiddens[pos].float().numpy())

    rope_s = [_rope(10_000.0, SLD_D_HEAD, p) for p in range(T)]
    rope_g = [_rope(1_000_000.0, GLB_ROT_DIM, p) for p in range(T)]
    amasks = [_amask(p) for p in range(T)]
    wmasks = [_wmask(p) for p in range(T)]

    # Stream layers; print stats at each global layer + after final layer.
    PROBE_LAYERS = {0, 5, 11, 17, 23, 28, 29}
    with torch.no_grad():
        for i, t in enumerate(layer_types):
            t0 = time.perf_counter()
            layer = (GemmaGlobalLayer(MAX_CTX) if t == "global"
                     else GemmaSlidingLayer(MAX_CTX)).half().eval()
            _load_layer_weights(layer, OUT_DIR / f"gemma_layer{i}_packed.npz")
            kc, vc = kv[i]
            for pos in range(T):
                cs, sn = rope_g[pos] if t == "global" else rope_s[pos]
                h_in = hiddens[pos]
                h_out, k_new, v_new = layer(h_in, cs, sn, kc, vc,
                                            amasks[pos], wmasks[pos])
                kc[:] = k_new; vc[:] = v_new
                hiddens[pos] = h_out
            del layer; gc.collect()
            if i in PROBE_LAYERS:
                print(f"\n--- after L{i} ({t}) [{(time.perf_counter()-t0)*1e3:.0f} ms] ---")
                for pos in range(T):
                    _stats(f"pos={pos}", hiddens[pos].float().numpy())

    # Now compute logits both ways: fp16-cast vs fp32-clean.
    print("\n--- final-norm + softcap analysis ---")
    g32 = gamma.astype(np.float32)
    e32 = embed.astype(np.float32)
    for pos in range(T):
        # The buggy path (fp16 hidden, fp32 norm).
        h16 = hiddens[pos].float().numpy().reshape(D_MODEL).astype(np.float16)
        h32 = h16.astype(np.float32)
        rms = float(np.sqrt((h32 ** 2).mean() + eps))
        h_norm = (h32 / rms) * (1.0 + g32)
        n_huge = int((np.abs(h_norm) >= FP16_MAX / 2).sum())
        nan = int(np.isnan(h_norm).sum())
        inf = int(np.isinf(h_norm).sum())

        # Clean path: keep fp32 from hidden onwards.
        h32_clean = hiddens[pos].float().numpy().reshape(D_MODEL)
        rms_c = float(np.sqrt((h32_clean ** 2).mean() + eps))
        hn_c = (h32_clean / rms_c) * (1.0 + g32)

        with np.errstate(all="ignore"):
            logits_buggy = e32 @ h_norm
            logits_clean = e32 @ hn_c
        if softcap > 0:
            logits_buggy = np.tanh(logits_buggy / softcap) * softcap
            logits_clean = np.tanh(logits_clean / softcap) * softcap

        def _cos(a, b):
            a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
            return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

        c_buggy = _cos(logits_buggy, hf_logits[pos])
        c_clean = _cos(logits_clean, hf_logits[pos])
        top_b = int(np.argmax(np.where(np.isfinite(logits_buggy), logits_buggy, -1e9)))
        top_c = int(np.argmax(logits_clean))
        hf_top = int(np.argmax(hf_logits[pos]))

        print(f"  pos={pos}: rms16={rms:.3e} rms32={rms_c:.3e} "
              f"h_norm huge={n_huge} nan={nan} inf={inf}")
        print(f"          cos_buggy={c_buggy:.4f} top1={top_b}  "
              f"cos_clean={c_clean:.4f} top1={top_c}  hf_top1={hf_top}")


if __name__ == "__main__":
    main()
