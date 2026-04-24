"""diag_pytorch_logit_gate.py — cheap PyTorch fp16 sanity for the logit gate.

Skips CoreML entirely. Runs the full 30-layer GemmaMixedStackWrap (all REAP
weights, full MoE branch, fp16) on the 6-token HF golden prompt and reports
per-position cos vs HF logits.

If this passes (cos>=0.97), the architecture is correct and any residual
T4.1.4 failure is purely a CoreML/INT4-quant gap → rebuilding shards with
the attn-scale fix is the next step.

If this fails, there's still an architectural bug (e.g. router math, MoE
weight-loading, position-embedding mismatch) and we debug locally before
burning time on CoreML rebuilds.

Run:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/diag_pytorch_logit_gate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaMixedStackWrap, _load_layer_weights, _layer_types_from_config,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM,
)
from gemma_mixedN_golden import _real_rope  # noqa: E402

OUT_DIR = Path("python/moe/out")
MAX_CTX = 1024
N = 30


def _rope_for_pos(theta: float, dh: int, pos: int):
    cos, sin = _real_rope(theta=theta, dh=dh, pos=pos)
    return (torch.from_numpy(cos.astype(np.float16).reshape(1, 1, dh)),
            torch.from_numpy(sin.astype(np.float16).reshape(1, 1, dh)))


def _attn_mask_for_pos(pos: int) -> torch.Tensor:
    m = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    m[..., : pos + 1] = 0.0
    return torch.from_numpy(m)


def _write_mask_for_pos(pos: int) -> torch.Tensor:
    w = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    w[0, 0, pos, 0] = 1.0
    return torch.from_numpy(w)


def _final_logits(hidden_fp16: np.ndarray, gamma: np.ndarray, eps: float,
                  embed: np.ndarray, softcap: float) -> np.ndarray:
    h = hidden_fp16.astype(np.float32).reshape(-1)
    rms = np.sqrt((h * h).mean() + eps)
    g32 = gamma.astype(np.float32)
    h_norm = (h / rms) * (1.0 + g32)
    e32 = embed.astype(np.float32)
    logits = e32 @ h_norm
    if softcap and softcap > 0:
        logits = np.tanh(logits / softcap) * softcap
    return logits


def _cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=== PyTorch fp16 logit gate (no CoreML, no INT4) ===")

    head = np.load(OUT_DIR / "gemma_logit_head.npz", allow_pickle=False)
    embed = head["embed_weight"]
    gamma = head["final_norm_gamma"]
    eps = float(head["rms_norm_eps"])
    softcap = float(head["softcap"])
    vocab, dm = embed.shape
    assert dm == D_MODEL

    gold = np.load(OUT_DIR / "gemma_hf_golden_logits.npz", allow_pickle=False)
    hf_logits = gold["logits"].astype(np.float32)
    input_ids = gold["input_ids"].astype(np.int64)
    T = int(input_ids.shape[0])
    print(f"  HF golden T={T} ids={input_ids.tolist()}")
    print(f"  HF top-1 last={int(np.argmax(hf_logits[-1]))}")

    # Streaming approach: hold only one layer at a time + per-layer KV caches.
    # Each KV cache is small (~2 MB sliding, ~2 MB global), so all 30 fit easily.
    # Layer weights are big (~700 MB), so rebuild & drop each step.
    layer_types = _layer_types_from_config(N)
    print(f"  layer_types: {layer_types}")
    print(f"  streaming layers (load → forward → free)")

    # Pre-allocate per-layer KV caches.
    from gemma_to_ane import _state_shape, GemmaSlidingLayer, GemmaGlobalLayer
    kv = []
    for i, t in enumerate(layer_types):
        shp = _state_shape(t == "global", MAX_CTX)
        kv.append((torch.zeros(*shp, dtype=torch.float16),
                   torch.zeros(*shp, dtype=torch.float16)))

    embed_scale = float(np.sqrt(D_MODEL))
    print(f"  embed_scale=sqrt(D)={embed_scale:.4f}")

    cos_per_pos = []
    pt_top1 = []
    t_decode = time.perf_counter()

    # We must process the prompt left→right per layer (KV grows). Process
    # all positions through layer 0, then layer 1, etc. — but that requires
    # buffering hiddens. With T=6 and D=2816 fp16, that's 6*5632 B = 34 KB per
    # layer, trivial. Strategy:
    #   For each layer i:
    #     load layer i once
    #     for pos in 0..T-1: run layer i on hidden[pos] (writes to kv[i])
    #     overwrite hiddens with this layer's output
    #     free layer i

    # Initialize hiddens with embedded tokens.
    hiddens = []
    for pos in range(T):
        tok = int(input_ids[pos])
        x_np = (embed[tok].astype(np.float32) * embed_scale).astype(np.float16)
        hiddens.append(torch.from_numpy(x_np.reshape(1, 1, D_MODEL)))

    # Pre-compute per-position rope/masks (cheap).
    rope_s = [_rope_for_pos(10_000.0, SLD_D_HEAD, p) for p in range(T)]
    rope_g = [_rope_for_pos(1_000_000.0, GLB_ROT_DIM, p) for p in range(T)]
    amasks = [_attn_mask_for_pos(p) for p in range(T)]
    wmasks = [_write_mask_for_pos(p) for p in range(T)]

    import gc
    with torch.no_grad():
        for i, t in enumerate(layer_types):
            t0 = time.perf_counter()
            if t == "global":
                layer = GemmaGlobalLayer(MAX_CTX).half().eval()
            else:
                layer = GemmaSlidingLayer(MAX_CTX).half().eval()
            _load_layer_weights(layer, OUT_DIR / f"gemma_layer{i}_packed.npz")
            kc, vc = kv[i]
            for pos in range(T):
                cs, sn = rope_g[pos] if t == "global" else rope_s[pos]
                h_in = hiddens[pos]
                h_out, k_new, v_new = layer(h_in, cs, sn, kc, vc,
                                            amasks[pos], wmasks[pos])
                # Write KV state (mimics CoreML coreml_update_state).
                kc[:] = k_new
                vc[:] = v_new
                hiddens[pos] = h_out
            del layer
            gc.collect()
            print(f"    L{i:>2} ({t[:3]}) "
                  f"loaded+ran T={T} pos in {(time.perf_counter()-t0)*1e3:.0f} ms")

    print(f"  decode wall: {(time.perf_counter()-t_decode)*1e3:.0f} ms")

    # Now compute logits per position.
    for pos in range(T):
        h_np = hiddens[pos].float().numpy().reshape(D_MODEL).astype(np.float16)
        logits = _final_logits(h_np, gamma, eps, embed, softcap)
        c = _cos(logits, hf_logits[pos])
        top1 = int(np.argmax(logits))
        cos_per_pos.append(c); pt_top1.append(top1)
        print(f"  pos={pos} tok={int(input_ids[pos]):6d}  "
              f"cos={c:.4f}  pt_top1={top1}  hf_top1={int(np.argmax(hf_logits[pos]))}")
    cml_top1 = pt_top1

    min_cos = min(cos_per_pos)
    last_top1 = cml_top1[-1]
    hf_last = int(np.argmax(hf_logits[-1]))
    print(f"\n  min_cos={min_cos:.4f}  cos_per_pos={[round(c,4) for c in cos_per_pos]}")
    print(f"  last_top1 pt={last_top1}  hf={hf_last}  "
          f"{'AGREE' if last_top1 == hf_last else 'DISAGREE'}")

    if min_cos >= 0.97 and last_top1 == hf_last:
        print("\n# PyTorch fp16 logit gate: PASS — architecture is correct, "
              "any T4.1.4 fail is purely CoreML/INT4 quant.")
        sys.exit(0)
    print("\n# PyTorch fp16 logit gate: FAIL — architecture still has issues.")
    sys.exit(1)


if __name__ == "__main__":
    main()
