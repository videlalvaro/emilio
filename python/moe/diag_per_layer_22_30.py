"""diag_per_layer_22_30.py — Path A.5 per-layer cliff probe inside shd2.

Loads layers 22..29 individually in PyTorch fp16 (NO quantization). Feeds the
HF prompt embeddings through CoreML shd0 + shd1 to get the real input to layer
22 at each position, then runs layers 22..29 in fp16 PyTorch and reports per-
layer ‖hidden‖ + cos(pos_k, pos_0).

If the attractor APPEARS in fp16 too → the bug is architectural / weights /
RoPE for those layers, not INT4 quant.
If the attractor is ABSENT in fp16 → confirms INT4 quant cliff. The exact
layer where cos(pos_k, pos_0) jumps is the cliff layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaSlidingLayer, GemmaGlobalLayer, _load_layer_weights,
    _layer_types_from_config,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM, GLB_D_HEAD,
)
from gemma_mixedN_golden import _real_rope  # noqa: E402

OUT_DIR = Path("python/moe/out")
HEAD_NPZ = OUT_DIR / "gemma_logit_head.npz"
HF_GOLDEN = OUT_DIR / "gemma_hf_golden_logits.npz"
SHD0 = OUT_DIR / "gemma4_shard0_15_real.mlmodelc"
SHD1 = OUT_DIR / "gemma4_shard15_22_real.mlmodelc"
LAYER_RANGE = (22, 30)
MAX_CTX = 1024


def _rope16(theta, dh, pos):
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


@torch.no_grad()
def main():
    print(f"=== Path A.5: per-layer probe layers {LAYER_RANGE[0]}..{LAYER_RANGE[1]-1} ===")

    # --- HF prompt ---
    head = np.load(HEAD_NPZ)
    embed = head["embed_weight"]
    gold = np.load(HF_GOLDEN)
    input_ids = gold["input_ids"].astype(np.int64)
    T = int(input_ids.shape[0])
    embed_scale = float(np.sqrt(D_MODEL))
    print(f"  prompt T={T}  ids={input_ids.tolist()}")

    # --- Run shd0 + shd1 in CoreML to get real input to layer 22 per position ---
    print(f"  loading {SHD0.name} + {SHD1.name}...")
    m0 = ct.models.CompiledMLModel(str(SHD0), compute_units=ct.ComputeUnit.CPU_AND_NE)
    m1 = ct.models.CompiledMLModel(str(SHD1), compute_units=ct.ComputeUnit.CPU_AND_NE)
    s0 = m0.make_state(); s1 = m1.make_state()

    INP22 = [None] * T  # (1,1,D) fp16 input to layer 22 at each position
    for pos in range(T):
        tok = int(input_ids[pos])
        x = (embed[tok].astype(np.float32) * embed_scale).astype(np.float16).reshape(1, 1, D_MODEL)
        cs_s, sn_s = _rope16(10_000.0, SLD_D_HEAD, pos)
        cs_g, sn_g = _rope16(1_000_000.0, GLB_ROT_DIM, pos)
        base = dict(cos_s=cs_s, sin_s=sn_s, cos_g=cs_g, sin_g=sn_g,
                    attn_mask=_amask(pos), kv_write_mask=_wmask(pos))
        h = m0.predict(dict(base, x=x), state=s0)["hidden"]
        h = np.asarray(h).astype(np.float16).reshape(1, 1, D_MODEL)
        h = m1.predict(dict(base, x=h), state=s1)["hidden"]
        INP22[pos] = np.asarray(h).astype(np.float16).reshape(1, 1, D_MODEL)
    del m0, m1, s0, s1

    # --- Build per-layer fp16 PyTorch modules for layers 22..29 ---
    full_types = _layer_types_from_config(30)
    layer_types = full_types[LAYER_RANGE[0]:LAYER_RANGE[1]]
    print(f"  layer_types[22..29] = {layer_types}")
    layers = []
    for li, t in enumerate(layer_types):
        gi = LAYER_RANGE[0] + li
        if t == "global":
            mod = GemmaGlobalLayer(MAX_CTX)
        else:
            mod = GemmaSlidingLayer(MAX_CTX)
        mod.half().eval()
        npz = OUT_DIR / f"gemma_layer{gi}_packed.npz"
        _load_layer_weights(mod, npz)
        layers.append(mod)

    # KV state per layer (sliding vs global shapes via _state_shape inside layer's k/v_cache).
    # Actually layers don't own state in this codebase — state is the wrap's buffer.
    # We'll allocate state tensors here per layer.
    from gemma_to_ane import _state_shape
    states = []
    for li, t in enumerate(layer_types):
        shp = _state_shape(t == "global", MAX_CTX)
        states.append([torch.zeros(*shp, dtype=torch.float16),
                       torch.zeros(*shp, dtype=torch.float16)])

    # --- Forward each position through all 8 layers, dumping per-layer hidden ---
    H = [[None] * T for _ in range(len(layers))]   # H[layer_idx][pos]
    for pos in range(T):
        cs_s, sn_s = _rope16(10_000.0, SLD_D_HEAD, pos)
        cs_g, sn_g = _rope16(1_000_000.0, GLB_ROT_DIM, pos)
        cs_s_t = torch.from_numpy(cs_s); sn_s_t = torch.from_numpy(sn_s)
        cs_g_t = torch.from_numpy(cs_g); sn_g_t = torch.from_numpy(sn_g)
        am_t = torch.from_numpy(_amask(pos))
        wm_t = torch.from_numpy(_wmask(pos))
        x = torch.from_numpy(INP22[pos])  # (1,1,D) fp16
        for li, layer in enumerate(layers):
            t = layer_types[li]
            kc, vc = states[li]
            cos_t = cs_g_t if t == "global" else cs_s_t
            sin_t = sn_g_t if t == "global" else sn_s_t
            x, k_new, v_new = layer.forward_layer(x, cos_t, sin_t, kc, vc, am_t, wm_t) \
                if hasattr(layer, "forward_layer") else \
                layer(x, cos_t, sin_t, kc, vc, am_t, wm_t)
            kc[:] = k_new
            vc[:] = v_new
            H[li][pos] = x.float().numpy().reshape(D_MODEL).copy()

    # --- Report per-layer norms + cos(pos_k, pos_0) ---
    print()
    print("  layer_idx | type | ‖h_p0‖  ‖h_p1‖  ‖h_p2‖  ‖h_p3‖  ‖h_p4‖  ‖h_p5‖")
    for li in range(len(layers)):
        gi = LAYER_RANGE[0] + li
        t = layer_types[li]
        ns = "  ".join(f"{_norm(H[li][p]):6.2f}" for p in range(T))
        print(f"  L{gi:>2} ({t[:4]})  | {ns}")

    print()
    print("  layer_idx | type | cos(p1,p0)  cos(p2,p0)  cos(p3,p0)  cos(p4,p0)  cos(p5,p0)")
    for li in range(len(layers)):
        gi = LAYER_RANGE[0] + li
        t = layer_types[li]
        cs = "  ".join(f"{_cos(H[li][0], H[li][p]):+.3f}     " for p in range(1, T))
        print(f"  L{gi:>2} ({t[:4]})  | {cs}")

    # Highlight the cliff: per-layer Δcos = cos_k(layer_i) - cos_k(layer_{i-1}) for k=1
    print()
    print("  Δcos(pos1,pos0) by layer (positive jump = cliff):")
    prev = None
    for li in range(len(layers)):
        gi = LAYER_RANGE[0] + li
        c = _cos(H[li][0], H[li][1])
        d = "" if prev is None else f"   Δ={c - prev:+.3f}"
        print(f"  L{gi:>2}  cos(p1,p0)={c:+.3f}{d}")
        prev = c


if __name__ == "__main__":
    main()
