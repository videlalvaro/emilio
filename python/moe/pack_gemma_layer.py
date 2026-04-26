"""pack_gemma_layer.py — extract one Gemma-4 layer, pack all 128 experts
into 8 groups of 16 contiguous experts, save as a single fp16 .npz
that gemma_to_ane.py consumes.

Run with the Xcode python (has safetensors + numpy):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/pack_gemma_layer.py --layer 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT_DIR   = Path("python/moe/out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL  = 2816
D_FFN    = 704     # moe_intermediate
D_DENSE  = 2112    # intermediate (dense MLP)
N_HEADS  = 16
SLD_NKV  = 8
SLD_DH   = 256
GLB_NHEADS = 16
GLB_NKV  = 2
GLB_DH   = 512
PACK_G   = 16
N_PACKS  = 8
N_EXPERTS = 128


def _layer_types() -> list[str]:
    with open(MODEL_DIR / "config.json") as f:
        c = json.load(f)
    return c["text_config"]["layer_types"]


def _load_index() -> dict[str, str]:
    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _read(idx: dict[str, str], key: str) -> np.ndarray:
    fname = idx[key]
    with safe_open(MODEL_DIR / fname, framework="pt") as f:
        # Use torch to handle bf16 → fp32; numpy can't natively decode bf16.
        t = f.get_tensor(key).to(torch.float32).contiguous().numpy()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()
    L = args.layer

    layer_types = _layer_types()
    is_global = layer_types[L] == "full_attention"
    print(f"=== pack_gemma_layer L={L}  type={layer_types[L]}  is_global={is_global} ===")

    idx = _load_index()
    base = f"model.language_model.layers.{L}"

    def R(k: str) -> np.ndarray:
        return _read(idx, f"{base}.{k}")

    # Attention
    q  = R("self_attn.q_proj.weight")
    k  = R("self_attn.k_proj.weight")
    o  = R("self_attn.o_proj.weight")
    qn = R("self_attn.q_norm.weight")
    kn = R("self_attn.k_norm.weight")
    if is_global:
        v = None  # k_eq_v=True for full_attention layers
    else:
        v = R("self_attn.v_proj.weight")
    # v_norm has with_scale=False → no weight stored; pure RMSNorm with γ=1.

    # Layernorms
    in_ln    = R("input_layernorm.weight")
    pa_ln    = R("post_attention_layernorm.weight")
    pre_ln   = R("pre_feedforward_layernorm.weight")
    post_ln  = R("post_feedforward_layernorm.weight")
    pre_ln2  = R("pre_feedforward_layernorm_2.weight")
    post_ln1 = R("post_feedforward_layernorm_1.weight")
    post_ln2 = R("post_feedforward_layernorm_2.weight")
    layer_scalar = R("layer_scalar")           # (1,)

    # Dense MLP
    mlp_g = R("mlp.gate_proj.weight")          # (2112, 2816)
    mlp_u = R("mlp.up_proj.weight")            # (2112, 2816)
    mlp_d = R("mlp.down_proj.weight")          # (2816, 2112)

    # MoE
    gate_up = R("experts.gate_up_proj")        # (128, 1408, 2816)
    down    = R("experts.down_proj")           # (128, 2816, 704)
    rprj    = R("router.proj.weight")          # (128, 2816)
    rscale  = R("router.scale")                # (2816,)
    rperexp = R("router.per_expert_scale")     # (128,)

    print(f"  shapes: q={q.shape} kv={k.shape} o={o.shape} mlp_g={mlp_g.shape}"
          f" gate_up={gate_up.shape} down={down.shape} rprj={rprj.shape}")

    # ── Pack all 128 experts into 8 × 16 ────────────────────────────────
    assert gate_up.shape[0] == N_EXPERTS
    gate_e = gate_up[:, :D_FFN, :]              # (128, 704, 2816)
    up_e   = gate_up[:, D_FFN:, :]              # (128, 704, 2816)
    down_e = down                                # (128, 2816, 704)

    gate_packs = gate_e.reshape(N_PACKS, PACK_G * D_FFN,   D_MODEL)  # (8, 11264, 2816)
    up_packs   = up_e.reshape(  N_PACKS, PACK_G * D_FFN,   D_MODEL)  # (8, 11264, 2816)
    down_packs = down_e.reshape(N_PACKS, PACK_G * D_MODEL, D_FFN)    # (8, 45056, 704)

    # ── To fp16 ──────────────────────────────────────────────────────────
    def to_fp16(x): return x.astype(np.float16)

    out_path = OUT_DIR / f"gemma_layer{L}_packed.npz"
    save_kwargs = dict(
        is_global=np.array(is_global),
        # attention
        q_proj=to_fp16(q), k_proj=to_fp16(k), o_proj=to_fp16(o),
        q_norm=to_fp16(qn), k_norm=to_fp16(kn),
    )
    if not is_global:
        save_kwargs["v_proj"] = to_fp16(v)
    np.savez(
        out_path,
        **save_kwargs,
        # layer norms
        input_ln=to_fp16(in_ln),
        post_attn_ln=to_fp16(pa_ln),
        pre_ffn_ln=to_fp16(pre_ln),
        post_ffn_ln=to_fp16(post_ln),
        pre_ffn_ln_2=to_fp16(pre_ln2),
        post_ffn_ln_1=to_fp16(post_ln1),
        post_ffn_ln_2=to_fp16(post_ln2),
        layer_scalar=to_fp16(layer_scalar),
        # dense MLP
        mlp_gate=to_fp16(mlp_g), mlp_up=to_fp16(mlp_u), mlp_down=to_fp16(mlp_d),
        # MoE packs
        gate_packs=to_fp16(gate_packs),
        up_packs=to_fp16(up_packs),
        down_packs=to_fp16(down_packs),
        # router (all 128 experts)
        router_proj=to_fp16(rprj),
        router_scale=to_fp16(rscale),
        router_per_expert_scale=to_fp16(rperexp),
    )
    sz = out_path.stat().st_size / 1e6
    print(f"  → {out_path}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
