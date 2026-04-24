"""gemma_quant_ablation.py — TurboQuant+ inspired diagnostic.

Tests which weight subsets are responsible for INT4 quality loss. Loads N
Gemma layers in PyTorch fp16 (the reference), then for each ablation applies
INT4 per-block-8 symmetric quant ONLY to selected weight tensors, runs one
forward, measures cos(hidden) vs the unquantized reference.

Hypothesis (TurboQuant+ findings on same model family):
  - K is the dominant failure axis
  - Boundary layers are disproportionately sensitive
  - V can be compressed harder than K
  - Globals may need higher precision than slidings (depth-boundary effect)

Usage:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_quant_ablation.py --n-layers 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaMixedStackWrap, _load_layer_weights,
    _layer_types_from_config,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM,
)

OUT_DIR = Path("python/moe/out")
MAX_CTX = 1024
SEED = 0xA1E
INT4_BLOCK = 8


def quantize_int4_per_block(w: torch.Tensor, block: int = INT4_BLOCK) -> torch.Tensor:
    """Symmetric per-block INT4 (4-bit signed, range [-8, 7]).

    Groups along the LAST dim in chunks of `block`. Returns a fp16 tensor with
    quant-then-dequant applied (lossy round-trip). Mirrors coremltools'
    cto.linear_quantize_weights(dtype='int4', granularity='per_block',
    block_size=8).
    """
    orig_shape = w.shape
    last = w.shape[-1]
    if last % block != 0:
        return w  # skip non-divisible (rare; alignment guard)
    w_f = w.float()
    grouped = w_f.reshape(*orig_shape[:-1], last // block, block)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 7.0).clamp(min=1e-8)
    q = torch.round(grouped / scale).clamp(-8, 7)
    deq = (q * scale).reshape(orig_shape).to(w.dtype)
    return deq


# Per-projection groups inside a single layer.
PROJ_NAMES = {
    "Q":   ["attn.q_proj.weight"],
    "K":   ["attn.k_proj.weight"],
    "V":   ["attn.v_proj.weight"],   # absent on globals
    "O":   ["attn.o_proj.weight"],
    "FFN": [
        "mlp_dense.gate.weight", "mlp_dense.up.weight", "mlp_dense.down.weight",
        # packed experts and router intentionally excluded — they're already
        # the lion's share of size, ablating them dwarfs attention signal
    ],
    "EXP": [
        "packs.0.gate.weight", "packs.0.up.weight", "packs.0.down.weight",
        "packs.1.gate.weight", "packs.1.up.weight", "packs.1.down.weight",
        "packs.2.gate.weight", "packs.2.up.weight", "packs.2.down.weight",
        "packs.3.gate.weight", "packs.3.up.weight", "packs.3.down.weight",
    ],
}


def _apply_quant(model: GemmaMixedStackWrap, ablation: dict) -> None:
    """Apply INT4 quant in-place to selected weights.

    ablation: {layer_idx: set(of group names from PROJ_NAMES)}
    """
    for li, layer in enumerate(model.layers):
        groups = ablation.get(li, set())
        if not groups:
            continue
        for g in groups:
            for pname in PROJ_NAMES[g]:
                # Resolve dotted name
                parts = pname.split(".")
                obj = layer
                ok = True
                for part in parts[:-1]:
                    if part.isdigit():
                        try:
                            obj = obj[int(part)]
                        except (IndexError, KeyError):
                            ok = False; break
                    else:
                        if not hasattr(obj, part):
                            ok = False; break
                        obj = getattr(obj, part)
                if not ok:
                    continue
                attr = parts[-1]
                if not hasattr(obj, attr):
                    continue
                p: torch.nn.Parameter = getattr(obj, attr)
                if p is None:
                    continue
                with torch.no_grad():
                    p.data.copy_(quantize_int4_per_block(p.data, INT4_BLOCK))


def _real_rope(theta: float, dh: int, pos: int):
    half = dh // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half) / half))
    freqs = pos * inv_freq
    full = np.concatenate([freqs, freqs])
    return np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _build(N: int, layer_types: list[str], npz_paths: list[Path]) -> GemmaMixedStackWrap:
    m = GemmaMixedStackWrap(MAX_CTX, layer_types)
    m.half().eval()
    for i, p in enumerate(npz_paths):
        _load_layer_weights(m.layers[i], p)
    return m


def _fwd(m: GemmaMixedStackWrap, inputs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        for i in range(m.n_layers):
            getattr(m, f"k_cache_{i}").zero_()
            getattr(m, f"v_cache_{i}").zero_()
        h, k, v = m(**inputs)
    return (h.float().numpy().reshape(D_MODEL),
            k.float().numpy(), v.float().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, required=True)
    args = ap.parse_args()
    N = args.n_layers
    layer_types = _layer_types_from_config(N)
    npz_paths = [OUT_DIR / f"gemma_layer{i}_packed.npz" for i in range(N)]
    for p in npz_paths:
        if not p.exists():
            print(f"FATAL missing pack: {p}", file=sys.stderr); sys.exit(2)

    print(f"=== mixed{N} INT4 ablation (block_size=8) ===")
    print(f"  layer_types: {layer_types}")
    globals_idx = [i for i, t in enumerate(layer_types) if t == "global"]
    boundary_idx = sorted({0, 1, N-2, N-1})
    print(f"  globals at: {globals_idx}")
    print(f"  boundary set: {boundary_idx}")

    rng = np.random.default_rng(SEED)
    x_np = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)
    cos_s_f, sin_s_f = _real_rope(theta=10000.0, dh=SLD_D_HEAD, pos=0)
    cos_g_f, sin_g_f = _real_rope(theta=1_000_000.0, dh=GLB_ROT_DIM, pos=0)
    attn_mask_np = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    attn_mask_np[..., 0] = 0.0
    wmask_np = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    wmask_np[0, 0, 0, 0] = 1.0
    inputs = {
        "x": torch.from_numpy(x_np),
        "cos_s": torch.from_numpy(cos_s_f.astype(np.float16).reshape(1,1,SLD_D_HEAD)),
        "sin_s": torch.from_numpy(sin_s_f.astype(np.float16).reshape(1,1,SLD_D_HEAD)),
        "cos_g": torch.from_numpy(cos_g_f.astype(np.float16).reshape(1,1,GLB_ROT_DIM)),
        "sin_g": torch.from_numpy(sin_g_f.astype(np.float16).reshape(1,1,GLB_ROT_DIM)),
        "attn_mask": torch.from_numpy(attn_mask_np),
        "kv_write_mask": torch.from_numpy(wmask_np),
    }

    print("\n  >> baseline fp16 reference forward...")
    t0 = time.perf_counter()
    ref = _build(N, layer_types, npz_paths)
    ref_h, ref_k, ref_v = _fwd(ref, inputs)
    print(f"     done in {time.perf_counter()-t0:.1f}s, ‖hidden‖={np.linalg.norm(ref_h):.3f}")
    del ref

    all_idx = list(range(N))
    interior_idx = [i for i in all_idx if i not in boundary_idx]
    sliding_idx = [i for i, t in enumerate(layer_types) if t == "sliding"]

    # Ablation matrix: each entry rebuilds the model, applies the quant,
    # forwards, reports cos(h)/cos(K)/cos(V).
    ABLATIONS = [
        ("ALL all-projs",      {i: {"Q","K","V","O","FFN","EXP"} for i in all_idx}),
        # Per-projection only (ALL layers)
        ("only K (all L)",     {i: {"K"} for i in all_idx}),
        ("only V (all L)",     {i: {"V"} for i in all_idx}),
        ("only Q (all L)",     {i: {"Q"} for i in all_idx}),
        ("only O (all L)",     {i: {"O"} for i in all_idx}),
        ("only FFN dense (all)", {i: {"FFN"} for i in all_idx}),
        ("only EXP packs (all)", {i: {"EXP"} for i in all_idx}),
        # Asymmetric K/V like TQ+: skip K, quant V everywhere
        ("ALL except K",       {i: {"Q","V","O","FFN","EXP"} for i in all_idx}),
        # Boundary protection
        ("ALL except boundary 0,1,N-2,N-1",
            {i: {"Q","K","V","O","FFN","EXP"} for i in interior_idx}),
        # Global protection
        ("ALL except globals (kept fp16)",
            {i: {"Q","K","V","O","FFN","EXP"} for i in sliding_idx}),
        # Both
        ("ALL except globals + boundary",
            {i: {"Q","K","V","O","FFN","EXP"} for i in interior_idx if i not in globals_idx}),
        # Only K on globals (test if global K is the failure)
        ("only K on globals",  {i: {"K"} for i in globals_idx}),
        # Only K on boundaries
        ("only K on boundaries", {i: {"K"} for i in boundary_idx}),
    ]

    print(f"\n  {'ablation':<42} {'cos_h':>8} {'cos_K':>8} {'cos_V':>8} {'rmse_h':>8}")
    print(f"  {'-'*42:<42} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
    for name, ab in ABLATIONS:
        m = _build(N, layer_types, npz_paths)
        _apply_quant(m, ab)
        h, k, v = _fwd(m, inputs)
        ch = _cos(ref_h, h)
        ck = _cos(ref_k, k)
        cv = _cos(ref_v, v)
        rmse = float(np.sqrt(np.mean((ref_h - h) ** 2)))
        print(f"  {name:<42} {ch:>8.4f} {ck:>8.4f} {cv:>8.4f} {rmse:>8.4f}")
        del m


if __name__ == "__main__":
    main()
