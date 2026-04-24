"""BWT-vs-LZ77 entropy of the MoE expert bank for Gemma 4.

Same methodology as expert_bwt_entropy.py but adapted to Gemma 4's stacked
expert tensor layout (`experts.gate_up_proj`, `experts.down_proj`).

Usage:
    python -m python.moe.expert_bwt_entropy_gemma \
        --model models/gemma-4-26b-a4b \
        --layers 0,1,4,15,29 \
        --proj gate_proj \
        --out python/moe/out/gemma_bwt_entropy_gate.json
"""

from __future__ import annotations

import argparse
import bz2
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def quant_int4_g32(W: np.ndarray, group_size: int = 32) -> np.ndarray:
    assert W.ndim == 2, W.shape
    rows, cols = W.shape
    pad = (-cols) % group_size
    if pad:
        W = np.pad(W, ((0, 0), (0, pad)))
        cols += pad
    Wg = W.reshape(rows, cols // group_size, group_size)
    amax = np.abs(Wg).max(axis=-1, keepdims=True).clip(min=1e-12)
    scale = amax / 7.0
    q = np.round(Wg / scale).clip(-8, 7).astype(np.int8)
    return q.reshape(rows, cols)


def load_experts_gemma(model_dir: Path, layer: int, proj: str,
                       n_experts: int, intermediate: int) -> np.ndarray:
    """Returns [n_experts, rows, cols] float32 for the given (layer, proj)."""
    base = f"model.language_model.layers.{layer}"
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    if proj in ("gate_proj", "up_proj"):
        k = f"{base}.experts.gate_up_proj"
        with safe_open(model_dir / wm[k], framework="pt", device="cpu") as f:
            t = f.get_tensor(k).to(torch.float32).numpy()  # [E, 2I, H]
        if proj == "gate_proj":
            return t[:, :intermediate, :]
        else:
            return t[:, intermediate:, :]
    else:  # down_proj
        k = f"{base}.experts.down_proj"
        with safe_open(model_dir / wm[k], framework="pt", device="cpu") as f:
            t = f.get_tensor(k).to(torch.float32).numpy()  # [E, H, I]
        return t


def greedy_similarity_order(W: np.ndarray) -> list[int]:
    E = W.shape[0]
    flat = W.reshape(E, -1).astype(np.float32)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True).clip(min=1e-12)
    S = flat @ flat.T
    np.fill_diagonal(S, -np.inf)
    order = [0]
    used = {0}
    while len(order) < E:
        last = order[-1]
        cand = S[last].copy()
        for u in used:
            cand[u] = -np.inf
        nxt = int(cand.argmax())
        order.append(nxt); used.add(nxt)
    return order


def measure(stream: bytes) -> dict:
    raw = len(stream)
    t0 = time.time()
    bz_sz = len(bz2.compress(stream, compresslevel=9))
    bz_t = time.time() - t0
    t0 = time.time()
    zl_sz = len(zlib.compress(stream, 9))
    zl_t = time.time() - t0
    return {
        "raw_bytes": raw,
        "bz2_bytes": bz_sz,
        "zlib_bytes": zl_sz,
        "bz2_ratio": bz_sz / raw,
        "zlib_ratio": zl_sz / raw,
        "bwt_gain_pct": 100.0 * (zl_sz - bz_sz) / zl_sz,
        "bz2_sec": bz_t,
        "zlib_sec": zl_t,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="0,1,4,15,29")
    p.add_argument("--proj", default="gate_proj",
                   choices=["gate_proj", "up_proj", "down_proj"])
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    model_dir = Path(args.model)
    cfg = json.loads((model_dir / "config.json").read_text())
    text_cfg = cfg["text_config"]
    n_routed = text_cfg["num_experts"]
    intermediate = text_cfg["moe_intermediate_size"]
    layers = [int(x) for x in args.layers.split(",")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {"model": str(model_dir), "n_experts": n_routed,
               "proj": args.proj, "group_size": args.group_size, "layers": {}}

    for layer in layers:
        print(f"\n=== layer {layer} :: {args.proj} ===")
        W = load_experts_gemma(model_dir, layer, args.proj, n_routed, intermediate)
        E, R, C = W.shape
        print(f"  shape: experts={E}  rows={R}  cols={C}")
        Q = np.stack([quant_int4_g32(W[e], args.group_size) for e in range(E)])
        Qb = (Q + 8).astype(np.uint8)

        natural = Qb.tobytes()
        perm = rng.permutation(E)
        randomized = Qb[perm].tobytes()
        order = greedy_similarity_order(W)
        sim_sorted = Qb[order].tobytes()
        per_tensor_zlib = sum(len(zlib.compress(Qb[e].tobytes(), 9)) for e in range(E))
        per_tensor_bz2 = sum(len(bz2.compress(Qb[e].tobytes(), 9)) for e in range(E))

        m_nat = measure(natural)
        m_rnd = measure(randomized)
        m_sim = measure(sim_sorted)

        layer_res = {
            "shape": [E, R, C],
            "raw_bytes_total": len(natural),
            "natural": m_nat,
            "random": m_rnd,
            "greedy_sim": m_sim,
            "per_tensor_zlib_bytes": per_tensor_zlib,
            "per_tensor_bz2_bytes": per_tensor_bz2,
            "cross_expert_zlib_gain_pct":
                100.0 * (per_tensor_zlib - m_nat["zlib_bytes"]) / per_tensor_zlib,
            "cross_expert_bz2_gain_pct":
                100.0 * (per_tensor_bz2 - m_nat["bz2_bytes"]) / per_tensor_bz2,
            "sim_order": order,
        }
        results["layers"][str(layer)] = layer_res

        print(f"  raw bytes (Q4 unpacked):     {len(natural):>12,}")
        print(f"  per-tensor zlib (baseline):  {per_tensor_zlib:>12,}  "
              f"(ratio {per_tensor_zlib / len(natural):.3f})")
        print(f"  per-tensor bz2:              {per_tensor_bz2:>12,}  "
              f"(ratio {per_tensor_bz2 / len(natural):.3f})")
        print(f"  natural-order zlib:          {m_nat['zlib_bytes']:>12,}  "
              f"(ratio {m_nat['zlib_ratio']:.3f})")
        print(f"  natural-order bz2:           {m_nat['bz2_bytes']:>12,}  "
              f"(ratio {m_nat['bz2_ratio']:.3f})  "
              f"BWT vs zlib: {m_nat['bwt_gain_pct']:+.1f}%")
        print(f"  random-order  bz2:           {m_rnd['bz2_bytes']:>12,}  "
              f"(ratio {m_rnd['bz2_ratio']:.3f})")
        print(f"  greedy-sim    bz2:           {m_sim['bz2_bytes']:>12,}  "
              f"(ratio {m_sim['bz2_ratio']:.3f})  "
              f"vs random: {100*(m_rnd['bz2_bytes']-m_sim['bz2_bytes'])/m_rnd['bz2_bytes']:+.1f}%")
        print(f"  cross-expert gain (bz2 vs per-tensor bz2): "
              f"{layer_res['cross_expert_bz2_gain_pct']:+.1f}%")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
