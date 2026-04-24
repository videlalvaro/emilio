"""BWT-vs-LZ77 entropy of the MoE expert bank.

The user's hypothesis: post-quantization weights are near-random per-tensor, but
the *cross-expert* dimension (64 structurally similar experts per layer in
DeepSeek-V2-Lite, 384 in Kimi K2.6) may have long-range repetition that BWT can
exploit beyond what LZ77 sees.

Methodology (no extra deps - uses stdlib bz2/zlib):
  - bz2 internally = BWT + MTF + RLE + Huffman. So len(bz2.compress(x))
    is a faithful upper bound on the BWT-RLE entropy of x.
  - zlib (deflate) = LZ77 + Huffman, no BWT. So the gap
        bz2_size / zlib_size
    quantifies the BWT-specific signal.
  - If bz2 < zlib by >5% on the expert stream, BWT is exploiting structure
    that LZ77 misses, and a real FM-index over the expert bank is worth
    building. If they're equal (or zlib wins, which happens on incompressible
    data), the BWT-on-weights idea is dead.

We test three orderings of the expert stream:
  1. natural    - experts in original index order (0,1,2,...,63) per layer
  2. random     - shuffled (control: kills any cross-expert structure)
  3. greedy_sim - greedy nearest-neighbor sort by cosine similarity
                  (uses similarity matrix from expert_similarity.py if
                  available; else computed on the fly)

Per-tensor zlib is also reported as the "no cross-expert structure" baseline.

Quantization: simple symmetric int4 with group_size=32 along the input dim,
matching DeepSeek/Kimi's native int4 scheme (config.json group_0).

Usage:
    python -m python.moe.expert_bwt_entropy \
        --model models/deepseek-v2-lite-chat \
        --layers 1,5,13,26 \
        --proj gate_proj \
        --out python/moe/out/bwt_entropy.json
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
    """Symmetric int4, per-group along the last axis. Returns int8 in [-8, 7]
    (one int4 per byte; we don't bother packing - the bz2/zlib comparison is
    invariant to a constant 2x bit width and packing only confuses things)."""
    assert W.ndim == 2, W.shape
    rows, cols = W.shape
    assert cols % group_size == 0, (cols, group_size)
    Wg = W.reshape(rows, cols // group_size, group_size)
    amax = np.abs(Wg).max(axis=-1, keepdims=True).clip(min=1e-12)
    scale = amax / 7.0
    q = np.round(Wg / scale).clip(-8, 7).astype(np.int8)
    return q.reshape(rows, cols)


def load_experts(model_dir: Path, layer: int, proj: str, n_experts: int) -> np.ndarray:
    """Returns [n_experts, rows, cols] float32 for the given (layer, proj)."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    # Group by shard.
    by_shard: dict[str, list[tuple[int, str]]] = {}
    for e in range(n_experts):
        k = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
        by_shard.setdefault(wm[k], []).append((e, k))
    out: list[np.ndarray | None] = [None] * n_experts
    for shard, items in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
            for e, k in items:
                out[e] = f.get_tensor(k).to(torch.float32).numpy()
    return np.stack(out)  # [E, R, C]


def greedy_similarity_order(W: np.ndarray) -> list[int]:
    """W: [E, R, C] float. Greedy NN sort by row-mean cosine similarity."""
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
        "bwt_gain_pct": 100.0 * (zl_sz - bz_sz) / zl_sz,  # +ve => BWT helps
        "bz2_sec": bz_t,
        "zlib_sec": zl_t,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="1,5,13,26",
                   help="comma-separated MoE layer ids")
    p.add_argument("--proj", default="gate_proj",
                   choices=["gate_proj", "up_proj", "down_proj"])
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    model_dir = Path(args.model)
    cfg = json.loads((model_dir / "config.json").read_text())
    n_routed = cfg["n_routed_experts"]
    layers = [int(x) for x in args.layers.split(",")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {"model": str(model_dir), "n_experts": n_routed,
               "proj": args.proj, "group_size": args.group_size, "layers": {}}

    for layer in layers:
        print(f"\n=== layer {layer} :: {args.proj} ===")
        W = load_experts(model_dir, layer, args.proj, n_routed)  # [E, R, C]
        E, R, C = W.shape
        print(f"  shape: experts={E}  rows={R}  cols={C}")
        # Quantize each expert (per-row groups of `group_size` along col axis).
        Q = np.stack([quant_int4_g32(W[e], args.group_size) for e in range(E)])
        # Q is int8 in [-8,7]. Convert to unsigned bytes for streaming.
        Qb = (Q + 8).astype(np.uint8)  # [E, R, C], values in 0..15

        # 1. natural order
        natural = Qb.tobytes()
        # 2. random order
        perm = rng.permutation(E)
        randomized = Qb[perm].tobytes()
        # 3. greedy similarity order
        order = greedy_similarity_order(W)
        sim_sorted = Qb[order].tobytes()
        # 4. per-tensor zlib baseline = sum of independent compressions
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
    print("\nReading the result:")
    print("  bwt_gain_pct > 5      => BWT exploits structure LZ77 misses (FM-index worth building)")
    print("  cross_expert_*_pct > 5 => stream-as-one-blob beats per-tensor (codebook sharing real)")
    print("  greedy_sim < random   => expert ordering matters (sort before storing)")


if __name__ == "__main__":
    main()
