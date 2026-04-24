"""Pairwise expert weight similarity within each MoE layer.

Streams one layer at a time to keep memory bounded (DeepSeek-V2-Lite has
64 experts * 3 projections per layer ~= 35 MB / layer in float32; the previous
all-at-once version OOM'd at ~57 GB).

Reads weights directly from safetensors - does not need to instantiate the model.

Usage:
    python -m python.moe.expert_similarity \
        --model models/deepseek-v2-lite-chat \
        --out python/moe/out/similarity
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


PROJS = ("gate_proj", "up_proj", "down_proj")


def _norm_rows_inplace(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    np.divide(x, n, out=x)
    return x


def _stats(S: np.ndarray, n: int) -> dict:
    iu = np.triu_indices(n, k=1)
    offdiag = S[iu]
    evals = np.sort(np.linalg.eigvalsh(S.astype(np.float64)))[::-1]
    return {
        "off_mean": float(offdiag.mean()),
        "off_std":  float(offdiag.std()),
        "off_max":  float(offdiag.max()),
        "off_min":  float(offdiag.min()),
        "off_p95":  float(np.quantile(offdiag, 0.95)),
        "top_eig_frac": float(evals[0] / evals.sum()),
        "evals_top5":   evals[:5].astype(np.float32).tolist(),
    }


def load_layer_experts(model_dir: Path, weight_map: dict, layer: int,
                       n_experts: int) -> dict[str, np.ndarray]:
    """Returns {proj: [n_experts, flat_dim] float32}."""
    by_shard: dict[str, list[tuple[int, str, str]]] = {}
    for proj in PROJS:
        for e in range(n_experts):
            k = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
            by_shard.setdefault(weight_map[k], []).append((e, proj, k))

    out: dict[str, list[np.ndarray | None]] = {p: [None] * n_experts for p in PROJS}
    for shard, items in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
            for e, proj, k in items:
                t = f.get_tensor(k).to(torch.float32).numpy().reshape(-1)
                out[proj][e] = t
    return {p: np.stack(v) for p, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out",   required=True,
                   help="output stem; writes <out>.json (and <out>.npz if --save-matrices)")
    p.add_argument("--save-matrices", action="store_true")
    args = p.parse_args()

    model_dir = Path(args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((model_dir / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    n_routed = cfg["n_routed_experts"]
    first_dense = cfg.get("first_k_dense_replace", 0)
    moe_layers = list(range(first_dense, n_layers))
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    print(f"[cfg] {len(moe_layers)} MoE layers, {n_routed} routed experts")

    summary: dict[str, list[dict]] = {p: [] for p in PROJS}
    summary["concat"] = []
    matrices: dict[str, list[np.ndarray]] = {p: [] for p in PROJS}
    matrices["concat"] = []

    for layer in moe_layers:
        ex = load_layer_experts(model_dir, weight_map, layer, n_routed)
        # Per-proj similarity
        for proj in PROJS:
            X = ex[proj]
            _norm_rows_inplace(X)
            S = (X @ X.T).astype(np.float32)
            stats = _stats(S, n_routed); stats["layer"] = layer
            summary[proj].append(stats)
            if args.save_matrices:
                matrices[proj].append(S.astype(np.float16))
        del ex; gc.collect()
        # Concatenated triple - reload (we destroyed normalization above)
        ex2 = load_layer_experts(model_dir, weight_map, layer, n_routed)
        Xcat = np.concatenate([ex2[p] for p in PROJS], axis=1)
        del ex2; gc.collect()
        _norm_rows_inplace(Xcat)
        Scat = (Xcat @ Xcat.T).astype(np.float32)
        stats = _stats(Scat, n_routed); stats["layer"] = layer
        summary["concat"].append(stats)
        if args.save_matrices:
            matrices["concat"].append(Scat.astype(np.float16))
        del Xcat, Scat; gc.collect()
        print(f"  layer {layer:2d}  "
              + "  ".join(f"{p[:4]}={summary[p][-1]['off_mean']:+.3f}"
                          for p in (*PROJS, "concat")))

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] {json_path}")
    if args.save_matrices:
        npz_path = out_path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            **{f"{p}_S": np.stack(matrices[p]) for p in (*PROJS, "concat")},
            moe_layers=np.array(moe_layers, dtype=np.int32),
        )
        print(f"[save] {npz_path}")

    print("\n=== summary (mean off-diagonal cosine across MoE layers) ===")
    for proj in (*PROJS, "concat"):
        m   = float(np.mean([d["off_mean"]    for d in summary[proj]]))
        mx  = float(np.max ([d["off_max"]     for d in summary[proj]]))
        eig = float(np.mean([d["top_eig_frac"] for d in summary[proj]]))
        print(f"  {proj:>10s}: mean={m:+.3f}  max={mx:+.3f}  "
              f"top_eig_frac={eig:.3f}")
    print("\nInterpretation:")
    print("  mean ~ 0.0  => experts uncorrelated (codebook sharing won't help)")
    print("  mean > 0.3  => meaningful redundancy (codebook sharing viable)")
    print("  top_eig_frac > 0.5 => effectively low-rank expert bank")


if __name__ == "__main__":
    main()
