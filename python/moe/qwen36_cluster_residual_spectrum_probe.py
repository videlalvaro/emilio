"""qwen36_cluster_residual_spectrum_probe.py — offline gate for a new decomposition family.

This does not build CoreML artifacts. It measures how compressible the raw expert,
cluster base, and cluster residual are for one real Qwen routed expert across
several cluster sizes.

Goal: decide whether a compute-in-compressed-space family is numerically
plausible before spending more ANE conversion time on it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen36_clusterbase_residual_probe import (
    _cluster_base,
    _load_all_experts,
    _parse_sizes,
    _similarity_scores,
)
from qwen36_pack_single_expert import _load_expert_weights
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR

PROBE_OUT_DIR = OUT_DIR / "cluster_residual_spectrum_probe"


def _as64(weight: np.ndarray) -> np.ndarray:
    arr = np.asarray(weight, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise SystemExit("non-finite values in analysis tensor")
    return arr


def _spectrum_metrics(weight: np.ndarray) -> dict[str, float | int]:
    weight64 = _as64(weight)
    singular = np.linalg.svd(weight64, full_matrices=False, compute_uv=False)
    if singular.size == 0:
        raise SystemExit("empty singular spectrum")
    energy = singular * singular
    total_energy = float(np.sum(energy))
    if total_energy <= 0.0:
        return {
            "stable_rank": 0.0,
            "rank_90": 0,
            "rank_95": 0,
            "rank_99": 0,
            "sigma_max": 0.0,
            "fro": 0.0,
        }

    cumulative = np.cumsum(energy) / total_energy

    def _rank_for(target: float) -> int:
        return int(np.searchsorted(cumulative, target, side="left") + 1)

    sigma_max = float(singular[0])
    fro = float(np.sqrt(total_energy))
    stable_rank = total_energy / max(sigma_max * sigma_max, 1e-12)
    return {
        "stable_rank": float(stable_rank),
        "rank_90": _rank_for(0.90),
        "rank_95": _rank_for(0.95),
        "rank_99": _rank_for(0.99),
        "sigma_max": sigma_max,
        "fro": fro,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--cluster-sizes", type=str, default="4,8,16,24,32")
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROBE_OUT_DIR)
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")

    expert = _load_expert_weights(layer_npz, args.expert_id)
    all_experts = _load_all_experts(layer_npz)
    cluster_sizes = _parse_sizes(args.cluster_sizes, int(all_experts["n_experts"]))
    similarity = _similarity_scores(all_experts, args.expert_id)
    order = np.argsort(-similarity)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / (
        f"qwen36_L{args.layer:02d}_expert{args.expert_id:03d}_cluster_residual_spectrum_summary.json"
    )

    summary: dict[str, object] = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "layer_npz": str(layer_npz),
        "cluster_sizes": cluster_sizes,
        "raw": {},
        "clusters": {},
    }

    print("=== Qwen3.6 cluster residual spectrum probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  expert_id: {args.expert_id}")
    print(f"  cluster_sizes: {cluster_sizes}")

    raw_metrics = {
        name: _spectrum_metrics(np.asarray(expert[name], dtype=np.float64))
        for name in ("gate", "up", "down")
    }
    summary["raw"] = raw_metrics
    print("  raw stable ranks:")
    for name in ("gate", "up", "down"):
        metrics = raw_metrics[name]
        print(
            f"    {name}: stable={metrics['stable_rank']:.1f} "
            f"r90={metrics['rank_90']} r95={metrics['rank_95']} r99={metrics['rank_99']}"
        )

    cluster_results: dict[str, object] = {}
    for cluster_size in cluster_sizes:
        cluster_indices = order[:cluster_size]
        base = _cluster_base(all_experts, cluster_indices)
        residual = {
            name: _as64(np.asarray(expert[name], dtype=np.float64)) - _as64(np.asarray(base[name], dtype=np.float64))
            for name in ("gate", "up", "down")
        }
        base_metrics = {name: _spectrum_metrics(base[name]) for name in ("gate", "up", "down")}
        residual_metrics = {name: _spectrum_metrics(residual[name]) for name in ("gate", "up", "down")}
        cluster_results[str(cluster_size)] = {
            "cluster_indices": [int(idx) for idx in cluster_indices.tolist()],
            "similarity_mean": float(np.mean(similarity[cluster_indices])),
            "similarity_min": float(np.min(similarity[cluster_indices])),
            "base": base_metrics,
            "residual": residual_metrics,
        }
        print(f"  cluster {cluster_size}:")
        for name in ("gate", "up", "down"):
            base_m = base_metrics[name]
            res_m = residual_metrics[name]
            print(
                f"    {name}: base stable={base_m['stable_rank']:.1f} r95={base_m['rank_95']} | "
                f"residual stable={res_m['stable_rank']:.1f} r95={res_m['rank_95']}"
            )

    summary["clusters"] = cluster_results
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()