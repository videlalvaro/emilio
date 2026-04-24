"""qwen36_clusterbase_residual_probe.py — H1 cluster-shared-base viability probe.

This is the cheapest offline gate for the H1 branch from the Qwen3.6 ANE plan:

  cluster-shared routed base + low-rank routed residuals

It does not build CoreML artifacts. Instead it asks a narrower question first:

For one real routed expert, does a cluster-shared base derived from nearby
routed experts reduce the residual burden enough to beat direct raw low-rank?

The base here is intentionally simple: the mean of the nearest routed experts by
parameter similarity. If even this cheap cluster base does not help, there is no
reason to spend CoreML time on the plain H1 path yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen36_pack_single_expert import _load_expert_weights
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR
from qwen36_sharedbase_residual_probe import (
    ANALYSIS_DTYPE,
    _as_analysis,
    _forward_expert,
    _forward_raw_lowrank,
    _forward_shared_residual_exact,
    _forward_shared_residual_lowrank,
    _metrics,
    _parse_ranks,
    _require_finite,
    _svd_cache,
    _truncated_factors,
)

PROBE_OUT_DIR = OUT_DIR / "clusterbase_residual_probe"


def _load_all_experts(layer_npz: Path) -> dict[str, np.ndarray | int]:
    with np.load(layer_npz, allow_pickle=False) as data:
        gate_up = np.asarray(data["mlp__experts__gate_up_proj"], dtype=np.float16)
        down = np.asarray(data["mlp__experts__down_proj"], dtype=np.float16)

    if gate_up.ndim != 3 or down.ndim != 3:
        raise SystemExit(
            "expected fused expert tensors with rank 3: "
            f"gate_up={gate_up.shape} down={down.shape}"
        )
    if gate_up.shape[0] != down.shape[0]:
        raise SystemExit(
            "mismatched expert counts: "
            f"gate_up={gate_up.shape[0]} down={down.shape[0]}"
        )
    if gate_up.shape[1] % 2 != 0:
        raise SystemExit(f"expected fused gate/up dimension, got {gate_up.shape}")

    d_ffn = gate_up.shape[1] // 2
    return {
        "gate": gate_up[:, :d_ffn],
        "up": gate_up[:, d_ffn:],
        "down": down,
        "d_model": int(down.shape[1]),
        "d_ffn": int(d_ffn),
        "n_experts": int(gate_up.shape[0]),
    }


def _parse_sizes(raw: str, max_size: int) -> list[int]:
    sizes = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not sizes:
        raise SystemExit("expected at least one cluster size")
    bad = [size for size in sizes if size <= 1 or size > max_size]
    if bad:
        raise SystemExit(f"invalid cluster sizes {bad}; valid range is 2..{max_size}")
    return sizes


def _flat_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = _as_analysis(left).reshape(-1)
    right64 = _as_analysis(right).reshape(-1)
    denom = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left64, right64) / denom)


def _similarity_scores(all_experts: dict[str, np.ndarray | int], expert_id: int) -> np.ndarray:
    gate = np.asarray(all_experts["gate"])
    up = np.asarray(all_experts["up"])
    down = np.asarray(all_experts["down"])
    target_gate = gate[expert_id]
    target_up = up[expert_id]
    target_down = down[expert_id]

    scores = np.empty((gate.shape[0],), dtype=ANALYSIS_DTYPE)
    for idx in range(gate.shape[0]):
        gate_cos = _flat_cosine(gate[idx], target_gate)
        up_cos = _flat_cosine(up[idx], target_up)
        down_cos = _flat_cosine(down[idx], target_down)
        scores[idx] = (gate_cos + up_cos + down_cos) / 3.0
    return _require_finite("cluster_similarity", scores)


def _cluster_base(
    all_experts: dict[str, np.ndarray | int],
    cluster_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    gate = _as_analysis(np.asarray(all_experts["gate"])[cluster_indices]).mean(axis=0)
    up = _as_analysis(np.asarray(all_experts["up"])[cluster_indices]).mean(axis=0)
    down = _as_analysis(np.asarray(all_experts["down"])[cluster_indices]).mean(axis=0)
    return {
        "gate": _require_finite("cluster_base_gate", gate),
        "up": _require_finite("cluster_base_up", up),
        "down": _require_finite("cluster_base_down", down),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranks", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--cluster-sizes", type=str, default="4,8,16")
    parser.add_argument("--out-dir", type=Path, default=PROBE_OUT_DIR)
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")
    if args.samples <= 0:
        raise SystemExit("samples must be positive")

    expert = _load_expert_weights(layer_npz, args.expert_id)
    all_experts = _load_all_experts(layer_npz)
    n_experts = int(all_experts["n_experts"])
    d_model = int(expert["d_model"])
    max_rank = min(d_model, int(expert["d_ffn"]))
    ranks = _parse_ranks(args.ranks, max_rank)
    cluster_sizes = _parse_sizes(args.cluster_sizes, n_experts)

    similarity = _similarity_scores(all_experts, args.expert_id)
    order = np.argsort(-similarity)

    rng = np.random.default_rng(args.seed)
    x = _require_finite(
        "probe_input",
        rng.standard_normal((args.samples, d_model)).astype(ANALYSIS_DTYPE, copy=False),
    )

    y_ref = _forward_expert(expert, x)
    raw_svd = {
        "gate": _svd_cache(expert["gate"]),
        "up": _svd_cache(expert["up"]),
        "down": _svd_cache(expert["down"]),
    }

    raw_by_rank: dict[int, dict[str, object]] = {}
    for rank in ranks:
        raw_factors = {}
        raw_rel_fro = {}
        for name in ("gate", "up", "down"):
            raw_first, raw_second, raw_err = _truncated_factors(raw_svd[name], rank)
            raw_factors[name] = (raw_first, raw_second)
            raw_rel_fro[name] = raw_err
        y_raw = _forward_raw_lowrank(raw_factors, x)
        raw_by_rank[rank] = {
            "metrics": _metrics(y_ref, y_raw),
            "rel_fro": raw_rel_fro,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"qwen36_L{args.layer:02d}_expert{args.expert_id:03d}_clusterbase_residual_summary.json"

    summary: dict[str, object] = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "layer_npz": str(layer_npz),
        "samples": args.samples,
        "seed": args.seed,
        "ranks": ranks,
        "cluster_sizes": cluster_sizes,
        "clusters_eval": [],
    }

    print("=== Qwen3.6 cluster-base residual probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  expert_id: {args.expert_id}")
    print(f"  samples: {args.samples}")
    print(f"  ranks: {ranks}")
    print(f"  cluster_sizes: {cluster_sizes}")

    for cluster_size in cluster_sizes:
        cluster_indices = order[:cluster_size]
        cluster_base = _cluster_base(all_experts, cluster_indices)
        residual = {
            "gate": _as_analysis(expert["gate"]) - cluster_base["gate"],
            "up": _as_analysis(expert["up"]) - cluster_base["up"],
            "down": _as_analysis(expert["down"]) - cluster_base["down"],
        }
        residual_svd = {
            "gate": _svd_cache(residual["gate"]),
            "up": _svd_cache(residual["up"]),
            "down": _svd_cache(residual["down"]),
        }

        y_base = _forward_expert(cluster_base, x)
        y_exact = _forward_shared_residual_exact(cluster_base, residual, x)

        cluster_entry: dict[str, object] = {
            "cluster_size": cluster_size,
            "cluster_indices": [int(idx) for idx in cluster_indices.tolist()],
            "cluster_similarity": {
                "target_mean": float(np.mean(similarity[cluster_indices])),
                "target_min": float(np.min(similarity[cluster_indices])),
                "target_max": float(np.max(similarity[cluster_indices])),
            },
            "base_only": _metrics(y_ref, y_base),
            "base_plus_full_residual": _metrics(y_ref, y_exact),
            "matrix_norms": {
                name: {
                    "expert_fro": float(raw_svd[name]["fro"]),
                    "residual_fro": float(residual_svd[name]["fro"]),
                    "residual_vs_expert": float(residual_svd[name]["fro"]) / max(float(raw_svd[name]["fro"]), 1e-12),
                }
                for name in ("gate", "up", "down")
            },
            "ranks_eval": [],
        }

        print(
            f"  cluster {cluster_size:>2}: idx={cluster_entry['cluster_indices']} "
            f"sim(mean={cluster_entry['cluster_similarity']['target_mean']:.6f}, "
            f"min={cluster_entry['cluster_similarity']['target_min']:.6f})"
        )
        print(
            "    base-only output: "
            f"min={cluster_entry['base_only']['cos_min']:.6f} "
            f"mean={cluster_entry['base_only']['cos_mean']:.6f}"
        )
        print(
            "    base+full residual sanity: "
            f"min={cluster_entry['base_plus_full_residual']['cos_min']:.6f} "
            f"mean={cluster_entry['base_plus_full_residual']['cos_mean']:.6f}"
        )
        for name, stats in cluster_entry["matrix_norms"].items():
            print(
                f"    residual norm ratio {name}: "
                f"{stats['residual_vs_expert']:.6f} "
                f"(residual_fro={stats['residual_fro']:.3f} expert_fro={stats['expert_fro']:.3f})"
            )

        for rank in ranks:
            residual_factors = {}
            residual_rel_fro = {}
            for name in ("gate", "up", "down"):
                res_first, res_second, res_err = _truncated_factors(residual_svd[name], rank)
                residual_factors[name] = (res_first, res_second)
                residual_rel_fro[name] = res_err

            y_cluster_res = _forward_shared_residual_lowrank(cluster_base, residual_factors, x)
            cluster_metrics = _metrics(y_ref, y_cluster_res)
            raw_metrics = dict(raw_by_rank[rank]["metrics"])
            entry = {
                "rank": rank,
                "raw_lowrank": {
                    **raw_metrics,
                    "gate_rel_fro": float(raw_by_rank[rank]["rel_fro"]["gate"]),
                    "up_rel_fro": float(raw_by_rank[rank]["rel_fro"]["up"]),
                    "down_rel_fro": float(raw_by_rank[rank]["rel_fro"]["down"]),
                },
                "clusterbase_residual": {
                    **cluster_metrics,
                    "gate_rel_fro": residual_rel_fro["gate"],
                    "up_rel_fro": residual_rel_fro["up"],
                    "down_rel_fro": residual_rel_fro["down"],
                },
            }
            cluster_entry["ranks_eval"].append(entry)
            print(
                f"    rank {rank:>3}: raw(mean={raw_metrics['cos_mean']:.6f}, min={raw_metrics['cos_min']:.6f}) "
                f"vs cluster+res(mean={cluster_metrics['cos_mean']:.6f}, min={cluster_metrics['cos_min']:.6f})"
            )

        summary["clusters_eval"].append(cluster_entry)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()