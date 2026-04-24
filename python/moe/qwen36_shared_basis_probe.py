"""qwen36_shared_basis_probe.py — shared-basis routed-op viability probe.

This is the next H1 offline gate after the packed-base factor-bank artifact
showed that enlarging only the shared base is insufficient. The new hypothesis
is that the routed side itself must be rewritten into a small number of large
shared basis operators across a tight expert cluster.

For one routed cluster, this probe factors each expert matrix family
(`gate`, `up`, `down`) across the expert axis:

  W_e ~= sum_l coeff[e, l] * basis[l]

This yields a small number of large basis linears plus cheap coefficient mixes,
which is the shape the next CoreML artifact would need.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen36_clusterbase_residual_probe import (
    _load_all_experts,
    _parse_sizes,
    _similarity_scores,
)
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR
from qwen36_sharedbase_residual_probe import (
    ANALYSIS_DTYPE,
    _as_analysis,
    _forward_expert,
    _metrics,
    _require_finite,
    _safe_mm,
    _sigmoid_linear_unit,
)

PROBE_OUT_DIR = OUT_DIR / "shared_basis_probe"


def _expert_from_all(all_experts: dict[str, np.ndarray | int], expert_id: int) -> dict[str, np.ndarray]:
    n_experts = int(all_experts["n_experts"])
    if expert_id < 0 or expert_id >= n_experts:
        raise SystemExit(f"expert_id {expert_id} out of range 0..{n_experts - 1}")
    return {
        "gate": np.asarray(all_experts["gate"])[expert_id],
        "up": np.asarray(all_experts["up"])[expert_id],
        "down": np.asarray(all_experts["down"])[expert_id],
        "d_model": np.array(int(all_experts["d_model"]), dtype=np.int32),
        "d_ffn": np.array(int(all_experts["d_ffn"]), dtype=np.int32),
    }


def _parse_basis_counts(raw: str, max_count: int) -> list[int]:
    counts = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not counts:
        raise SystemExit("expected at least one basis count")
    bad = [count for count in counts if count <= 0 or count > max_count]
    if bad:
        raise SystemExit(f"invalid basis counts {bad}; valid range is 1..{max_count}")
    return counts


def _factor_shared_basis(weight_bank: np.ndarray, basis_count: int) -> dict[str, np.ndarray | float]:
    bank64 = _require_finite("basis_weight_bank", _as_analysis(weight_bank))
    n_experts, out_dim, in_dim = bank64.shape
    flat = bank64.reshape(n_experts, out_dim * in_dim)
    u, s, vt = np.linalg.svd(flat, full_matrices=False)
    coeff = _as_analysis(u[:, :basis_count] * s[:basis_count][None, :])
    basis_flat = _as_analysis(vt[:basis_count, :])
    recon = _safe_mm("basis_reconstruction", coeff, basis_flat)
    rel_fro = float(np.linalg.norm(recon - flat) / max(np.linalg.norm(flat), 1e-12))
    return {
        "coeff": coeff,
        "basis": basis_flat.reshape(basis_count, out_dim, in_dim),
        "rel_fro": rel_fro,
    }


def _forward_basis_bank(x: np.ndarray, basis: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    batch = x.shape[0]
    n_experts = coeff.shape[0]
    out_dim = basis.shape[1]
    result = np.zeros((batch, n_experts, out_dim), dtype=ANALYSIS_DTYPE)
    for basis_idx in range(basis.shape[0]):
        basis_out = _safe_mm(f"basis_bank_{basis_idx}", x, basis[basis_idx].T)
        result += basis_out[:, None, :] * coeff[None, :, basis_idx][:, :, None]
    return _require_finite("basis_bank_result", result)


def _forward_down_basis(hidden: np.ndarray, basis: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    batch, n_experts, d_ffn = hidden.shape
    d_model = basis.shape[1]
    result = np.zeros((batch, n_experts, d_model), dtype=ANALYSIS_DTYPE)
    hidden_flat = hidden.reshape(batch * n_experts, d_ffn)
    for basis_idx in range(basis.shape[0]):
        basis_out = _safe_mm(f"down_basis_{basis_idx}", hidden_flat, basis[basis_idx].T)
        basis_out = basis_out.reshape(batch, n_experts, d_model)
        result += basis_out * coeff[None, :, basis_idx][:, :, None]
    return _require_finite("down_basis_result", result)


def _forward_shared_basis(
    x: np.ndarray,
    gate_basis: dict[str, np.ndarray | float],
    up_basis: dict[str, np.ndarray | float],
    down_basis: dict[str, np.ndarray | float],
) -> np.ndarray:
    gate = _forward_basis_bank(x, np.asarray(gate_basis["basis"]), np.asarray(gate_basis["coeff"]))
    up = _forward_basis_bank(x, np.asarray(up_basis["basis"]), np.asarray(up_basis["coeff"]))
    hidden = _require_finite("shared_basis_hidden", _sigmoid_linear_unit(gate) * up)
    return _forward_down_basis(hidden, np.asarray(down_basis["basis"]), np.asarray(down_basis["coeff"]))


def _family_raw_int4_mb(out_dim: int, in_dim: int, basis_count: int) -> float:
    return (out_dim * in_dim * basis_count * 0.5) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--cluster-size", type=int, default=24)
    parser.add_argument("--basis-counts", type=str, default="8,12,16,24")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROBE_OUT_DIR)
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")
    if args.samples <= 0:
        raise SystemExit("samples must be positive")

    all_experts = _load_all_experts(layer_npz)
    cluster_sizes = _parse_sizes(str(args.cluster_size), int(all_experts["n_experts"]))
    if len(cluster_sizes) != 1:
        raise SystemExit("cluster-size must resolve to exactly one value")
    cluster_size = cluster_sizes[0]

    similarity = _similarity_scores(all_experts, args.expert_id)
    cluster_indices = np.argsort(-similarity)[:cluster_size]
    experts = [_expert_from_all(all_experts, int(idx)) for idx in cluster_indices]
    basis_counts = _parse_basis_counts(args.basis_counts, cluster_size)

    d_model = int(experts[0]["d_model"])
    d_ffn = int(experts[0]["d_ffn"])
    rng = np.random.default_rng(args.seed)
    x = _require_finite(
        "probe_input",
        rng.standard_normal((args.samples, d_model)).astype(ANALYSIS_DTYPE, copy=False),
    )

    y_ref_parts = [_forward_expert(expert, x) for expert in experts]
    y_ref = np.stack(y_ref_parts, axis=1)

    gate_bank = np.stack([_as_analysis(expert["gate"]) for expert in experts], axis=0)
    up_bank = np.stack([_as_analysis(expert["up"]) for expert in experts], axis=0)
    down_bank = np.stack([_as_analysis(expert["down"]) for expert in experts], axis=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"qwen36_L{args.layer:02d}_expert{args.expert_id:03d}_cluster{cluster_size}_shared_basis_summary.json"
    summary: dict[str, object] = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "layer_npz": str(layer_npz),
        "cluster_size": cluster_size,
        "cluster_indices": [int(idx) for idx in cluster_indices.tolist()],
        "cluster_similarity_mean": float(np.mean(similarity[cluster_indices])),
        "cluster_similarity_min": float(np.min(similarity[cluster_indices])),
        "samples": args.samples,
        "seed": args.seed,
        "basis_counts": basis_counts,
        "basis_eval": [],
    }

    print("=== Qwen3.6 shared-basis routed-op probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  anchor expert_id: {args.expert_id}")
    print(f"  cluster_size: {cluster_size}")
    print(f"  cluster indices: {[int(idx) for idx in cluster_indices.tolist()]}")
    print(f"  basis_counts: {basis_counts}")

    for basis_count in basis_counts:
        gate_basis = _factor_shared_basis(gate_bank, basis_count)
        up_basis = _factor_shared_basis(up_bank, basis_count)
        down_basis = _factor_shared_basis(down_bank, basis_count)
        y_hat = _forward_shared_basis(x, gate_basis, up_basis, down_basis)

        overall = _metrics(y_ref.reshape(args.samples, cluster_size * d_model), y_hat.reshape(args.samples, cluster_size * d_model))
        per_expert = [
            _metrics(y_ref[:, expert_idx, :], y_hat[:, expert_idx, :])
            for expert_idx in range(cluster_size)
        ]
        family_raw_mb = {
            "gate": _family_raw_int4_mb(d_ffn, d_model, basis_count),
            "up": _family_raw_int4_mb(d_ffn, d_model, basis_count),
            "down": _family_raw_int4_mb(d_model, d_ffn, basis_count),
        }
        entry = {
            "basis_count": basis_count,
            "overall": overall,
            "per_expert": per_expert,
            "rel_fro": {
                "gate": float(gate_basis["rel_fro"]),
                "up": float(up_basis["rel_fro"]),
                "down": float(down_basis["rel_fro"]),
            },
            "family_raw_int4_mb": family_raw_mb,
        }
        summary["basis_eval"].append(entry)
        print(
            f"  basis {basis_count:>2}: "
            f"overall(mean={overall['cos_mean']:.6f}, min={overall['cos_min']:.6f}) "
            f"rel_fro(g={gate_basis['rel_fro']:.4f}, u={up_basis['rel_fro']:.4f}, d={down_basis['rel_fro']:.4f}) "
            f"family_raw_mb={family_raw_mb['gate']:.2f}"
        )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()