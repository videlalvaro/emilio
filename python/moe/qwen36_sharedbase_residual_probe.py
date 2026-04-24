"""qwen36_sharedbase_residual_probe.py — shared-base residual viability probe.

This is the first concrete test of the "shared expert as dense base, routed
experts as residual adapters" hypothesis for Qwen3.6.

It does not build a CoreML artifact. Instead it answers the cheaper gating
question first:

Is a routed expert materially easier to approximate as

  shared expert + low-rank residual

than as a direct low-rank factorization of the routed expert itself?

Run with any Python that has NumPy:
  python3 python/moe/qwen36_sharedbase_residual_probe.py --layer 0 --expert-id 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen36_pack_single_expert import _load_expert_weights
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR

PROBE_OUT_DIR = OUT_DIR / "sharedbase_residual_probe"
ANALYSIS_DTYPE = np.float64
MM_CHUNK = 128


def _as_analysis(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=ANALYSIS_DTYPE)


def _load_shared_weights(layer_npz: Path) -> dict[str, np.ndarray]:
    with np.load(layer_npz, allow_pickle=False) as data:
        return {
            "gate": _as_analysis(data["mlp__shared_expert__gate_proj__weight"]),
            "up": _as_analysis(data["mlp__shared_expert__up_proj__weight"]),
            "down": _as_analysis(data["mlp__shared_expert__down_proj__weight"]),
        }


def _parse_ranks(raw: str, max_rank: int) -> list[int]:
    ranks = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not ranks:
        raise SystemExit("expected at least one rank")
    bad = [rank for rank in ranks if rank <= 0 or rank > max_rank]
    if bad:
        raise SystemExit(f"invalid ranks {bad}; max rank is {max_rank}")
    return ranks


def _sigmoid_linear_unit(x: np.ndarray) -> np.ndarray:
    positive = x >= 0
    neg_exp = np.exp(-np.abs(x))
    sigmoid = np.empty_like(x, dtype=ANALYSIS_DTYPE)
    sigmoid[positive] = 1.0 / (1.0 + neg_exp[positive])
    sigmoid[~positive] = neg_exp[~positive] / (1.0 + neg_exp[~positive])
    return x * sigmoid


def _require_finite(name: str, array: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        raise SystemExit(f"non-finite values encountered in {name}")
    return array


def _safe_mm(name: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = _require_finite(f"{name}_left", _as_analysis(left))
    right64 = _require_finite(f"{name}_right", _as_analysis(right))
    if left64.ndim != 2 or right64.ndim != 2:
        raise SystemExit(f"expected rank-2 inputs for {name}, got {left64.shape} and {right64.shape}")
    if left64.shape[1] != right64.shape[0]:
        raise SystemExit(
            f"shape mismatch for {name}: {left64.shape} x {right64.shape}"
        )

    out = np.zeros((left64.shape[0], right64.shape[1]), dtype=ANALYSIS_DTYPE)
    for start in range(0, left64.shape[1], MM_CHUNK):
        stop = min(start + MM_CHUNK, left64.shape[1])
        out += np.einsum(
            "ik,kj->ij",
            left64[:, start:stop],
            right64[start:stop, :],
            optimize=False,
        )
    return _require_finite(name, out)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.maximum(left_norm * right_norm, 1e-12)
    return np.sum(left * right, axis=1) / denom


def _svd_cache(weight: np.ndarray) -> dict[str, np.ndarray | float]:
    weight64 = _require_finite("svd_weight", _as_analysis(weight))
    u, s, vt = np.linalg.svd(weight64, full_matrices=False)
    return {
        "weight": weight64,
        "u": u.astype(ANALYSIS_DTYPE),
        "s": s.astype(ANALYSIS_DTYPE),
        "vt": vt.astype(ANALYSIS_DTYPE),
        "fro": float(np.linalg.norm(weight64)),
    }


def _truncated_factors(cache: dict[str, np.ndarray | float], rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    u = _as_analysis(np.asarray(cache["u"]))
    s = _as_analysis(np.asarray(cache["s"]))
    vt = _as_analysis(np.asarray(cache["vt"]))
    weight = _as_analysis(np.asarray(cache["weight"]))
    fro = float(cache["fro"])
    sqrt_s = np.sqrt(s[:rank], dtype=ANALYSIS_DTYPE)
    second = u[:, :rank] * sqrt_s[None, :]
    first = sqrt_s[:, None] * vt[:rank, :]
    recon = _safe_mm("svd_reconstruction", second, first)
    rel_fro = float(np.linalg.norm(recon - weight) / max(fro, 1e-12))
    return first, second, rel_fro


def _forward_expert(expert: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    gate = _safe_mm("expert_gate", x, _as_analysis(expert["gate"]).T)
    up = _safe_mm("expert_up", x, _as_analysis(expert["up"]).T)
    hidden = _require_finite("expert_hidden", _sigmoid_linear_unit(gate) * up)
    return _safe_mm("expert_out", hidden, _as_analysis(expert["down"]).T)


def _forward_shared(shared: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    gate = _safe_mm("shared_gate", x, _as_analysis(shared["gate"]).T)
    up = _safe_mm("shared_up", x, _as_analysis(shared["up"]).T)
    hidden = _require_finite("shared_hidden", _sigmoid_linear_unit(gate) * up)
    return _safe_mm("shared_out", hidden, _as_analysis(shared["down"]).T)


def _forward_raw_lowrank(factors: dict[str, tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    gate_mid = _safe_mm("raw_gate_mid", x, factors["gate"][0].T)
    gate = _safe_mm("raw_gate", gate_mid, factors["gate"][1].T)
    up_mid = _safe_mm("raw_up_mid", x, factors["up"][0].T)
    up = _safe_mm("raw_up", up_mid, factors["up"][1].T)
    hidden = _require_finite("raw_hidden", _sigmoid_linear_unit(gate) * up)
    down_mid = _safe_mm("raw_down_mid", hidden, factors["down"][0].T)
    return _safe_mm("raw_out", down_mid, factors["down"][1].T)


def _forward_shared_residual_exact(
    shared: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    gate = _safe_mm("res_exact_gate", x, _as_analysis(shared["gate"]).T)
    gate += _safe_mm("res_exact_gate_delta", x, _as_analysis(residual["gate"]).T)
    up = _safe_mm("res_exact_up", x, _as_analysis(shared["up"]).T)
    up += _safe_mm("res_exact_up_delta", x, _as_analysis(residual["up"]).T)
    hidden = _require_finite("res_exact_hidden", _sigmoid_linear_unit(gate) * up)
    y = _safe_mm("res_exact_out_base", hidden, _as_analysis(shared["down"]).T)
    y += _safe_mm("res_exact_out_delta", hidden, _as_analysis(residual["down"]).T)
    return _require_finite("res_exact_out", y)


def _forward_shared_residual_lowrank(
    shared: dict[str, np.ndarray],
    factors: dict[str, tuple[np.ndarray, np.ndarray]],
    x: np.ndarray,
) -> np.ndarray:
    gate = _safe_mm("res_lr_gate", x, _as_analysis(shared["gate"]).T)
    gate += _safe_mm("res_lr_gate_delta", _safe_mm("res_lr_gate_mid", x, factors["gate"][0].T), factors["gate"][1].T)
    up = _safe_mm("res_lr_up", x, _as_analysis(shared["up"]).T)
    up += _safe_mm("res_lr_up_delta", _safe_mm("res_lr_up_mid", x, factors["up"][0].T), factors["up"][1].T)
    hidden = _require_finite("res_lr_hidden", _sigmoid_linear_unit(gate) * up)
    y = _safe_mm("res_lr_out_base", hidden, _as_analysis(shared["down"]).T)
    y += _safe_mm("res_lr_out_delta", _safe_mm("res_lr_out_mid", hidden, factors["down"][0].T), factors["down"][1].T)
    return _require_finite("res_lr_out", y)


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = _require_finite("metrics_reference", reference)
    candidate = _require_finite("metrics_candidate", candidate)
    cos = _require_finite("metrics_cos", _cosine_rows(reference, candidate))
    diff = _require_finite("metrics_diff", reference - candidate)
    return {
        "cos_min": float(np.min(cos)),
        "cos_mean": float(np.mean(cos)),
        "cos_p50": float(np.median(cos)),
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranks", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--out-dir", type=Path, default=PROBE_OUT_DIR)
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")
    if args.samples <= 0:
        raise SystemExit("samples must be positive")

    expert = _load_expert_weights(layer_npz, args.expert_id)
    shared = _load_shared_weights(layer_npz)
    d_model = int(expert["d_model"])
    max_rank = min(d_model, int(expert["d_ffn"]))
    ranks = _parse_ranks(args.ranks, max_rank)

    residual = {
        "gate": _as_analysis(expert["gate"]) - _as_analysis(shared["gate"]),
        "up": _as_analysis(expert["up"]) - _as_analysis(shared["up"]),
        "down": _as_analysis(expert["down"]) - _as_analysis(shared["down"]),
    }

    rng = np.random.default_rng(args.seed)
    x = _require_finite(
        "probe_input",
        rng.standard_normal((args.samples, d_model)).astype(ANALYSIS_DTYPE, copy=False),
    )

    y_ref = _forward_expert(expert, x)
    y_shared = _forward_shared(shared, x)
    y_residual_exact = _forward_shared_residual_exact(shared, residual, x)

    raw_svd = {
        "gate": _svd_cache(expert["gate"]),
        "up": _svd_cache(expert["up"]),
        "down": _svd_cache(expert["down"]),
    }
    residual_svd = {
        "gate": _svd_cache(residual["gate"]),
        "up": _svd_cache(residual["up"]),
        "down": _svd_cache(residual["down"]),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"qwen36_L{args.layer:02d}_expert{args.expert_id:03d}_sharedbase_residual_summary.json"

    summary: dict[str, object] = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "layer_npz": str(layer_npz),
        "samples": args.samples,
        "seed": args.seed,
        "ranks": ranks,
        "shared_only": _metrics(y_ref, y_shared),
        "shared_plus_full_residual": _metrics(y_ref, y_residual_exact),
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

    print("=== Qwen3.6 shared-base residual probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  expert_id: {args.expert_id}")
    print(f"  samples: {args.samples}")
    print(f"  ranks: {ranks}")
    print(
        "  shared-only output: "
        f"min={summary['shared_only']['cos_min']:.6f} "
        f"mean={summary['shared_only']['cos_mean']:.6f}"
    )
    print(
        "  shared+full residual sanity: "
        f"min={summary['shared_plus_full_residual']['cos_min']:.6f} "
        f"mean={summary['shared_plus_full_residual']['cos_mean']:.6f} "
        f"max_abs={summary['shared_plus_full_residual']['max_abs']:.6f}"
    )
    for name, stats in summary["matrix_norms"].items():
        print(
            f"  residual norm ratio {name}: "
            f"{stats['residual_vs_expert']:.6f} "
            f"(residual_fro={stats['residual_fro']:.3f} expert_fro={stats['expert_fro']:.3f})"
        )

    for rank in ranks:
        raw_factors = {}
        raw_rel_fro = {}
        residual_factors = {}
        residual_rel_fro = {}
        for name in ("gate", "up", "down"):
            raw_first, raw_second, raw_err = _truncated_factors(raw_svd[name], rank)
            res_first, res_second, res_err = _truncated_factors(residual_svd[name], rank)
            raw_factors[name] = (raw_first, raw_second)
            residual_factors[name] = (res_first, res_second)
            raw_rel_fro[name] = raw_err
            residual_rel_fro[name] = res_err

        y_raw = _forward_raw_lowrank(raw_factors, x)
        y_shared_res = _forward_shared_residual_lowrank(shared, residual_factors, x)
        raw_metrics = _metrics(y_ref, y_raw)
        shared_res_metrics = _metrics(y_ref, y_shared_res)

        entry = {
            "rank": rank,
            "raw_lowrank": {
                **raw_metrics,
                "gate_rel_fro": raw_rel_fro["gate"],
                "up_rel_fro": raw_rel_fro["up"],
                "down_rel_fro": raw_rel_fro["down"],
            },
            "sharedbase_residual": {
                **shared_res_metrics,
                "gate_rel_fro": residual_rel_fro["gate"],
                "up_rel_fro": residual_rel_fro["up"],
                "down_rel_fro": residual_rel_fro["down"],
            },
        }
        summary["ranks_eval"].append(entry)
        print(
            f"  rank {rank:>3}: raw(mean={raw_metrics['cos_mean']:.6f}, min={raw_metrics['cos_min']:.6f}) "
            f"vs shared+res(mean={shared_res_metrics['cos_mean']:.6f}, min={shared_res_metrics['cos_min']:.6f})"
        )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()