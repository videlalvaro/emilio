"""qwen36_single_expert_gate.py — bounded quality + placement gate for one expert artifact.

Validates an already-built Qwen3.6 routed-expert artifact against the exact
NumPy expert slice and reports CoreML compute-plan placement for its hot linear
ops. This is the gatekeeper-friendly check to run before timing any new
per-expert dispatch branch.

Run with Xcode python3 because coremltools 9 is required:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/qwen36_single_expert_gate.py --layer 0 --expert-id 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import coremltools as ct

from qwen36_pack_single_expert import EXPERTS_OUT_DIR, _load_expert_weights, _package_size_mb
from qwen36_phase0_spec import WEIGHTS_OUT_DIR


def _artifact_paths(layer: int, expert_id: int, experts_dir: Path) -> tuple[Path, Path, Path]:
    tag = f"qwen36_L{layer:02d}_expert{expert_id:03d}_int4"
    return (
        experts_dir / f"{tag}.mlpackage",
        experts_dir / f"{tag}.mlmodelc",
        experts_dir / f"{tag}_gate_summary.json",
    )


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _reference_forward(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    gate = x @ weights["gate"].astype(np.float32).T
    up = x @ weights["up"].astype(np.float32).T
    hidden = _silu(gate) * up
    return hidden @ weights["down"].astype(np.float32).T


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def _linear_device_summary(compiled_path: Path) -> dict[str, object]:
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        path=str(compiled_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    program = plan.model_structure.program
    if program is None:
        return {"devices": [], "counts": {}}

    devices: list[str] = []
    for fn in program.functions.values():
        for op in fn.block.operations:
            if op.operator_name not in ("ios18.linear", "linear"):
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            name = usage.preferred_compute_device.__class__.__name__ if usage else "?"
            if "Neural" in name:
                devices.append("ANE")
            elif "GPU" in name:
                devices.append("GPU")
            elif "CPU" in name:
                devices.append("CPU")
            else:
                devices.append("?")

    counts: dict[str, int] = {}
    for device in devices:
        counts[device] = counts.get(device, 0) + 1
    return {"devices": devices, "counts": counts}


def _predict_rows(model: ct.models.MLModel, rows: np.ndarray) -> np.ndarray:
    input_name = list(model.input_description)[0]
    output_rows: list[np.ndarray] = []
    for row in rows:
        pred = model.predict({input_name: row[None, :].astype(np.float32)})
        value = np.asarray(next(iter(pred.values())), dtype=np.float32).reshape(1, -1)
        output_rows.append(value)
    return np.concatenate(output_rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--experts-dir", type=Path, default=EXPERTS_OUT_DIR)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")

    pkg_path, compiled_path, summary_path = _artifact_paths(args.layer, args.expert_id, args.experts_dir)
    if not pkg_path.exists():
        raise SystemExit(f"missing mlpackage: {pkg_path}")
    if not compiled_path.exists():
        raise SystemExit(f"missing mlmodelc: {compiled_path}")

    weights = _load_expert_weights(layer_npz, args.expert_id)
    d_model = int(weights["d_model"])

    rng = np.random.default_rng(args.seed)
    rows = rng.standard_normal((args.n_samples, d_model), dtype=np.float32) * 0.1
    reference = _reference_forward(weights, rows)

    model = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    predicted = _predict_rows(model, rows)

    cosines = [_cosine(predicted[i], reference[i]) for i in range(args.n_samples)]
    max_abs = float(np.max(np.abs(predicted - reference)))
    placement = _linear_device_summary(compiled_path)

    summary = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "layer_npz": str(layer_npz),
        "mlpackage": str(pkg_path),
        "mlmodelc": str(compiled_path),
        "n_samples": args.n_samples,
        "seed": args.seed,
        "d_model": d_model,
        "d_ffn": int(weights["d_ffn"]),
        "package_size_mb": _package_size_mb(pkg_path),
        "compiled_size_mb": _package_size_mb(compiled_path),
        "cos_mean": float(np.mean(cosines)),
        "cos_min": float(np.min(cosines)),
        "cos_max": float(np.max(cosines)),
        "max_abs": max_abs,
        "linear_devices": placement["devices"],
        "linear_device_counts": placement["counts"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("=== Qwen3.6 single expert gate ===")
    print(f"  mlpackage: {pkg_path}")
    print(f"  mlmodelc : {compiled_path}")
    print(f"  package_size_mb : {summary['package_size_mb']:.6f}")
    print(f"  compiled_size_mb: {summary['compiled_size_mb']:.6f}")
    print(
        "  cosine: "
        f"mean={summary['cos_mean']:.9f} min={summary['cos_min']:.9f} max={summary['cos_max']:.9f}"
    )
    print(f"  max_abs: {summary['max_abs']:.9e}")
    print(f"  linears: {placement['devices']}")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()