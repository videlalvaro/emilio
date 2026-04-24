"""qwen36_lowrank_expert_probe.py — low-rank expert residency probe for Qwen.

This is the next geometry-change probe after three failures on the raw expert
shape:

- separate-function MFD bank fell back to CPU
- packed executed-op probes fell back to CPU
- conv-as-linear authoring fell back to CPU

The new hypothesis is that the full expert matrices are the wrong executed
geometry for ANE, and that replacing each expert linear with a two-stage
low-rank factorization may change placement enough to become viable.

Run with the Xcode Python that has coremltools 9:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/qwen36_lowrank_expert_probe.py --layer 0 --expert-id 0 --rank 64
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb

from qwen36_pack_single_expert import (
    INT4_BLOCK_SIZE,
    _load_expert_weights,
    _package_size_mb,
)
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR

LOWRANK_OUT_DIR = OUT_DIR / "lowrank_probe"


def _factor_linear(weight: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    out_dim, in_dim = weight.shape
    max_rank = min(out_dim, in_dim)
    if rank <= 0 or rank > max_rank:
        raise SystemExit(f"rank must be in 1..{max_rank}, got {rank}")

    w32 = np.asarray(weight, dtype=np.float32)
    u, s, vt = np.linalg.svd(w32, full_matrices=False)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vt_r = vt[:rank, :]
    sqrt_s = np.sqrt(s_r, dtype=np.float32)

    second = (u_r * sqrt_s[None, :]).astype(np.float16)
    first = (sqrt_s[:, None] * vt_r).astype(np.float16)

    recon = second.astype(np.float32) @ first.astype(np.float32)
    rel_fro = float(np.linalg.norm(recon - w32) / max(np.linalg.norm(w32), 1e-12))
    return first, second, rel_fro


def _factor_expert(weights: dict[str, np.ndarray], rank: int) -> dict[str, np.ndarray | float]:
    gate_1, gate_2, gate_rel_fro = _factor_linear(np.asarray(weights["gate"], dtype=np.float16), rank)
    up_1, up_2, up_rel_fro = _factor_linear(np.asarray(weights["up"], dtype=np.float16), rank)
    down_1, down_2, down_rel_fro = _factor_linear(np.asarray(weights["down"], dtype=np.float16), rank)
    return {
        "gate_1": gate_1,
        "gate_2": gate_2,
        "up_1": up_1,
        "up_2": up_2,
        "down_1": down_1,
        "down_2": down_2,
        "gate_rel_fro": gate_rel_fro,
        "up_rel_fro": up_rel_fro,
        "down_rel_fro": down_rel_fro,
        "d_model": np.array(weights["d_model"], dtype=np.int32),
        "d_ffn": np.array(weights["d_ffn"], dtype=np.int32),
        "rank": np.array(rank, dtype=np.int32),
    }


def _build_probe(weights: dict[str, np.ndarray | float], out_path: Path, batch: int) -> ct.models.MLModel:
    d_model = int(weights["d_model"])
    d_ffn = int(weights["d_ffn"])
    rank = int(weights["rank"])
    fp16 = ct.converters.mil.mil.types.fp16
    zeros_rank = np.zeros((rank,), dtype=np.float16)
    zeros_ffn = np.zeros((d_ffn,), dtype=np.float16)
    zeros_model = np.zeros((d_model,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, d_model), dtype=fp16)])
    def prog(x):
        gate_mid = mb.linear(x=x, weight=weights["gate_1"], bias=zeros_rank, name="gate_mid")
        gate = mb.linear(x=gate_mid, weight=weights["gate_2"], bias=zeros_ffn, name="gate")
        up_mid = mb.linear(x=x, weight=weights["up_1"], bias=zeros_rank, name="up_mid")
        up = mb.linear(x=up_mid, weight=weights["up_2"], bias=zeros_ffn, name="up")
        act = mb.silu(x=gate, name="silu_gate")
        hidden = mb.mul(x=act, y=up, name="swiglu_mul")
        down_mid = mb.linear(x=hidden, weight=weights["down_1"], bias=zeros_rank, name="down_mid")
        y = mb.linear(x=down_mid, weight=weights["down_2"], bias=zeros_model, name="down")
        return y

    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel = cto.linear_quantize_weights(
        mlmodel,
        config=cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int4",
                granularity="per_block",
                block_size=INT4_BLOCK_SIZE,
                weight_threshold=0,
            )
        ),
    )
    mlmodel.save(str(out_path))
    return mlmodel


def _device_summary(pkg_path: Path) -> str:
    try:
        from coremltools.models.compute_plan import MLComputePlan

        compiled = pkg_path.with_suffix(".mlmodelc")
        if compiled.exists():
            shutil.rmtree(compiled)
        compiled = Path(ct.utils.compile_model(str(pkg_path), str(compiled)))
        plan = MLComputePlan.load_from_path(
            path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
        program = plan.model_structure.program
        if program is None:
            return "?"
        devices: list[str] = []
        for fn in program.functions.values():
            for op in fn.block.operations:
                if op.operator_name not in ("ios18.linear", "linear"):
                    continue
                try:
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
                except Exception:
                    devices.append("?")
        return "/".join(devices) if devices else "?"
    except Exception as exc:
        return f"err:{type(exc).__name__}"


def _time_predict(model: ct.models.MLModel, batch: int, d_model: int, n_iter: int = 60, warmup: int = 10) -> float:
    feed = {list(model.input_description)[0]: np.zeros((batch, d_model), dtype=np.float32)}
    for _ in range(warmup):
        model.predict(feed)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.predict(feed)
    return (time.perf_counter() - t0) / n_iter


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.maximum(left_norm * right_norm, 1e-12)
    return np.sum(left * right, axis=1) / denom


def _forward_original(expert: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    gate = x @ np.asarray(expert["gate"], dtype=np.float32).T
    up = x @ np.asarray(expert["up"], dtype=np.float32).T
    act = gate / (1.0 + np.exp(-gate))
    hidden = act * up
    return hidden @ np.asarray(expert["down"], dtype=np.float32).T


def _forward_factorized(factored: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    gate_mid = x @ np.asarray(factored["gate_1"], dtype=np.float32).T
    gate = gate_mid @ np.asarray(factored["gate_2"], dtype=np.float32).T
    up_mid = x @ np.asarray(factored["up_1"], dtype=np.float32).T
    up = up_mid @ np.asarray(factored["up_2"], dtype=np.float32).T
    act = gate / (1.0 + np.exp(-gate))
    hidden = act * up
    down_mid = hidden @ np.asarray(factored["down_1"], dtype=np.float32).T
    return down_mid @ np.asarray(factored["down_2"], dtype=np.float32).T


def _validate_outputs(expert: dict[str, np.ndarray], factored: dict[str, np.ndarray | float]) -> dict[str, float]:
    d_model = int(expert["d_model"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, d_model), dtype=np.float32)
    y_ref = _forward_original(expert, x)
    y_hat = _forward_factorized(factored, x)
    cos = _cosine_rows(y_ref, y_hat)
    max_abs = float(np.max(np.abs(y_ref - y_hat)))
    return {
        "validation_cos_min": float(np.min(cos)),
        "validation_cos_mean": float(np.mean(cos)),
        "validation_cos_max": float(np.max(cos)),
        "validation_max_abs": max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=LOWRANK_OUT_DIR)
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")
    if args.batch <= 0:
        raise SystemExit("batch must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen36_L{args.layer:02d}_expert{args.expert_id:03d}_r{args.rank}_B{args.batch}_lowrank_int4"
    out_pkg = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    if compiled.exists():
        shutil.rmtree(compiled)

    print("=== Qwen3.6 low-rank expert probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  expert_id: {args.expert_id}")
    print(f"  rank: {args.rank}")
    print(f"  batch: {args.batch}")
    print(f"  block_size: {INT4_BLOCK_SIZE}")

    expert = _load_expert_weights(layer_npz, args.expert_id)
    factored = _factor_expert(expert, args.rank)
    numerics = _validate_outputs(expert, factored)
    print(
        "  factor shapes: "
        f"gate=({factored['gate_1'].shape}, {factored['gate_2'].shape}) "
        f"up=({factored['up_1'].shape}, {factored['up_2'].shape}) "
        f"down=({factored['down_1'].shape}, {factored['down_2'].shape})"
    )
    print(
        "  rel fro: "
        f"gate={factored['gate_rel_fro']:.5f} "
        f"up={factored['up_rel_fro']:.5f} "
        f"down={factored['down_rel_fro']:.5f}"
    )
    print(
        "  output cos: "
        f"min={numerics['validation_cos_min']:.6f} "
        f"mean={numerics['validation_cos_mean']:.6f} "
        f"max_abs={numerics['validation_max_abs']:.6f}"
    )

    t0 = time.perf_counter()
    _build_probe(factored, out_pkg, args.batch)
    convert_wall_s = time.perf_counter() - t0
    pkg_mb = _package_size_mb(out_pkg)
    print(f"  convert+quant wall: {convert_wall_s:.1f}s")
    print(f"  mlpackage size: {pkg_mb:.2f} MB")

    print("  coremlcompiler...")
    t0 = time.perf_counter()
    compiled = Path(ct.utils.compile_model(str(out_pkg), str(compiled)))
    compile_wall_s = time.perf_counter() - t0
    compiled_mb = _package_size_mb(compiled)
    print(f"  compile wall: {compile_wall_s:.1f}s")
    print(f"  mlmodelc size: {compiled_mb:.2f} MB")

    model = ct.models.MLModel(str(out_pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    latency_ms = _time_predict(model, args.batch, int(expert["d_model"])) * 1e3
    devices = _device_summary(out_pkg)
    print(f"  predict latency: {latency_ms:.3f} ms")
    print(f"  linears: {devices}")

    summary = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "rank": args.rank,
        "batch": args.batch,
        "layer_npz": str(layer_npz),
        "out_pkg": str(out_pkg),
        "compiled": str(compiled),
        "block_size": INT4_BLOCK_SIZE,
        "d_model": int(expert["d_model"]),
        "d_ffn": int(expert["d_ffn"]),
        "package_size_mb": pkg_mb,
        "compiled_size_mb": compiled_mb,
        "convert_wall_s": convert_wall_s,
        "compile_wall_s": compile_wall_s,
        "predict_latency_ms": latency_ms,
        "linears": devices,
        "gate_rel_fro": float(factored["gate_rel_fro"]),
        "up_rel_fro": float(factored["up_rel_fro"]),
        "down_rel_fro": float(factored["down_rel_fro"]),
        **numerics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()