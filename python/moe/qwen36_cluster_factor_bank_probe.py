"""qwen36_cluster_factor_bank_probe.py — packed cluster-base factor-bank probe for Qwen.

This is the first CoreML-ready authoring path for the H1 branch:

  tight routed-expert cluster base + per-expert low-rank residuals

It does not claim to be the final serving layout. The purpose is narrower: build
one small factor-bank from real Qwen weights so we can measure package size,
compiled size, device placement, and prediction latency on the first H1-shaped
artifact that is actually supported by the offline numerics.

The important authoring detail is that the cluster axis is packed into the live
base linears and then reduced back to the same cluster mean used by the offline
probe. That keeps the math aligned with the positive H1 evidence while avoiding
another below-floor single-expert graph.

Run with the Xcode Python that has coremltools 9:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/qwen36_cluster_factor_bank_probe.py --layer 0 --expert-id 0 --cluster-size 16 --rank 128
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

from qwen36_clusterbase_residual_probe import (
    _cluster_base,
    _load_all_experts,
    _parse_sizes,
    _similarity_scores,
)
from qwen36_pack_single_expert import INT4_BLOCK_SIZE, _package_size_mb
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR
from qwen36_sharedbase_residual_probe import (
    _as_analysis,
    _forward_expert,
    _forward_shared_residual_lowrank,
    _metrics,
    _parse_ranks,
    _safe_mm,
)

FACTOR_BANK_OUT_DIR = OUT_DIR / "cluster_factor_bank_probe"


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
        "n_experts": np.array(n_experts, dtype=np.int32),
    }


def _factor_linear(weight: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    out_dim, in_dim = weight.shape
    max_rank = min(out_dim, in_dim)
    if rank <= 0 or rank > max_rank:
        raise SystemExit(f"rank must be in 1..{max_rank}, got {rank}")

    weight32 = np.asarray(weight, dtype=np.float32)
    u, s, vt = np.linalg.svd(weight32, full_matrices=False)
    sqrt_s = np.sqrt(s[:rank], dtype=np.float32)
    second = (u[:, :rank] * sqrt_s[None, :]).astype(np.float16)
    first = (sqrt_s[:, None] * vt[:rank, :]).astype(np.float16)
    recon = _safe_mm("cluster_factor_recon", second.astype(np.float32), first.astype(np.float32))
    rel_fro = float(np.linalg.norm(recon - weight32) / max(np.linalg.norm(weight32), 1e-12))
    return first, second, rel_fro


def _factor_residual_expert(expert: dict[str, np.ndarray], base: dict[str, np.ndarray], rank: int) -> dict[str, np.ndarray | float]:
    gate_residual = _as_analysis(expert["gate"]) - _as_analysis(base["gate"])
    up_residual = _as_analysis(expert["up"]) - _as_analysis(base["up"])
    down_residual = _as_analysis(expert["down"]) - _as_analysis(base["down"])
    gate_1, gate_2, gate_rel_fro = _factor_linear(gate_residual, rank)
    up_1, up_2, up_rel_fro = _factor_linear(up_residual, rank)
    down_1, down_2, down_rel_fro = _factor_linear(down_residual, rank)
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
    }


def _pack_base_bank(experts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    gate = np.concatenate([np.asarray(expert["gate"], dtype=np.float16) for expert in experts], axis=0)
    up = np.concatenate([np.asarray(expert["up"], dtype=np.float16) for expert in experts], axis=0)
    down = np.concatenate([np.asarray(expert["down"], dtype=np.float16) for expert in experts], axis=0)
    return {
        "gate": gate,
        "up": up,
        "down": down,
    }


def _build_probe(
    packed_base: dict[str, np.ndarray],
    residuals: list[dict[str, np.ndarray | float]],
    out_path: Path,
    batch: int,
) -> ct.models.MLModel:
    cluster_size = len(residuals)
    d_model = int(np.asarray(residuals[0]["down_2"]).shape[0])
    d_ffn = int(np.asarray(residuals[0]["gate_2"]).shape[0])
    rank = int(np.asarray(residuals[0]["gate_1"]).shape[0])
    fp16 = ct.converters.mil.mil.types.fp16
    zeros_rank = np.zeros((rank,), dtype=np.float16)
    zeros_ffn = np.zeros((d_ffn,), dtype=np.float16)
    zeros_model = np.zeros((d_model,), dtype=np.float16)
    zeros_base_ffn = np.zeros((cluster_size * d_ffn,), dtype=np.float16)
    zeros_base_model = np.zeros((cluster_size * d_model,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, d_model), dtype=fp16)])
    def prog(x):
        base_gate_bank = mb.linear(x=x, weight=packed_base["gate"], bias=zeros_base_ffn, name="base_gate_bank")
        base_gate_bank = mb.reshape(x=base_gate_bank, shape=(batch, cluster_size, d_ffn), name="base_gate_bank_r")
        base_gate = mb.reduce_mean(x=base_gate_bank, axes=[1], keep_dims=False, name="base_gate")
        base_up_bank = mb.linear(x=x, weight=packed_base["up"], bias=zeros_base_ffn, name="base_up_bank")
        base_up_bank = mb.reshape(x=base_up_bank, shape=(batch, cluster_size, d_ffn), name="base_up_bank_r")
        base_up = mb.reduce_mean(x=base_up_bank, axes=[1], keep_dims=False, name="base_up")
        outputs = []
        for slot, residual in enumerate(residuals):
            gate_mid = mb.linear(x=x, weight=residual["gate_1"], bias=zeros_rank, name=f"e{slot}_gate_mid")
            gate_delta = mb.linear(x=gate_mid, weight=residual["gate_2"], bias=zeros_ffn, name=f"e{slot}_gate_delta")
            gate = mb.add(x=base_gate, y=gate_delta, name=f"e{slot}_gate")
            up_mid = mb.linear(x=x, weight=residual["up_1"], bias=zeros_rank, name=f"e{slot}_up_mid")
            up_delta = mb.linear(x=up_mid, weight=residual["up_2"], bias=zeros_ffn, name=f"e{slot}_up_delta")
            up = mb.add(x=base_up, y=up_delta, name=f"e{slot}_up")
            act = mb.silu(x=gate, name=f"e{slot}_silu_gate")
            hidden = mb.mul(x=act, y=up, name=f"e{slot}_swiglu_mul")
            down_base_bank = mb.linear(x=hidden, weight=packed_base["down"], bias=zeros_base_model, name=f"e{slot}_down_base_bank")
            down_base_bank = mb.reshape(x=down_base_bank, shape=(batch, cluster_size, d_model), name=f"e{slot}_down_base_bank_r")
            down_base = mb.reduce_mean(x=down_base_bank, axes=[1], keep_dims=False, name=f"e{slot}_down_base")
            down_mid = mb.linear(x=hidden, weight=residual["down_1"], bias=zeros_rank, name=f"e{slot}_down_mid")
            down_delta = mb.linear(x=down_mid, weight=residual["down_2"], bias=zeros_model, name=f"e{slot}_down_delta")
            outputs.append(mb.add(x=down_base, y=down_delta, name=f"e{slot}_out"))
        if len(outputs) == 1:
            return outputs[0]
        return mb.concat(values=outputs, axis=1, name="concat_out")

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


def _time_predict(model: ct.models.MLModel, d_model: int, batch: int, n_iter: int = 60, warmup: int = 10) -> float:
    feed = {list(model.input_description)[0]: np.zeros((batch, d_model), dtype=np.float32)}
    for _ in range(warmup):
        model.predict(feed)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.predict(feed)
    return (time.perf_counter() - t0) / n_iter


def _raw_int4_mb(packed_base: dict[str, np.ndarray], residuals: list[dict[str, np.ndarray | float]]) -> float:
    total_weights = int(np.asarray(packed_base["gate"]).size + np.asarray(packed_base["up"]).size + np.asarray(packed_base["down"]).size)
    for residual in residuals:
        total_weights += int(np.asarray(residual["gate_1"]).size + np.asarray(residual["gate_2"]).size)
        total_weights += int(np.asarray(residual["up_1"]).size + np.asarray(residual["up_2"]).size)
        total_weights += int(np.asarray(residual["down_1"]).size + np.asarray(residual["down_2"]).size)
    return (total_weights * 0.5) / 1e6


def _validate_outputs(
    experts: list[dict[str, np.ndarray]],
    base: dict[str, np.ndarray],
    residuals: list[dict[str, np.ndarray | float]],
) -> dict[str, object]:
    d_model = int(experts[0]["d_model"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, d_model), dtype=np.float64)
    per_expert = []
    y_ref_parts = []
    y_hat_parts = []
    for expert, residual in zip(experts, residuals):
        residual_pairs = {
            "gate": (residual["gate_1"], residual["gate_2"]),
            "up": (residual["up_1"], residual["up_2"]),
            "down": (residual["down_1"], residual["down_2"]),
        }
        y_ref = _forward_expert(expert, x)
        y_hat = _forward_shared_residual_lowrank(base, residual_pairs, x)
        y_ref_parts.append(y_ref)
        y_hat_parts.append(y_hat)
        per_expert.append(_metrics(y_ref, y_hat))
    y_ref_cat = np.concatenate(y_ref_parts, axis=1)
    y_hat_cat = np.concatenate(y_hat_parts, axis=1)
    return {
        "overall": _metrics(y_ref_cat, y_hat_cat),
        "per_expert": per_expert,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--cluster-size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=FACTOR_BANK_OUT_DIR)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")
    if args.batch <= 0:
        raise SystemExit("batch must be positive")

    all_experts = _load_all_experts(layer_npz)
    cluster_sizes = _parse_sizes(str(args.cluster_size), int(all_experts["n_experts"]))
    if len(cluster_sizes) != 1:
        raise SystemExit("cluster-size must resolve to exactly one value")
    cluster_size = cluster_sizes[0]

    sample_expert = _expert_from_all(all_experts, args.expert_id)
    max_rank = min(int(sample_expert["d_model"]), int(sample_expert["d_ffn"]))
    ranks = _parse_ranks(str(args.rank), max_rank)
    if len(ranks) != 1:
        raise SystemExit("rank must resolve to exactly one value")
    rank = ranks[0]

    similarity = _similarity_scores(all_experts, args.expert_id)
    cluster_indices = np.argsort(-similarity)[:cluster_size]
    experts = [_expert_from_all(all_experts, int(idx)) for idx in cluster_indices]
    base = _cluster_base(all_experts, cluster_indices)
    packed_base = _pack_base_bank(experts)
    residuals = [_factor_residual_expert(expert, base, rank) for expert in experts]
    numerics = _validate_outputs(experts, base, residuals)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen36_L{args.layer:02d}_cluster{cluster_size}_expert{args.expert_id:03d}_r{rank}_B{args.batch}_int4"
    out_pkg = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    if compiled.exists():
        shutil.rmtree(compiled)

    print("=== Qwen3.6 cluster factor-bank probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  anchor expert_id: {args.expert_id}")
    print(f"  cluster_size: {cluster_size}")
    print(f"  cluster indices: {[int(idx) for idx in cluster_indices.tolist()]}")
    print(f"  rank: {rank}")
    print(f"  batch: {args.batch}")
    print(f"  block_size: {INT4_BLOCK_SIZE}")
    print(
        "  packed base shapes: "
        f"gate={packed_base['gate'].shape} up={packed_base['up'].shape} down={packed_base['down'].shape}"
    )
    print(f"  approx raw int4 bytes: {_raw_int4_mb(packed_base, residuals):.2f} MB")
    print(
        "  validation overall: "
        f"min={numerics['overall']['cos_min']:.6f} "
        f"mean={numerics['overall']['cos_mean']:.6f} "
        f"max_abs={numerics['overall']['max_abs']:.6f}"
    )

    t0 = time.perf_counter()
    _build_probe(packed_base, residuals, out_pkg, args.batch)
    convert_wall_s = time.perf_counter() - t0
    pkg_mb = _package_size_mb(out_pkg)
    print(f"  convert+quant wall: {convert_wall_s:.1f}s")
    print(f"  mlpackage size: {pkg_mb:.2f} MB")

    compiled_mb: float | None = None
    compile_wall_s: float | None = None
    linears: str | None = None
    predict_latency_ms: float | None = None
    if not args.skip_compile:
        print("  coremlcompiler...")
        t0 = time.perf_counter()
        compiled = Path(ct.utils.compile_model(str(out_pkg), str(compiled)))
        compile_wall_s = time.perf_counter() - t0
        compiled_mb = _package_size_mb(compiled)
        print(f"  compile wall: {compile_wall_s:.1f}s")
        print(f"  mlmodelc size: {compiled_mb:.2f} MB")
        linears = _device_summary(out_pkg)
        print(f"  linears: {linears}")
        if not args.skip_predict:
            model = ct.models.MLModel(str(out_pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
            predict_latency_ms = _time_predict(model, int(sample_expert['d_model']), args.batch) * 1e3
            print(f"  predict latency: {predict_latency_ms:.3f} ms")

    summary = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "cluster_size": cluster_size,
        "cluster_indices": [int(idx) for idx in cluster_indices.tolist()],
        "cluster_similarity_mean": float(np.mean(similarity[cluster_indices])),
        "cluster_similarity_min": float(np.min(similarity[cluster_indices])),
        "rank": rank,
        "batch": args.batch,
        "layer_npz": str(layer_npz),
        "out_pkg": str(out_pkg),
        "compiled": None if args.skip_compile else str(compiled),
        "block_size": INT4_BLOCK_SIZE,
        "packed_base_shapes": {
            "gate": list(packed_base["gate"].shape),
            "up": list(packed_base["up"].shape),
            "down": list(packed_base["down"].shape),
        },
        "raw_int4_mb": _raw_int4_mb(packed_base, residuals),
        "numerics": numerics,
        "package_size_mb": pkg_mb,
        "compiled_size_mb": compiled_mb,
        "convert_wall_s": convert_wall_s,
        "compile_wall_s": compile_wall_s,
        "predict_latency_ms": predict_latency_ms,
        "linears": linears,
        "residual_rel_fro": [
            {
                "gate": float(residual["gate_rel_fro"]),
                "up": float(residual["up_rel_fro"]),
                "down": float(residual["down_rel_fro"]),
            }
            for residual in residuals
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()