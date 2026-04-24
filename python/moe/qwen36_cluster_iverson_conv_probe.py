"""qwen36_cluster_iverson_conv_probe.py — clustered packed-array ANE smoke probe.

This is the first Qwen-specific probe for a materially different execution
strategy after the shared-basis bank family closed. Instead of basis banks or
per-expert dispatch, it packs a tight cluster of real experts directly over the
expert axis:

  - one fused gate_up packed 1x1 conv over clustered experts
  - gate-weighted SwiGLU in packed expert space
  - one or more split down-projection 1x1 convs reduced by summation

The design mirrors the privacy-filter Iverson/Stepanov pattern, but on Qwen's
`2048 -> 512 -> 2048` routed expert geometry.

Book tie:
    - BOOK_ANALYSIS.md Experiment 17 (Iverson): rewrite expert routing as a packed
        array program over the expert axis instead of scalar per-expert dispatch.
    - BOOK_ANALYSIS.md Experiment 19 (Stepanov): split the packed down-projection
        into additive chunks when the packed expert axis becomes too wide for one op.
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen36_clusterbase_residual_probe import _load_all_experts, _parse_sizes, _similarity_scores
from qwen36_conv_expert_probe import _device_summary as _conv_device_summary
from qwen36_pack_single_expert import INT4_BLOCK_SIZE, _package_size_mb
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR

IVERSON_CONV_OUT_DIR = OUT_DIR / "cluster_iverson_conv_probe"


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


def _as64(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise SystemExit("non-finite values in analysis tensor")
    return out


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.maximum(left_norm * right_norm, 1e-12)
    return np.sum(left * right, axis=1) / denom


def _forward_reference(experts: list[dict[str, np.ndarray]], x: np.ndarray, gate_weights: np.ndarray) -> np.ndarray:
    gate_bank = np.stack([_as64(np.asarray(expert["gate"], dtype=np.float64)) for expert in experts], axis=0)
    up_bank = np.stack([_as64(np.asarray(expert["up"], dtype=np.float64)) for expert in experts], axis=0)
    down_bank = np.stack([_as64(np.asarray(expert["down"], dtype=np.float64)) for expert in experts], axis=0)

    gate = np.einsum("bd,gfd->bgf", _as64(x), gate_bank, optimize=False)
    up = np.einsum("bd,gfd->bgf", _as64(x), up_bank, optimize=False)
    hidden = gate / (1.0 + np.exp(-gate)) * up
    hidden = hidden * _as64(gate_weights)[:, :, None]
    down = np.einsum("bgf,gdf->bgd", hidden, down_bank, optimize=False)
    return np.einsum("bgd->bd", down, optimize=False)


def _validate_outputs(experts: list[dict[str, np.ndarray]], mod: nn.Module, cluster_size: int, d_model: int) -> dict[str, float]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, d_model), dtype=np.float32)
    gate_weights = rng.random((16, cluster_size), dtype=np.float32)
    gate_weights /= np.maximum(np.sum(gate_weights, axis=1, keepdims=True), 1e-12)

    y_ref = _forward_reference(experts, x, gate_weights)
    with torch.no_grad():
        y_hat = mod(
            torch.from_numpy(x).reshape(16, d_model, 1, 1),
            torch.from_numpy(gate_weights).reshape(16, cluster_size, 1, 1),
        ).detach().cpu().numpy().reshape(16, d_model)

    cos = _cosine_rows(y_ref, y_hat)
    return {
        "validation_cos_min": float(np.min(cos)),
        "validation_cos_mean": float(np.mean(cos)),
        "validation_cos_max": float(np.max(cos)),
        "validation_max_abs": float(np.max(np.abs(y_ref - y_hat))),
    }


def _time_predict(model: ct.models.MLModel, batch: int, d_model: int, cluster_size: int, n_iter: int = 60, warmup: int = 10) -> float:
    input_names = list(model.input_description)
    feed = {
        input_names[0]: np.zeros((batch, d_model, 1, 1), dtype=np.float32),
        input_names[1]: np.full((batch, cluster_size, 1, 1), 1.0 / cluster_size, dtype=np.float32),
    }
    for _ in range(warmup):
        model.predict(feed)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.predict(feed)
    return (time.perf_counter() - t0) / n_iter


class ClusterIversonConvProbe(nn.Module):
    def __init__(
        self,
        gate_bank: np.ndarray,
        up_bank: np.ndarray,
        down_bank: np.ndarray,
        pack2_splits: int,
    ):
        super().__init__()

        gate_bank_arr = np.asarray(gate_bank, dtype=np.float32)
        up_bank_arr = np.asarray(up_bank, dtype=np.float32)
        down_bank_arr = np.asarray(down_bank, dtype=np.float32)

        self.cluster_size = int(gate_bank_arr.shape[0])
        self.d_ffn = int(gate_bank_arr.shape[1])
        self.d_model = int(gate_bank_arr.shape[2])
        self.pack2_splits = int(pack2_splits)
        if self.pack2_splits <= 0 or self.cluster_size % self.pack2_splits != 0:
            raise ValueError(
                f"pack2_splits must divide cluster_size exactly, got {self.pack2_splits} for {self.cluster_size}"
            )
        self.experts_per_split = self.cluster_size // self.pack2_splits

        gate_up_bank_arr = np.concatenate([gate_bank_arr, up_bank_arr], axis=1)
        self.pack1 = nn.Conv2d(self.d_model, self.cluster_size * 2 * self.d_ffn, 1, bias=False)
        pack1_weight = gate_up_bank_arr.reshape(self.cluster_size * 2 * self.d_ffn, self.d_model, 1, 1)
        self.pack1.weight = nn.Parameter(torch.from_numpy(pack1_weight), requires_grad=False)

        self.pack2_chunks = nn.ModuleList()
        for split_index in range(self.pack2_splits):
            start = split_index * self.experts_per_split
            end = start + self.experts_per_split
            chunk_weight = np.concatenate(
                [down_bank_arr[expert_index] for expert_index in range(start, end)],
                axis=1,
            ).reshape(self.d_model, self.experts_per_split * self.d_ffn, 1, 1)
            conv = nn.Conv2d(self.experts_per_split * self.d_ffn, self.d_model, 1, bias=False)
            conv.weight = nn.Parameter(torch.from_numpy(chunk_weight), requires_grad=False)
            self.pack2_chunks.append(conv)

    def forward(self, x: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        gate_up_bank = self.pack1(x).view(batch, self.cluster_size, 2 * self.d_ffn, 1, 1)
        gate_bank = gate_up_bank[:, :, :self.d_ffn, :, :]
        up_bank = gate_up_bank[:, :, self.d_ffn:, :, :]
        hidden = F.silu(gate_bank) * up_bank
        hidden = hidden * gate_weights.view(batch, self.cluster_size, 1, 1, 1)

        acc = None
        for split_index, conv in enumerate(self.pack2_chunks):
            start = split_index * self.experts_per_split
            end = start + self.experts_per_split
            chunk = hidden[:, start:end, :, :, :].reshape(batch, self.experts_per_split * self.d_ffn, 1, 1)
            part = conv(chunk)
            acc = part if acc is None else acc + part
        return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--cluster-size", type=int, required=True)
    parser.add_argument("--pack2-splits", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=IVERSON_CONV_OUT_DIR)
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
    if args.pack2_splits <= 0 or cluster_size % args.pack2_splits != 0:
        raise SystemExit(f"pack2-splits must divide cluster-size exactly, got {args.pack2_splits} for {cluster_size}")

    similarity = _similarity_scores(all_experts, args.expert_id)
    cluster_indices = np.argsort(-similarity)[:cluster_size]
    experts = [_expert_from_all(all_experts, int(idx)) for idx in cluster_indices]
    sample_expert = experts[0]
    d_model = int(sample_expert["d_model"])
    d_ffn = int(sample_expert["d_ffn"])

    gate_bank = np.stack([np.asarray(expert["gate"], dtype=np.float32) for expert in experts], axis=0)
    up_bank = np.stack([np.asarray(expert["up"], dtype=np.float32) for expert in experts], axis=0)
    down_bank = np.stack([np.asarray(expert["down"], dtype=np.float32) for expert in experts], axis=0)

    mod = ClusterIversonConvProbe(gate_bank, up_bank, down_bank, args.pack2_splits).eval()
    numerics = _validate_outputs(experts, mod, cluster_size, d_model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"qwen36_L{args.layer:02d}_cluster{cluster_size}_expert{args.expert_id:03d}_"
        f"pack2s{args.pack2_splits}_B{args.batch}_iverson_conv_int4"
    )
    out_pkg = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    if compiled.exists():
        shutil.rmtree(compiled)

    pack1_raw_int4_mb = (d_model * cluster_size * 2 * d_ffn * 0.5) / 1e6
    pack2_total_raw_int4_mb = (d_model * cluster_size * d_ffn * 0.5) / 1e6
    pack2_per_split_raw_int4_mb = pack2_total_raw_int4_mb / args.pack2_splits

    print("=== Qwen3.6 clustered Iverson conv probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  anchor expert_id: {args.expert_id}")
    print(f"  cluster_size: {cluster_size}")
    print(f"  cluster indices: {[int(idx) for idx in cluster_indices.tolist()]}")
    print(f"  pack2_splits: {args.pack2_splits}")
    print(f"  batch: {args.batch}")
    print(f"  block_size: {INT4_BLOCK_SIZE}")
    print(f"  pack1 gate_up raw int4 mb: {pack1_raw_int4_mb:.2f}")
    print(f"  pack2 total raw int4 mb: {pack2_total_raw_int4_mb:.2f}")
    print(f"  pack2 per-split raw int4 mb: {pack2_per_split_raw_int4_mb:.2f}")
    print(
        "  validation: "
        f"cos_mean={numerics['validation_cos_mean']:.6f} "
        f"cos_min={numerics['validation_cos_min']:.6f} "
        f"max_abs={numerics['validation_max_abs']:.6f}"
    )

    x_ex = torch.ones(args.batch, d_model, 1, 1, dtype=torch.float32)
    g_ex = torch.full((args.batch, cluster_size, 1, 1), 1.0 / cluster_size, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex, g_ex))

    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x_in", shape=(args.batch, d_model, 1, 1), dtype=np.float16),
            ct.TensorType(name="g_in", shape=(args.batch, cluster_size, 1, 1), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="y_out", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
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
    mlmodel.save(str(out_pkg))
    convert_wall_s = time.perf_counter() - t0
    pkg_mb = _package_size_mb(out_pkg)
    print(f"  convert+quant wall: {convert_wall_s:.1f}s")
    print(f"  mlpackage size: {pkg_mb:.2f} MB")

    compiled_mb: float | None = None
    compile_wall_s: float | None = None
    convs: str | None = None
    predict_latency_ms: float | None = None
    if not args.skip_compile:
        print("  coremlcompiler...")
        t0 = time.perf_counter()
        compiled = Path(ct.utils.compile_model(str(out_pkg), str(compiled)))
        compile_wall_s = time.perf_counter() - t0
        compiled_mb = _package_size_mb(compiled)
        print(f"  compile wall: {compile_wall_s:.1f}s")
        print(f"  mlmodelc size: {compiled_mb:.2f} MB")
        convs = _conv_device_summary(out_pkg)
        print(f"  convs: {convs}")
        if not args.skip_predict:
            model = ct.models.MLModel(str(out_pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
            predict_latency_ms = _time_predict(model, args.batch, d_model, cluster_size) * 1e3
            print(f"  predict latency: {predict_latency_ms:.3f} ms")

    summary = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "cluster_size": cluster_size,
        "cluster_indices": [int(idx) for idx in cluster_indices.tolist()],
        "cluster_similarity_mean": float(np.mean(similarity[cluster_indices])),
        "cluster_similarity_min": float(np.min(similarity[cluster_indices])),
        "pack2_splits": args.pack2_splits,
        "batch": args.batch,
        "layer_npz": str(layer_npz),
        "out_pkg": str(out_pkg),
        "compiled": None if args.skip_compile else str(compiled),
        "block_size": INT4_BLOCK_SIZE,
        "d_model": d_model,
        "d_ffn": d_ffn,
        "pack1_gate_up_raw_int4_mb": pack1_raw_int4_mb,
        "pack2_total_raw_int4_mb": pack2_total_raw_int4_mb,
        "pack2_per_split_raw_int4_mb": pack2_per_split_raw_int4_mb,
        "package_size_mb": pkg_mb,
        "compiled_size_mb": compiled_mb,
        "convert_wall_s": convert_wall_s,
        "compile_wall_s": compile_wall_s,
        "predict_latency_ms": predict_latency_ms,
        "convs": convs,
        **numerics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()