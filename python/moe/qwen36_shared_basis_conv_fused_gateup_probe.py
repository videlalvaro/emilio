"""qwen36_shared_basis_conv_fused_gateup_probe.py — fused Gate+Up conv-bank probe.

This is the next smallest public-CoreML rewrite after the minimal three-conv
isolate still placed all hot conv banks on CPU. It keeps the same cluster-shared
basis geometry, but fuses Gate+Up into one wider 1x1 conv bank so the hot path
becomes:

  - one fused gate_up basis conv bank
  - one down basis conv bank
  - reshape + reduce_mean around each bank

The goal is to test whether SwiGLU-style Gate+Up fusion changes ANE placement
for the same exact shared-basis bank bytes.
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
from qwen36_conv_expert_probe import _device_summary as _conv_device_summary, _time_predict as _conv_time_predict
from qwen36_pack_single_expert import INT4_BLOCK_SIZE, _package_size_mb
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR
from qwen36_shared_basis_bank_probe import _expert_from_all, _raw_int4_mb
from qwen36_shared_basis_probe import _factor_shared_basis, _parse_basis_counts

FUSED_GATEUP_CONV_OUT_DIR = OUT_DIR / "shared_basis_conv_fused_gateup_probe"


class FusedGateUpSharedBasisConvProbe(nn.Module):
    def __init__(self, gate_basis: np.ndarray, up_basis: np.ndarray, down_basis: np.ndarray):
        super().__init__()
        gate_basis_arr = np.asarray(gate_basis, dtype=np.float32)
        up_basis_arr = np.asarray(up_basis, dtype=np.float32)
        down_basis_arr = np.asarray(down_basis, dtype=np.float32)

        self.basis_count = int(gate_basis_arr.shape[0])
        self.d_ffn = int(gate_basis_arr.shape[1])
        self.d_model = int(gate_basis_arr.shape[2])

        gate_up_basis_arr = np.concatenate([gate_basis_arr, up_basis_arr], axis=1)
        self.gate_up_basis = nn.Conv2d(self.d_model, self.basis_count * 2 * self.d_ffn, 1, bias=True)
        self.down_basis = nn.Conv2d(self.d_ffn, self.basis_count * self.d_model, 1, bias=True)

        gate_up_weight = gate_up_basis_arr.reshape(self.basis_count * 2 * self.d_ffn, self.d_model, 1, 1)
        down_weight = down_basis_arr.reshape(self.basis_count * self.d_model, self.d_ffn, 1, 1)
        zeros_gate_up = np.zeros((self.basis_count * 2 * self.d_ffn,), dtype=np.float32)
        zeros_model = np.zeros((self.basis_count * self.d_model,), dtype=np.float32)

        self.gate_up_basis.weight = nn.Parameter(torch.from_numpy(gate_up_weight), requires_grad=False)
        self.gate_up_basis.bias = nn.Parameter(torch.from_numpy(zeros_gate_up), requires_grad=False)
        self.down_basis.weight = nn.Parameter(torch.from_numpy(down_weight), requires_grad=False)
        self.down_basis.bias = nn.Parameter(torch.from_numpy(zeros_model), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        gate_up_bank = self.gate_up_basis(x).view(batch, self.basis_count, 2 * self.d_ffn, 1, 1).mean(dim=1).view(batch, 2, self.d_ffn, 1, 1)
        gate_bank = gate_up_bank[:, 0]
        up_bank = gate_up_bank[:, 1]
        hidden = F.silu(gate_bank) * up_bank
        down_bank = self.down_basis(hidden).view(batch, self.basis_count, self.d_model, 1, 1).mean(dim=1)
        return down_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--cluster-size", type=int, required=True)
    parser.add_argument("--basis-count", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=FUSED_GATEUP_CONV_OUT_DIR)
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
    basis_counts = _parse_basis_counts(str(args.basis_count), cluster_size)
    if len(basis_counts) != 1:
        raise SystemExit("basis-count must resolve to exactly one value")
    basis_count = basis_counts[0]

    similarity = _similarity_scores(all_experts, args.expert_id)
    cluster_indices = np.argsort(-similarity)[:cluster_size]
    experts = [_expert_from_all(all_experts, int(idx)) for idx in cluster_indices]
    d_model = int(experts[0]["d_model"])
    d_ffn = int(experts[0]["d_ffn"])

    gate_bank = np.stack([np.asarray(expert["gate"], dtype=np.float64) for expert in experts], axis=0)
    up_bank = np.stack([np.asarray(expert["up"], dtype=np.float64) for expert in experts], axis=0)
    down_bank = np.stack([np.asarray(expert["down"], dtype=np.float64) for expert in experts], axis=0)

    gate_basis = _factor_shared_basis(gate_bank, basis_count)
    up_basis = _factor_shared_basis(up_bank, basis_count)
    down_basis = _factor_shared_basis(down_bank, basis_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen36_L{args.layer:02d}_cluster{cluster_size}_expert{args.expert_id:03d}_basis{basis_count}_B{args.batch}_conv_fused_gateup_int4"
    out_pkg = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    if compiled.exists():
        shutil.rmtree(compiled)

    fused_family_raw_mb = (d_model * d_ffn * basis_count) / 1e6
    print("=== Qwen3.6 fused Gate+Up shared-basis conv probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  anchor expert_id: {args.expert_id}")
    print(f"  cluster_size: {cluster_size}")
    print(f"  cluster indices: {[int(idx) for idx in cluster_indices.tolist()]}")
    print(f"  basis_count: {basis_count}")
    print(f"  batch: {args.batch}")
    print(f"  block_size: {INT4_BLOCK_SIZE}")
    print(f"  fused gate_up raw int4 mb: {fused_family_raw_mb:.2f}")
    print(f"  down raw int4 mb: {(d_model * d_ffn * basis_count * 0.5) / 1e6:.2f}")
    print(f"  total raw int4 mb: {_raw_int4_mb(d_model, d_ffn, basis_count):.2f}")

    mod = FusedGateUpSharedBasisConvProbe(
        np.asarray(gate_basis["basis"]),
        np.asarray(up_basis["basis"]),
        np.asarray(down_basis["basis"]),
    ).eval()
    x_ex = torch.zeros(args.batch, d_model, 1, 1, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex,))

    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x_in", shape=(args.batch, d_model, 1, 1), dtype=np.float16)],
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
            predict_latency_ms = _conv_time_predict(model, args.batch, d_model) * 1e3
            print(f"  predict latency: {predict_latency_ms:.3f} ms")

    summary = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "cluster_size": cluster_size,
        "cluster_indices": [int(idx) for idx in cluster_indices.tolist()],
        "cluster_similarity_mean": float(np.mean(similarity[cluster_indices])),
        "cluster_similarity_min": float(np.min(similarity[cluster_indices])),
        "basis_count": basis_count,
        "batch": args.batch,
        "layer_npz": str(layer_npz),
        "out_pkg": str(out_pkg),
        "compiled": None if args.skip_compile else str(compiled),
        "block_size": INT4_BLOCK_SIZE,
        "raw_int4_mb": _raw_int4_mb(d_model, d_ffn, basis_count),
        "fused_gate_up_raw_int4_mb": fused_family_raw_mb,
        "down_raw_int4_mb": (d_model * d_ffn * basis_count * 0.5) / 1e6,
        "rel_fro": {
            "gate": float(gate_basis["rel_fro"]),
            "up": float(up_basis["rel_fro"]),
            "down": float(down_basis["rel_fro"]),
        },
        "package_size_mb": pkg_mb,
        "compiled_size_mb": compiled_mb,
        "convert_wall_s": convert_wall_s,
        "compile_wall_s": compile_wall_s,
        "predict_latency_ms": predict_latency_ms,
        "convs": convs,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()