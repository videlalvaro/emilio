"""qwen36_shared_basis_conv_bank_probe.py — conv-authored shared-basis bank probe.

This is the next H1 operator-family probe after exact shared-basis linear banks
at cluster24/basis24 and cluster32/basis32 still placed all hot ops on CPU.

The geometry is unchanged from the shared-basis bank probe:
  - one gate basis bank
  - one up basis bank
  - one down basis bank

The only substantive change is operator family: the three hot banks are authored
as 1x1 Conv2d layers instead of MIL linears, following the strongest surviving
PF operator-family prior.
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
from qwen36_shared_basis_bank_probe import _expert_from_all, _raw_int4_mb, _validate_outputs
from qwen36_shared_basis_probe import _factor_shared_basis, _parse_basis_counts

CONV_SHARED_BASIS_OUT_DIR = OUT_DIR / "shared_basis_conv_bank_probe"


class SharedBasisConvBank(nn.Module):
    def __init__(
        self,
        gate_basis: dict[str, np.ndarray | float],
        up_basis: dict[str, np.ndarray | float],
        down_basis: dict[str, np.ndarray | float],
        cluster_size: int,
    ):
        super().__init__()
        gate_basis_arr = np.asarray(gate_basis["basis"], dtype=np.float32)
        up_basis_arr = np.asarray(up_basis["basis"], dtype=np.float32)
        down_basis_arr = np.asarray(down_basis["basis"], dtype=np.float32)
        gate_coeff = np.asarray(gate_basis["coeff"], dtype=np.float32)
        up_coeff = np.asarray(up_basis["coeff"], dtype=np.float32)
        down_coeff = np.asarray(down_basis["coeff"], dtype=np.float32)

        self.cluster_size = cluster_size
        self.basis_count = int(gate_basis_arr.shape[0])
        self.d_ffn = int(gate_basis_arr.shape[1])
        self.d_model = int(gate_basis_arr.shape[2])

        self.gate_basis = nn.Conv2d(self.d_model, self.basis_count * self.d_ffn, 1, bias=True)
        self.up_basis = nn.Conv2d(self.d_model, self.basis_count * self.d_ffn, 1, bias=True)
        self.down_basis = nn.Conv2d(self.d_ffn, self.basis_count * self.d_model, 1, bias=True)

        gate_weight = gate_basis_arr.reshape(self.basis_count * self.d_ffn, self.d_model, 1, 1)
        up_weight = up_basis_arr.reshape(self.basis_count * self.d_ffn, self.d_model, 1, 1)
        down_weight = down_basis_arr.reshape(self.basis_count * self.d_model, self.d_ffn, 1, 1)
        zeros_basis_ffn = np.zeros((self.basis_count * self.d_ffn,), dtype=np.float32)
        zeros_basis_model = np.zeros((self.basis_count * self.d_model,), dtype=np.float32)

        self.gate_basis.weight = nn.Parameter(torch.from_numpy(gate_weight), requires_grad=False)
        self.gate_basis.bias = nn.Parameter(torch.from_numpy(zeros_basis_ffn), requires_grad=False)
        self.up_basis.weight = nn.Parameter(torch.from_numpy(up_weight), requires_grad=False)
        self.up_basis.bias = nn.Parameter(torch.from_numpy(zeros_basis_ffn), requires_grad=False)
        self.down_basis.weight = nn.Parameter(torch.from_numpy(down_weight), requires_grad=False)
        self.down_basis.bias = nn.Parameter(torch.from_numpy(zeros_basis_model), requires_grad=False)

        self.register_buffer("gate_coeff", torch.from_numpy(gate_coeff), persistent=False)
        self.register_buffer("up_coeff", torch.from_numpy(up_coeff), persistent=False)
        self.register_buffer("down_coeff", torch.from_numpy(down_coeff), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        gate_bank = self.gate_basis(x).view(batch, self.basis_count, self.d_ffn, 1, 1)
        up_bank = self.up_basis(x).view(batch, self.basis_count, self.d_ffn, 1, 1)

        hidden_parts = []
        for slot in range(self.cluster_size):
            gate = (gate_bank * self.gate_coeff[slot].view(1, self.basis_count, 1, 1, 1)).sum(dim=1)
            up = (up_bank * self.up_coeff[slot].view(1, self.basis_count, 1, 1, 1)).sum(dim=1)
            hidden_parts.append(F.silu(gate) * up)

        hidden_all = torch.stack(hidden_parts, dim=1).reshape(batch * self.cluster_size, self.d_ffn, 1, 1)
        down_bank = self.down_basis(hidden_all).view(
            batch,
            self.cluster_size,
            self.basis_count * self.d_model,
            1,
            1,
        )

        outputs = []
        for slot in range(self.cluster_size):
            slot_bank = down_bank[:, slot].view(batch, self.basis_count, self.d_model, 1, 1)
            out = (slot_bank * self.down_coeff[slot].view(1, self.basis_count, 1, 1, 1)).sum(dim=1)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--cluster-size", type=int, required=True)
    parser.add_argument("--basis-count", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=CONV_SHARED_BASIS_OUT_DIR)
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
    numerics = _validate_outputs(experts, gate_basis, up_basis, down_basis)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen36_L{args.layer:02d}_cluster{cluster_size}_expert{args.expert_id:03d}_basis{basis_count}_B{args.batch}_conv_int4"
    out_pkg = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    if compiled.exists():
        shutil.rmtree(compiled)

    print("=== Qwen3.6 shared-basis conv bank probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  anchor expert_id: {args.expert_id}")
    print(f"  cluster_size: {cluster_size}")
    print(f"  cluster indices: {[int(idx) for idx in cluster_indices.tolist()]}")
    print(f"  basis_count: {basis_count}")
    print(f"  batch: {args.batch}")
    print(f"  block_size: {INT4_BLOCK_SIZE}")
    print(f"  family raw int4 mb: {(d_model * d_ffn * basis_count * 0.5) / 1e6:.2f}")
    print(f"  total raw int4 mb: {_raw_int4_mb(d_model, d_ffn, basis_count):.2f}")
    print(
        "  validation overall: "
        f"min={numerics['overall']['cos_min']:.6f} "
        f"mean={numerics['overall']['cos_mean']:.6f} "
        f"max_abs={numerics['overall']['max_abs']:.6f}"
    )

    mod = SharedBasisConvBank(gate_basis, up_basis, down_basis, cluster_size).eval()
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
        "family_raw_int4_mb": (d_model * d_ffn * basis_count * 0.5) / 1e6,
        "numerics": numerics,
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