"""qwen36_shared_basis_bank_probe.py — shared-basis bank CoreML probe for Qwen.

This is the next H1 CoreML authoring path after the packed-base residual graph
showed that widening only the base is not enough. The graph here uses three
large shared-basis banks across a routed cluster:

  gate_e ~= sum_l a_gate[e, l] * B_gate[l]
  up_e   ~= sum_l a_up[e, l]   * B_up[l]
  down_e ~= sum_l a_down[e, l] * B_down[l]

The goal is to keep the hot linears large and few:
- one large gate basis linear
- one large up basis linear
- one large down basis linear over all cluster hidden states

Everything else is coefficient mixing and elementwise glue.
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

from qwen36_cluster_factor_bank_probe import _device_summary, _time_predict
from qwen36_clusterbase_residual_probe import _load_all_experts, _parse_sizes, _similarity_scores
from qwen36_pack_single_expert import INT4_BLOCK_SIZE, _package_size_mb
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR
from qwen36_shared_basis_probe import _factor_shared_basis, _parse_basis_counts
from qwen36_sharedbase_residual_probe import _as_analysis, _forward_expert, _metrics, _require_finite

SHARED_BASIS_OUT_DIR = OUT_DIR / "shared_basis_bank_probe"


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


def _pack_basis_bank(basis: np.ndarray) -> np.ndarray:
    basis16 = np.asarray(basis, dtype=np.float16)
    return basis16.reshape(basis16.shape[0] * basis16.shape[1], basis16.shape[2])


def _coeff_const(coeff_row: np.ndarray) -> np.ndarray:
    return np.asarray(coeff_row, dtype=np.float16).reshape(1, -1, 1)


def _build_probe(
    gate_basis: dict[str, np.ndarray | float],
    up_basis: dict[str, np.ndarray | float],
    down_basis: dict[str, np.ndarray | float],
    cluster_size: int,
    d_model: int,
    d_ffn: int,
    batch: int,
    out_path: Path,
) -> ct.models.MLModel:
    basis_count = int(np.asarray(gate_basis["basis"]).shape[0])
    packed_gate = _pack_basis_bank(np.asarray(gate_basis["basis"]))
    packed_up = _pack_basis_bank(np.asarray(up_basis["basis"]))
    packed_down = _pack_basis_bank(np.asarray(down_basis["basis"]))

    gate_coeff = np.asarray(gate_basis["coeff"], dtype=np.float16)
    up_coeff = np.asarray(up_basis["coeff"], dtype=np.float16)
    down_coeff = np.asarray(down_basis["coeff"], dtype=np.float16)

    fp16 = ct.converters.mil.mil.types.fp16
    zeros_basis_ffn = np.zeros((basis_count * d_ffn,), dtype=np.float16)
    zeros_basis_model = np.zeros((basis_count * d_model,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, d_model), dtype=fp16)])
    def prog(x):
        gate_basis_bank = mb.linear(x=x, weight=packed_gate, bias=zeros_basis_ffn, name="gate_basis_bank")
        gate_basis_bank = mb.reshape(x=gate_basis_bank, shape=(batch, basis_count, d_ffn), name="gate_basis_bank_r")
        up_basis_bank = mb.linear(x=x, weight=packed_up, bias=zeros_basis_ffn, name="up_basis_bank")
        up_basis_bank = mb.reshape(x=up_basis_bank, shape=(batch, basis_count, d_ffn), name="up_basis_bank_r")

        hidden_parts = []
        for slot in range(cluster_size):
            gate_mix = mb.mul(x=gate_basis_bank, y=_coeff_const(gate_coeff[slot]), name=f"e{slot}_gate_mix")
            gate = mb.reduce_sum(x=gate_mix, axes=[1], keep_dims=False, name=f"e{slot}_gate")
            up_mix = mb.mul(x=up_basis_bank, y=_coeff_const(up_coeff[slot]), name=f"e{slot}_up_mix")
            up = mb.reduce_sum(x=up_mix, axes=[1], keep_dims=False, name=f"e{slot}_up")
            act = mb.silu(x=gate, name=f"e{slot}_silu_gate")
            hidden_parts.append(mb.mul(x=act, y=up, name=f"e{slot}_hidden"))

        hidden_all = mb.concat(values=hidden_parts, axis=1, name="hidden_concat")
        hidden_all = mb.reshape(x=hidden_all, shape=(batch * cluster_size, d_ffn), name="hidden_flat")
        down_basis_bank = mb.linear(x=hidden_all, weight=packed_down, bias=zeros_basis_model, name="down_basis_bank")
        down_basis_bank = mb.reshape(
            x=down_basis_bank,
            shape=(batch, cluster_size, basis_count, d_model),
            name="down_basis_bank_r",
        )

        outputs = []
        for slot in range(cluster_size):
            down_mix = mb.mul(
                x=mb.slice_by_index(
                    x=down_basis_bank,
                    begin=[0, slot, 0, 0],
                    end=[batch, slot + 1, basis_count, d_model],
                    squeeze_mask=[False, True, False, False],
                    name=f"e{slot}_down_slice",
                ),
                y=_coeff_const(down_coeff[slot]),
                name=f"e{slot}_down_mix",
            )
            outputs.append(mb.reduce_sum(x=down_mix, axes=[1], keep_dims=False, name=f"e{slot}_out"))

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


def _forward_shared_basis_runtime(
    x: np.ndarray,
    gate_basis: dict[str, np.ndarray | float],
    up_basis: dict[str, np.ndarray | float],
    down_basis: dict[str, np.ndarray | float],
) -> np.ndarray:
    gate_bank = np.einsum("bi,loi->blo", _as_analysis(x), _as_analysis(np.asarray(gate_basis["basis"])), optimize=False)
    up_bank = np.einsum("bi,loi->blo", _as_analysis(x), _as_analysis(np.asarray(up_basis["basis"])), optimize=False)
    gate = np.einsum("blo,kl->bko", gate_bank, _as_analysis(np.asarray(gate_basis["coeff"])), optimize=False)
    up = np.einsum("blo,kl->bko", up_bank, _as_analysis(np.asarray(up_basis["coeff"])), optimize=False)
    hidden = gate / (1.0 + np.exp(-gate)) * up
    down_bank = np.einsum("bkf,ldf->bkld", hidden, _as_analysis(np.asarray(down_basis["basis"])), optimize=False)
    return np.einsum("bkld,kl->bkd", down_bank, _as_analysis(np.asarray(down_basis["coeff"])), optimize=False)


def _validate_outputs(
    experts: list[dict[str, np.ndarray]],
    gate_basis: dict[str, np.ndarray | float],
    up_basis: dict[str, np.ndarray | float],
    down_basis: dict[str, np.ndarray | float],
) -> dict[str, object]:
    d_model = int(experts[0]["d_model"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, d_model), dtype=np.float64)
    y_ref = np.stack([_forward_expert(expert, x) for expert in experts], axis=1)
    y_hat = _require_finite("shared_basis_runtime", _forward_shared_basis_runtime(x, gate_basis, up_basis, down_basis))
    overall = _metrics(y_ref.reshape(16, y_ref.shape[1] * y_ref.shape[2]), y_hat.reshape(16, y_hat.shape[1] * y_hat.shape[2]))
    per_expert = [_metrics(y_ref[:, idx, :], y_hat[:, idx, :]) for idx in range(y_ref.shape[1])]
    return {"overall": overall, "per_expert": per_expert}


def _raw_int4_mb(d_model: int, d_ffn: int, basis_count: int) -> float:
    return 3.0 * (d_model * d_ffn * basis_count * 0.5) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--cluster-size", type=int, required=True)
    parser.add_argument("--basis-count", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=SHARED_BASIS_OUT_DIR)
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

    gate_bank = np.stack([_as_analysis(expert["gate"]) for expert in experts], axis=0)
    up_bank = np.stack([_as_analysis(expert["up"]) for expert in experts], axis=0)
    down_bank = np.stack([_as_analysis(expert["down"]) for expert in experts], axis=0)

    gate_basis = _factor_shared_basis(gate_bank, basis_count)
    up_basis = _factor_shared_basis(up_bank, basis_count)
    down_basis = _factor_shared_basis(down_bank, basis_count)
    numerics = _validate_outputs(experts, gate_basis, up_basis, down_basis)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen36_L{args.layer:02d}_cluster{cluster_size}_expert{args.expert_id:03d}_basis{basis_count}_B{args.batch}_int4"
    out_pkg = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    if compiled.exists():
        shutil.rmtree(compiled)

    print("=== Qwen3.6 shared-basis bank probe ===")
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

    t0 = time.perf_counter()
    _build_probe(gate_basis, up_basis, down_basis, cluster_size, d_model, d_ffn, args.batch, out_pkg)
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
            predict_latency_ms = _time_predict(model, d_model, args.batch) * 1e3
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
        "linears": linears,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()