"""qwen36_packed_expert_probe.py — packed executed-op residency probe for Qwen.

This is not the final banked dispatch implementation. It is the smallest
representative executed-op probe after the separate-function MFD bank failed
the ANE residency gate. The goal is to enlarge the executed linears by packing
G experts into one function built from real Qwen layer-0 weights.

Run with the Xcode Python that has coremltools 9:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/qwen36_packed_expert_probe.py --layer 0 --start-expert 0 --pack-size 24
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

from qwen36_pack_single_expert import INT4_BLOCK_SIZE, _package_size_mb
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR

PACKED_OUT_DIR = OUT_DIR / "packed_probe"


def _load_packed_weights(layer_npz: Path, start_expert: int, pack_size: int) -> dict[str, np.ndarray]:
    with np.load(layer_npz, allow_pickle=False) as data:
        gate_up = np.asarray(data["mlp__experts__gate_up_proj"], dtype=np.float16)
        down = np.asarray(data["mlp__experts__down_proj"], dtype=np.float16)

    end_expert = start_expert + pack_size
    if start_expert < 0 or end_expert > gate_up.shape[0]:
        raise SystemExit(
            f"expert slice {start_expert}:{end_expert} out of range 0..{gate_up.shape[0]}"
        )

    d_ffn = gate_up.shape[1] // 2
    packed = gate_up[start_expert:end_expert]
    gate = packed[:, :d_ffn, :].reshape(pack_size * d_ffn, gate_up.shape[2])
    up = packed[:, d_ffn:, :].reshape(pack_size * d_ffn, gate_up.shape[2])
    down_packed = down[start_expert:end_expert].reshape(pack_size * down.shape[1], down.shape[2])

    return {
        "gate": gate,
        "up": up,
        "down": down_packed,
        "d_model": np.array(down.shape[1], dtype=np.int32),
        "d_ffn": np.array(d_ffn, dtype=np.int32),
    }


def _build_probe(weights: dict[str, np.ndarray], pack_size: int, out_path: Path) -> ct.models.MLModel:
    d_model = int(weights["d_model"])
    d_ffn = int(weights["d_ffn"])
    fp16 = ct.converters.mil.mil.types.fp16
    zeros_ffn = np.zeros((pack_size * d_ffn,), dtype=np.float16)
    zeros_down = np.zeros((pack_size * d_model,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, d_model), dtype=fp16)])
    def prog(x):
        gate = mb.linear(x=x, weight=weights["gate"], bias=zeros_ffn, name="gate")
        up = mb.linear(x=x, weight=weights["up"], bias=zeros_ffn, name="up")
        act = mb.silu(x=gate, name="silu_gate")
        hidden = mb.mul(x=act, y=up, name="swiglu_mul")
        hidden = mb.reshape(x=hidden, shape=(1, pack_size, d_ffn), name="hidden_r")
        hidden = mb.reduce_sum(x=hidden, axes=[1], keep_dims=False, name="hidden_sum")
        y = mb.linear(x=hidden, weight=weights["down"], bias=zeros_down, name="down")
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


def _time_predict(model: ct.models.MLModel, d_model: int, n_iter: int = 60, warmup: int = 10) -> float:
    feed = {list(model.input_description)[0]: np.zeros((1, d_model), dtype=np.float32)}
    for _ in range(warmup):
        model.predict(feed)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.predict(feed)
    return (time.perf_counter() - t0) / n_iter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--start-expert", type=int, default=0)
    parser.add_argument("--pack-size", type=int, required=True)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PACKED_OUT_DIR)
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")

    end_expert = args.start_expert + args.pack_size - 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen36_L{args.layer:02d}_packedG{args.pack_size}_{args.start_expert:03d}_{end_expert:03d}_int4"
    out_pkg = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    if compiled.exists():
        shutil.rmtree(compiled)

    print("=== Qwen3.6 packed expert residency probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  experts: {args.start_expert}..{end_expert}")
    print(f"  pack_size: {args.pack_size}")
    print(f"  block_size: {INT4_BLOCK_SIZE}")

    weights = _load_packed_weights(layer_npz, args.start_expert, args.pack_size)
    print(
        "  packed shapes: "
        f"gate={weights['gate'].shape} up={weights['up'].shape} down={weights['down'].shape}"
    )

    t0 = time.perf_counter()
    _build_probe(weights, args.pack_size, out_pkg)
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
    latency_ms = _time_predict(model, int(weights["d_model"])) * 1e3
    devices = _device_summary(out_pkg)
    print(f"  predict latency: {latency_ms:.3f} ms")
    print(f"  linears: {devices}")

    summary = {
        "layer": args.layer,
        "start_expert": args.start_expert,
        "end_expert": end_expert,
        "pack_size": args.pack_size,
        "layer_npz": str(layer_npz),
        "out_pkg": str(out_pkg),
        "compiled": str(compiled),
        "block_size": INT4_BLOCK_SIZE,
        "package_size_mb": pkg_mb,
        "compiled_size_mb": compiled_mb,
        "convert_wall_s": convert_wall_s,
        "compile_wall_s": compile_wall_s,
        "predict_latency_ms": latency_ms,
        "linears": devices,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()