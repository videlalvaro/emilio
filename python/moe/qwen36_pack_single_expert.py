"""qwen36_pack_single_expert.py — Phase 0 single routed-expert INT4 size probe.

Builds exactly one routed expert from the extracted Qwen layer NPZ and reports
the resulting `.mlpackage` and `.mlmodelc` sizes. This is the smallest real
artifact needed to replace size estimates in the Qwen plan.

Run with the Xcode Python that has coremltools 9:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/qwen36_pack_single_expert.py --layer 0 --expert-id 0
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

from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR

INT4_BLOCK_SIZE = 8
EXPERTS_OUT_DIR = OUT_DIR / "experts"


def _package_size_mb(path: Path) -> float:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 1e6


def _load_expert_weights(layer_npz: Path, expert_id: int) -> dict[str, np.ndarray]:
    with np.load(layer_npz, allow_pickle=False) as data:
        gate_up = np.asarray(data["mlp__experts__gate_up_proj"], dtype=np.float16)
        down = np.asarray(data["mlp__experts__down_proj"], dtype=np.float16)

    if gate_up.ndim != 3 or down.ndim != 3:
        raise SystemExit(
            "expected fused expert tensors with rank 3: "
            f"gate_up={gate_up.shape} down={down.shape}"
        )
    if expert_id < 0 or expert_id >= gate_up.shape[0]:
        raise SystemExit(f"expert_id {expert_id} out of range 0..{gate_up.shape[0] - 1}")
    if gate_up.shape[0] != down.shape[0]:
        raise SystemExit(
            "mismatched expert counts: "
            f"gate_up={gate_up.shape[0]} down={down.shape[0]}"
        )
    if gate_up.shape[1] % 2 != 0:
        raise SystemExit(f"expected fused gate/up dimension, got {gate_up.shape}")

    d_ffn = gate_up.shape[1] // 2
    fused = gate_up[expert_id]
    return {
        "gate": fused[:d_ffn],
        "up": fused[d_ffn:],
        "down": down[expert_id],
        "d_model": np.array(down.shape[1], dtype=np.int32),
        "d_ffn": np.array(d_ffn, dtype=np.int32),
        "n_experts": np.array(gate_up.shape[0], dtype=np.int32),
    }


def _build_mlpackage(weights: dict[str, np.ndarray], out_path: Path) -> ct.models.MLModel:
    d_model = int(weights["d_model"])
    fp16 = ct.converters.mil.mil.types.fp16
    zeros_ffn = np.zeros((int(weights["d_ffn"]),), dtype=np.float16)
    zeros_model = np.zeros((d_model,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, d_model), dtype=fp16)])
    def prog(x):
        gate = mb.linear(x=x, weight=weights["gate"], bias=zeros_ffn, name="gate")
        up = mb.linear(x=x, weight=weights["up"], bias=zeros_ffn, name="up")
        act = mb.silu(x=gate, name="silu_gate")
        hidden = mb.mul(x=act, y=up, name="swiglu_mul")
        y = mb.linear(x=hidden, weight=weights["down"], bias=zeros_model, name="down")
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


def _device_summary(compiled_path: Path) -> str:
    try:
        from coremltools.models.compute_plan import MLComputePlan

        plan = MLComputePlan.load_from_path(
            path=str(compiled_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
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
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=EXPERTS_OUT_DIR)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen36_L{args.layer:02d}_expert{args.expert_id:03d}_int4"
    out_pkg = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    if compiled.exists():
        shutil.rmtree(compiled)

    print("=== Qwen3.6 single expert INT4 probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  expert_id: {args.expert_id}")
    print(f"  out_pkg: {out_pkg}")
    print(f"  block_size: {INT4_BLOCK_SIZE}")

    weights = _load_expert_weights(layer_npz, args.expert_id)
    print(
        "  expert shapes: "
        f"gate={weights['gate'].shape} up={weights['up'].shape} down={weights['down'].shape}"
    )

    t0 = time.perf_counter()
    _build_mlpackage(weights, out_pkg)
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
        linears = _device_summary(compiled)
        print(f"  linears: {linears}")
        model = ct.models.MLModel(str(out_pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
        predict_latency_ms = _time_predict(model, int(weights["d_model"])) * 1e3
        print(f"  predict latency: {predict_latency_ms:.3f} ms")

    summary = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "layer_npz": str(layer_npz),
        "out_pkg": str(out_pkg),
        "compiled": None if args.skip_compile else str(compiled),
        "block_size": INT4_BLOCK_SIZE,
        "d_model": int(weights["d_model"]),
        "d_ffn": int(weights["d_ffn"]),
        "n_experts": int(weights["n_experts"]),
        "package_size_mb": pkg_mb,
        "compiled_size_mb": compiled_mb,
        "convert_wall_s": convert_wall_s,
        "compile_wall_s": compile_wall_s,
        "linears": linears,
        "predict_latency_ms": predict_latency_ms,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()