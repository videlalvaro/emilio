"""qwen36_conv_expert_probe.py — single-expert conv-as-linear ANE probe for Qwen.

This tests the strongest operator-family hypothesis from the PF handoff: the
successful per-expert path was authored as 1x1 Conv2d rather than MIL linear.
The probe keeps the same real Qwen weights and INT4 quantization, but changes
only the authoring geometry.

Run with the Xcode Python that has coremltools + torch:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/qwen36_conv_expert_probe.py --layer 0 --expert-id 0
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

from qwen36_pack_single_expert import INT4_BLOCK_SIZE, _load_expert_weights, _package_size_mb
from qwen36_phase0_spec import OUT_DIR, WEIGHTS_OUT_DIR

CONV_OUT_DIR = OUT_DIR / "conv_probe"


class ExpertConv(nn.Module):
    def __init__(self, weights: dict[str, np.ndarray]):
        super().__init__()
        d_model = int(weights["d_model"])
        d_ffn = int(weights["d_ffn"])

        gate_w = np.asarray(weights["gate"], dtype=np.float32).reshape(d_ffn, d_model, 1, 1)
        up_w = np.asarray(weights["up"], dtype=np.float32).reshape(d_ffn, d_model, 1, 1)
        down_w = np.asarray(weights["down"], dtype=np.float32).reshape(d_model, d_ffn, 1, 1)
        zeros_ffn = np.zeros((d_ffn,), dtype=np.float32)
        zeros_model = np.zeros((d_model,), dtype=np.float32)

        self.gate = nn.Conv2d(d_model, d_ffn, 1, bias=True)
        self.up = nn.Conv2d(d_model, d_ffn, 1, bias=True)
        self.down = nn.Conv2d(d_ffn, d_model, 1, bias=True)

        self.gate.weight = nn.Parameter(torch.from_numpy(gate_w), requires_grad=False)
        self.gate.bias = nn.Parameter(torch.from_numpy(zeros_ffn), requires_grad=False)
        self.up.weight = nn.Parameter(torch.from_numpy(up_w), requires_grad=False)
        self.up.bias = nn.Parameter(torch.from_numpy(zeros_ffn), requires_grad=False)
        self.down.weight = nn.Parameter(torch.from_numpy(down_w), requires_grad=False)
        self.down.bias = nn.Parameter(torch.from_numpy(zeros_model), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        up = self.up(x)
        hidden = F.silu(gate) * up
        return self.down(hidden)


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
                if op.operator_name not in ("ios18.conv", "conv", "ios18.convolution"):
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
    feed = {list(model.input_description)[0]: np.zeros((batch, d_model, 1, 1), dtype=np.float32)}
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
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=CONV_OUT_DIR)
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")
    if args.batch <= 0:
        raise SystemExit("batch must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen36_L{args.layer:02d}_expert{args.expert_id:03d}_B{args.batch}_conv_int4"
    out_pkg = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    if compiled.exists():
        shutil.rmtree(compiled)

    print("=== Qwen3.6 single expert conv probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  expert_id: {args.expert_id}")
    print(f"  batch: {args.batch}")
    print(f"  out_pkg: {out_pkg}")
    print(f"  block_size: {INT4_BLOCK_SIZE}")

    weights = _load_expert_weights(layer_npz, args.expert_id)
    d_model = int(weights["d_model"])
    d_ffn = int(weights["d_ffn"])
    print(
        "  expert shapes: "
        f"gate={weights['gate'].shape} up={weights['up'].shape} down={weights['down'].shape}"
    )

    mod = ExpertConv(weights).eval()
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

    print("  coremlcompiler...")
    t0 = time.perf_counter()
    compiled = Path(ct.utils.compile_model(str(out_pkg), str(compiled)))
    compile_wall_s = time.perf_counter() - t0
    compiled_mb = _package_size_mb(compiled)
    print(f"  compile wall: {compile_wall_s:.1f}s")
    print(f"  mlmodelc size: {compiled_mb:.2f} MB")

    model = ct.models.MLModel(str(out_pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    latency_ms = _time_predict(model, args.batch, d_model) * 1e3
    devices = _device_summary(out_pkg)
    print(f"  predict latency: {latency_ms:.3f} ms")
    print(f"  convs: {devices}")

    summary = {
        "layer": args.layer,
        "expert_id": args.expert_id,
        "batch": args.batch,
        "layer_npz": str(layer_npz),
        "out_pkg": str(out_pkg),
        "compiled": str(compiled),
        "block_size": INT4_BLOCK_SIZE,
        "d_model": d_model,
        "d_ffn": d_ffn,
        "package_size_mb": pkg_mb,
        "compiled_size_mb": compiled_mb,
        "convert_wall_s": convert_wall_s,
        "compile_wall_s": compile_wall_s,
        "predict_latency_ms": latency_ms,
        "convs": devices,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()