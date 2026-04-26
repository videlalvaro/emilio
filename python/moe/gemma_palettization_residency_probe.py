#!/usr/bin/env python3
"""Probe INT4 palettization residency for representative Gemma shard shapes.

This is a small, synthetic probe. It does not load Gemma weights. It builds a
representative CoreML graph, saves an fp16 baseline, applies 4-bit
per-grouped-channel palettization, audits MLComputePlan placement, and compares
palettized output against the fp16 baseline with cosine similarity.

Run only after gatekeeper approval if the selected shape is expected to take
more than 60 seconds to convert/compile.

Examples:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
    python/moe/gemma_palettization_residency_probe.py --kind ffn

  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
    python/moe/gemma_palettization_residency_probe.py --kind lm-head --lm-vocab-chunk 4096

Use --lm-vocab-chunk 32768 to match the current 8-way LM-head shard shape, but
that is no longer a tiny probe and needs explicit approval.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


D_MODEL = 2816
D_FFN = 704
DEFAULT_LM_VOCAB_CHUNK = 4096

CONST_OPS = {
    "const",
    "constexpr_lut_to_dense",
    "constexpr_affine_dequantize",
    "constexpr_blockwise_shift_scale",
    "constexpr_sparse_to_dense",
    "constexpr_cast",
}

COMPUTE_HEAVY_OPS = {
    "conv",
    "linear",
    "matmul",
    "batch_norm",
    "layer_norm",
    "instance_norm",
    "gelu",
    "silu",
    "relu",
    "softmax",
    "mul",
    "add",
    "sub",
    "reduce_mean",
    "reduce_sum",
    "rsqrt",
    "pow",
}


def _base_op(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    return float(aa @ bb / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))


def _iter_ops(block, func_name: str):
    for op in block.operations:
        yield func_name, op
        for nested in getattr(op, "blocks", ()) or ():
            yield from _iter_ops(nested, func_name)


def _device_label(usage) -> str:
    if usage is None:
        return "unknown"
    preferred = getattr(usage, "preferred", None)
    if preferred is None:
        preferred = getattr(usage, "preferred_compute_device", None)
    if preferred is None:
        return "unknown"
    name = type(preferred).__name__
    if "Neural" in name or "ANE" in name:
        return "ANE"
    if "GPU" in name:
        return "GPU"
    if "CPU" in name:
        return "CPU"
    return name


def build_torch_model(kind: str, d_model: int, d_ffn: int, lm_vocab_chunk: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)

    if kind == "ffn":
        class GemmaFFNProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Conv2d(d_model, d_ffn, 1, bias=False)
                self.up = nn.Conv2d(d_model, d_ffn, 1, bias=False)
                self.down = nn.Conv2d(d_ffn, d_model, 1, bias=False)

            def forward(self, x):
                return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))

        model = GemmaFFNProbe()
        sample = torch.randn(1, d_model, 1, 1, dtype=torch.float16)
        input_name = "x"
        output_name = "hidden"
        shape_note = f"ffn d_model={d_model} d_ffn={d_ffn}"
    elif kind == "lm-head":
        class GemmaLMHeadProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm_weight = nn.Parameter(torch.ones(1, d_model, 1, 1), requires_grad=False)
                self.proj = nn.Conv2d(d_model, lm_vocab_chunk, 1, bias=False)

            def forward(self, x):
                xs = x * (1.0 / (d_model ** 0.5))
                var = (xs * xs).mean(dim=1, keepdim=True)
                normed = xs * torch.rsqrt(var + 1e-6 / d_model)
                return self.proj(normed * self.norm_weight)

        model = GemmaLMHeadProbe()
        sample = torch.randn(1, d_model, 1, 1, dtype=torch.float16)
        input_name = "hidden"
        output_name = "logits"
        shape_note = f"lm-head d_model={d_model} vocab_chunk={lm_vocab_chunk}"
    else:
        raise ValueError(f"unknown kind: {kind}")

    for parameter in model.parameters():
        if parameter.requires_grad:
            nn.init.normal_(parameter, mean=0.0, std=0.02)
    model.half().eval()

    with torch.no_grad():
        traced = torch.jit.trace(model, sample, check_trace=False)
    return traced, sample.numpy(), input_name, output_name, shape_note


def convert_fp16(traced, input_name: str, output_name: str, d_model: int):
    import coremltools as ct

    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=input_name, shape=(1, d_model, 1, 1), dtype=np.float16)],
        outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


def palettize_int4(mlmodel, group_size: int):
    import coremltools as ct

    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            granularity="per_grouped_channel",
            group_size=group_size,
            weight_threshold=0,
        )
    )
    return ct.optimize.coreml.palettize_weights(mlmodel, config)


def save_model(mlmodel, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))


def compile_model(pkg_path: Path, out_dir: Path) -> Path:
    import coremltools as ct

    compiled = out_dir / f"{pkg_path.stem}.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)
    return Path(ct.utils.compile_model(str(pkg_path), str(compiled)))


def audit_residency(compiled_path: Path) -> tuple[bool, Counter, Counter]:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        path=str(compiled_path),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    if program is None:
        raise RuntimeError("compiled model has no MLProgram structure")

    by_device = Counter()
    fallback_by_op = Counter()
    const_by_op = Counter()

    for func_name, func in program.functions.items():
        for _, op in _iter_ops(func.block, func_name):
            base = _base_op(op.operator_name)
            if base in CONST_OPS:
                const_by_op[base] += 1
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            device = _device_label(usage)
            by_device[device] += 1
            if device != "ANE" and base in COMPUTE_HEAVY_OPS:
                fallback_by_op[f"{op.operator_name} -> {device}"] += 1

    return not fallback_by_op, by_device + const_by_op, fallback_by_op


def predict(mlmodel, input_name: str, sample: np.ndarray, output_name: str) -> np.ndarray:
    out = mlmodel.predict({input_name: sample.astype(np.float16)})
    return np.asarray(out[output_name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["ffn", "lm-head"], default="ffn")
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--d-ffn", type=int, default=D_FFN)
    parser.add_argument("--lm-vocab-chunk", type=int, default=DEFAULT_LM_VOCAB_CHUNK)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--cos-floor", type=float, default=0.995)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/gemma_palettization_probe"))
    args = parser.parse_args()

    if args.kind == "lm-head" and args.lm_vocab_chunk >= 32768:
        print("WARNING: full LM-head shard shape is not a tiny probe; get gatekeeper approval first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Gemma palettization residency probe: {args.kind} ===")

    t0 = time.time()
    traced, sample, input_name, output_name, shape_note = build_torch_model(
        args.kind, args.d_model, args.d_ffn, args.lm_vocab_chunk
    )
    print(f"shape: {shape_note}")

    fp16_model = convert_fp16(traced, input_name, output_name, args.d_model)
    fp16_pkg = args.out_dir / f"{args.kind}_fp16.mlpackage"
    save_model(fp16_model, fp16_pkg)
    print(f"fp16 package: {fp16_pkg} ({_dir_size_mb(fp16_pkg):.1f} MB)")

    q4_model = palettize_int4(fp16_model, args.group_size)
    q4_pkg = args.out_dir / f"{args.kind}_int4_palette_g{args.group_size}.mlpackage"
    save_model(q4_model, q4_pkg)
    print(f"q4 palette package: {q4_pkg} ({_dir_size_mb(q4_pkg):.1f} MB)")

    compiled = compile_model(q4_pkg, args.out_dir)
    print(f"compiled: {compiled}")

    residency_ok, counts, fallbacks = audit_residency(compiled)
    print("\nMLComputePlan counts:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")

    fp16_out = predict(fp16_model, input_name, sample, output_name)
    q4_out = predict(q4_model, input_name, sample, output_name)
    cosine = _cosine(fp16_out, q4_out)
    quality_ok = cosine >= args.cos_floor

    print(f"\nlocal golden cosine(fp16, q4_palette) = {cosine:.6f} (floor {args.cos_floor})")
    if fallbacks:
        print("\nNon-ANE compute-heavy ops:")
        for op, count in fallbacks.most_common():
            print(f"  {op}: {count}")

    print(f"\nwall time: {time.time() - t0:.1f}s")
    if residency_ok and quality_ok:
        print("VERDICT: PASS — palettized probe is ANE-resident and matches fp16 local golden")
        print("Next gate before scale-out: golden-validator on the real shard/model artifact.")
        return 0

    print("VERDICT: FAIL")
    if not residency_ok:
        print("  residency gate failed")
    if not quality_ok:
        print("  local golden cosine gate failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())