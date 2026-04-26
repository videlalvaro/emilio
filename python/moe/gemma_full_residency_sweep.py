#!/usr/bin/env python3
"""Full Gemma-4 ANE residency sweep for existing compiled artifacts.

This is read-only over model artifacts. It enumerates the validated 90 split
layer shards plus the 8 LM-head shards, audits each compiled `.mlmodelc` with
MLComputePlan, and writes a JSON report under `tmp/`.

Run with Xcode Python/CoreMLTools 9:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
    python/moe/gemma_full_residency_sweep.py \
    --report tmp/gemma_full_residency_sweep.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "python" / "moe" / "out"
LM_HEAD_DIR = OUT / "lm_head_shards"

CONSTISH_OPS = {
    "const",
    "constexpr_lut_to_dense",
    "constexpr_affine_dequantize",
    "constexpr_blockwise_shift_scale",
    "constexpr_sparse_to_dense",
    "constexpr_cast",
}

COMPUTE_HEAVY_OPS = {
    "add",
    "batch_norm",
    "conv",
    "einsum",
    "floor_div",
    "gelu",
    "instance_norm",
    "layer_norm",
    "linear",
    "matmul",
    "mul",
    "pow",
    "real_div",
    "reduce_mean",
    "reduce_sum",
    "relu",
    "rsqrt",
    "sigmoid",
    "silu",
    "softmax",
    "sub",
}

CACHE_DIRS = [
    Path.home() / "Library" / "Caches" / "com.apple.python3" / "com.apple.e5rt.e5bundlecache",
    Path.home() / "Library" / "Caches" / "gemma_ane_smoke" / "com.apple.e5rt.e5bundlecache",
]


def base_op(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def iter_ops(block):
    for op in block.operations:
        yield op
        for nested in getattr(op, "blocks", ()) or ():
            yield from iter_ops(nested)


def device_label(usage) -> str:
    if usage is None:
        return "unknown"
    preferred = getattr(usage, "preferred_compute_device", None)
    if preferred is None:
        preferred = getattr(usage, "preferred", None)
    if preferred is None:
        return "unknown"
    name = preferred.__class__.__name__
    if "Neural" in name or "ANE" in name:
        return "ANE"
    if "GPU" in name:
        return "GPU"
    if "CPU" in name:
        return "CPU"
    return name


def dir_size_gib(path: Path) -> float | None:
    if not path.exists():
        return None
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024 ** 3)


def free_gib(path: Path) -> float:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024 ** 3)


def expected_artifacts() -> list[Path]:
    paths: list[Path] = []
    for layer in range(30):
        paths.extend([
            OUT / f"gemma4_shard{layer}_{layer + 1}_real_attn_q8.mlmodelc",
            OUT / f"gemma4_shard{layer}_{layer + 1}_real_ffn_p0of2_q8.mlmodelc",
            OUT / f"gemma4_shard{layer}_{layer + 1}_real_ffn_p1of2_q8.mlmodelc",
        ])
    paths.extend(LM_HEAD_DIR / f"GemmaLMHead_s{idx}_q8.mlmodelc" for idx in range(8))
    return paths


def audit_artifact(path: Path) -> dict:
    started = time.time()
    plan = MLComputePlan.load_from_path(
        path=str(path),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    counts: Counter[str] = Counter()
    bad: list[dict] = []

    if program is None:
        return {
            "path": str(path.relative_to(ROOT)),
            "status": "FAIL",
            "counts": {"no_program": 1},
            "bad": [{"op": "NO_PROGRAM", "device": "unknown"}],
            "seconds": time.time() - started,
        }

    for function in program.functions.values():
        for op in iter_ops(function.block):
            op_base = base_op(op.operator_name)
            if op_base in CONSTISH_OPS:
                counts[op_base] += 1
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            device = device_label(usage)
            counts[device] += 1
            if op_base in COMPUTE_HEAVY_OPS and device != "ANE":
                bad.append({"op": op.operator_name, "device": device})

    return {
        "path": str(path.relative_to(ROOT)),
        "status": "PASS" if not bad else "FAIL",
        "counts": dict(sorted(counts.items())),
        "bad": bad,
        "seconds": time.time() - started,
    }


def write_report(report_path: Path, report: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2) + "\n")
    tmp_path.replace(report_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "tmp" / "gemma_full_residency_sweep.json")
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    args = parser.parse_args()

    started = time.time()
    artifacts = expected_artifacts()
    missing = [str(path.relative_to(ROOT)) for path in artifacts if not path.exists()]
    free = free_gib(ROOT)
    cache_sizes = {str(path): dir_size_gib(path) for path in CACHE_DIRS}
    report = {
        "kind": "gemma_full_residency_sweep",
        "root": str(ROOT),
        "max_seconds": args.max_seconds,
        "free_gib": free,
        "cache_sizes_gib": cache_sizes,
        "expected_artifacts": len(artifacts),
        "missing": missing,
        "results": [],
        "status": "RUNNING",
    }

    print("=== Gemma full ANE residency sweep ===")
    print(f"artifacts_expected={len(artifacts)} missing={len(missing)} free_gib={free:.1f}")
    for cache_path, size in cache_sizes.items():
        size_text = "missing" if size is None else f"{size:.1f} GiB"
        print(f"cache {cache_path}: {size_text}")

    if missing:
        report["status"] = "FAIL_MISSING"
        write_report(args.report, report)
        print(f"FATAL missing artifacts; report={args.report}", file=sys.stderr)
        return 2
    if free < args.min_free_gib:
        report["status"] = "FAIL_LOW_DISK"
        write_report(args.report, report)
        print(f"FATAL free space {free:.1f} GiB < {args.min_free_gib:.1f} GiB", file=sys.stderr)
        return 3

    all_ok = True
    timed_out = False
    for index, artifact in enumerate(artifacts, start=1):
        elapsed = time.time() - started
        if elapsed >= args.max_seconds:
            timed_out = True
            all_ok = False
            print(f"TIMEOUT before artifact {index}/{len(artifacts)} elapsed={elapsed:.1f}s")
            break
        print(f"[{index:03d}/{len(artifacts)}] {artifact.relative_to(ROOT)}", flush=True)
        try:
            result = audit_artifact(artifact)
        except Exception as exc:  # keep partial progress on CoreML failures
            all_ok = False
            result = {
                "path": str(artifact.relative_to(ROOT)),
                "status": "ERROR",
                "error": repr(exc),
                "counts": {},
                "bad": [{"op": "EXCEPTION", "device": "unknown", "error": repr(exc)}],
                "seconds": 0.0,
            }
        if result["status"] != "PASS":
            all_ok = False
        report["results"].append(result)
        print(
            f"  {result['status']} seconds={result['seconds']:.2f} "
            f"counts={result.get('counts', {})} bad={len(result.get('bad', []))}",
            flush=True,
        )
        write_report(args.report, report)

    report["elapsed_seconds"] = time.time() - started
    report["checked_artifacts"] = len(report["results"])
    report["failed_artifacts"] = [r["path"] for r in report["results"] if r["status"] != "PASS"]
    if timed_out:
        report["status"] = "TIMEOUT_PARTIAL"
    elif all_ok:
        report["status"] = "PASS"
    else:
        report["status"] = "FAIL"
    write_report(args.report, report)

    print(f"OVERALL {report['status']} checked={report['checked_artifacts']}/{len(artifacts)} report={args.report}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())