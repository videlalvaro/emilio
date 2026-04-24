"""qwen36_pack_expert_bank.py — Phase 0 32-expert MFD bank size probe.

Builds a bank of routed experts as a public CoreML multi-function package and
reports the resulting `.mlpackage` and `.mlmodelc` sizes. This is the first
representative artifact for the banked deployment path in the Qwen plan.

Run with the Xcode Python that has coremltools 9:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/qwen36_pack_expert_bank.py --layer 0 --start-expert 0 --bank-size 32
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

from qwen36_pack_single_expert import (
    EXPERTS_OUT_DIR,
    INT4_BLOCK_SIZE,
    _build_mlpackage,
    _load_expert_weights,
    _package_size_mb,
)
from qwen36_phase0_spec import WEIGHTS_OUT_DIR

BANKS_OUT_DIR = EXPERTS_OUT_DIR / "banks"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--start-expert", type=int, default=0)
    parser.add_argument("--bank-size", type=int, default=32)
    parser.add_argument("--layer-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=BANKS_OUT_DIR)
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    layer_npz = args.layer_npz or (WEIGHTS_OUT_DIR / f"qwen36_layer{args.layer:02d}.npz")
    if not layer_npz.exists():
        raise SystemExit(f"missing layer npz: {layer_npz}")
    if args.bank_size <= 0:
        raise SystemExit("bank_size must be positive")

    end_expert = args.start_expert + args.bank_size - 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen36_L{args.layer:02d}_bank{args.bank_size}_{args.start_expert:03d}_{end_expert:03d}_int4"
    parts_dir = args.out_dir / f"{tag}_parts"
    multi_path = args.out_dir / f"{tag}.mlpackage"
    compiled = args.out_dir / f"{tag}.mlmodelc"
    summary_path = args.out_dir / f"{tag}_summary.json"

    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    if multi_path.exists():
        shutil.rmtree(multi_path)
    if compiled.exists():
        shutil.rmtree(compiled)
    parts_dir.mkdir(parents=True, exist_ok=True)

    print("=== Qwen3.6 expert bank INT4 probe ===")
    print(f"  layer_npz: {layer_npz}")
    print(f"  experts: {args.start_expert}..{end_expert}")
    print(f"  bank_size: {args.bank_size}")
    print(f"  block_size: {INT4_BLOCK_SIZE}")
    print(f"  parts_dir: {parts_dir}")

    part_paths: list[Path] = []
    t0 = time.perf_counter()
    for expert_id in range(args.start_expert, args.start_expert + args.bank_size):
        weights = _load_expert_weights(layer_npz, expert_id)
        part_path = parts_dir / f"expert_{expert_id:03d}.mlpackage"
        _build_mlpackage(weights, part_path)
        part_paths.append(part_path)
    parts_wall_s = time.perf_counter() - t0
    print(f"  built {len(part_paths)} expert parts in {parts_wall_s:.1f}s")

    desc = MultiFunctionDescriptor()
    for expert_id, part_path in zip(range(args.start_expert, args.start_expert + args.bank_size), part_paths):
        desc.add_function(str(part_path), "main", f"expert_{expert_id:03d}")
    desc.default_function_name = f"expert_{args.start_expert:03d}"

    print("  save_multifunction...")
    t0 = time.perf_counter()
    save_multifunction(desc, str(multi_path))
    pack_wall_s = time.perf_counter() - t0
    pkg_mb = _package_size_mb(multi_path)
    print(f"  multifunction save wall: {pack_wall_s:.1f}s")
    print(f"  mlpackage size: {pkg_mb:.2f} MB")

    compiled_mb: float | None = None
    compile_wall_s: float | None = None
    if not args.skip_compile:
        import coremltools as ct

        print("  coremlcompiler...")
        t0 = time.perf_counter()
        compiled = Path(ct.utils.compile_model(str(multi_path), str(compiled)))
        compile_wall_s = time.perf_counter() - t0
        compiled_mb = _package_size_mb(compiled)
        print(f"  compile wall: {compile_wall_s:.1f}s")
        print(f"  mlmodelc size: {compiled_mb:.2f} MB")

    if not args.keep_parts:
        shutil.rmtree(parts_dir)
        print("  removed temporary part packages")

    summary = {
        "layer": args.layer,
        "start_expert": args.start_expert,
        "end_expert": end_expert,
        "bank_size": args.bank_size,
        "layer_npz": str(layer_npz),
        "out_pkg": str(multi_path),
        "compiled": None if args.skip_compile else str(compiled),
        "block_size": INT4_BLOCK_SIZE,
        "parts_wall_s": parts_wall_s,
        "multifunction_save_wall_s": pack_wall_s,
        "compile_wall_s": compile_wall_s,
        "package_size_mb": pkg_mb,
        "compiled_size_mb": compiled_mb,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()