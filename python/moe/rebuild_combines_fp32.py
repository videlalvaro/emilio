#!/usr/bin/env python3
"""Rebuild only the 30 combine shards with compute_precision=FLOAT32.

Uses the same build_combine_shard() from build_gemma_per_expert.py,
which now uses FLOAT32 internal precision. Overwrites existing
combine_L{i}_fp16.mlmodelc in each per_expert/L{i}/ directory.
"""
import sys, time, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_gemma_per_expert import (
    build_combine_shard, _read_tensor, _load_index,
    D_MODEL, D_FFN, D_DENSE, OUT_BASE,
)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "gemma-4-26b-a4b"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=30)
    args = ap.parse_args()

    idx = _load_index()

    for li in range(args.layer_start, args.layer_end):
        t0 = time.time()
        base = f"model.language_model.layers.{li}"
        def R(k):
            return _read_tensor(idx, f"{base}.{k}")

        pre_ffn_ln_w = R("pre_feedforward_layernorm.weight")
        post_ffn_ln_1_w = R("post_feedforward_layernorm_1.weight")
        post_ffn_ln_2_w = R("post_feedforward_layernorm_2.weight")
        post_ffn_ln_w = R("post_feedforward_layernorm.weight")
        layer_scalar = float(R("layer_scalar").item())
        mlp_gate = R("mlp.gate_proj.weight")
        mlp_up = R("mlp.up_proj.weight")
        mlp_down = R("mlp.down_proj.weight")

        layer_dir = OUT_BASE / f"L{li}"
        combine_name = f"combine_L{li}_fp16"

        # Remove old combine
        import shutil
        old_pkg = layer_dir / f"{combine_name}.mlpackage"
        old_modelc = layer_dir / f"{combine_name}.mlmodelc"
        if old_pkg.exists():
            shutil.rmtree(old_pkg, ignore_errors=True)
        if old_modelc.exists():
            shutil.rmtree(old_modelc, ignore_errors=True)

        pkg, modelc, sz = build_combine_shard(
            pre_ffn_ln_w=pre_ffn_ln_w,
            mlp_gate_w=mlp_gate, mlp_up_w=mlp_up, mlp_down_w=mlp_down,
            post_ffn_ln_1_w=post_ffn_ln_1_w,
            post_ffn_ln_2_w=post_ffn_ln_2_w,
            post_ffn_ln_w=post_ffn_ln_w,
            layer_scalar=layer_scalar,
            rms_eps=1e-6,
            shard_name=combine_name, out_dir=layer_dir,
            quant_bits=0,
        )
        # Clean up mlpackage
        shutil.rmtree(pkg, ignore_errors=True)

        # Update manifest combine entry
        manifest_path = layer_dir / "manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            m["combine"]["mlmodelc"] = modelc
            m["combine"]["pkg_mb"] = sz
            manifest_path.write_text(json.dumps(m, indent=2))

        elapsed = time.time() - t0
        print(f"  L{li}: combine {sz:.1f} MB, {elapsed:.1f}s")

if __name__ == "__main__":
    main()
