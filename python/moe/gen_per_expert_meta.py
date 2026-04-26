"""Generate per-expert meta JSON for the Gemma ANE Swift driver.

Reads the existing meta (for embed, tokenizer, LM head shards, etc.)
and replaces the 'layers' array with 'per_expert_layers' pointing to
individual expert .mlmodelc + combine .mlmodelc + router binaries.

Usage:
  python3 python/moe/gen_per_expert_meta.py
"""
import json
from pathlib import Path

EXISTING_META = Path("python/moe/out/gemma_swift_head_meta.json")
PER_EXPERT_BASE = Path("python/moe/out/per_expert")
OUT_META = Path("python/moe/out/gemma_per_expert_meta.json")
N_LAYERS = 30
N_EXPERTS = 64
TOP_K = 8


def main():
    with open(EXISTING_META) as f:
        meta = json.load(f)

    # Build per_expert_layers array
    pe_layers = []
    for L in range(N_LAYERS):
        layer_dir = PER_EXPERT_BASE / f"L{L}"
        manifest_path = layer_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Get attn path from existing meta
        attn_path = meta["layers"][L]["attn"]

        # Expert paths (relative from repo root)
        expert_paths = []
        for expert in manifest["experts"]:
            modelc = expert["mlmodelc"]
            # Convert absolute to relative if needed
            if modelc.startswith("/"):
                try:
                    modelc = str(Path(modelc).relative_to(Path.cwd()))
                except ValueError:
                    pass  # keep absolute if not under cwd
            expert_paths.append(modelc)

        # Combine path
        combine_path = manifest["combine"]["mlmodelc"]
        if combine_path.startswith("/"):
            try:
                combine_path = str(Path(combine_path).relative_to(Path.cwd()))
            except ValueError:
                pass

        # Router paths (relative)
        router_proj = str(layer_dir / "router" / "proj_fp16.bin")
        router_scale = str(layer_dir / "router" / "per_expert_scale_fp16.bin")

        pe_layers.append({
            "layer": L,
            "attn": attn_path,
            "n_experts": N_EXPERTS,
            "top_k": TOP_K,
            "experts": expert_paths,
            "combine": combine_path,
            "router_proj_bin": router_proj,
            "router_per_expert_scale_bin": router_scale,
        })

    # Create new meta with per_expert_layers
    new_meta = {k: v for k, v in meta.items() if k != "layers"}
    new_meta["layers"] = meta["layers"]  # keep for backward compat
    new_meta["per_expert_layers"] = pe_layers

    with open(OUT_META, "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"Wrote {OUT_META} ({len(pe_layers)} per-expert layers)")


if __name__ == "__main__":
    main()
