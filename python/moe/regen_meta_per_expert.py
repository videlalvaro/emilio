#!/usr/bin/env python3
"""Regenerate gemma_swift_head_meta.json with all 30 per_expert_layers."""
import json
from pathlib import Path

ROOT = Path("/Users/alvarovidela/Code/em2")
META = ROOT / "python/moe/out/gemma_swift_head_meta.json"
PE_DIR = ROOT / "python/moe/out/per_expert"

with open(META) as f:
    meta = json.load(f)

# Build layer→attn mapping from packed layers array
attn_by_layer = {l["layer"]: l["attn"] for l in meta["layers"]}

per_expert_layers = []
for li in range(30):
    mf = PE_DIR / f"L{li}" / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"Missing manifest: {mf}")
    with open(mf) as f:
        manifest = json.load(f)

    # Expert paths (absolute)
    experts = [e["mlmodelc"] for e in manifest["experts"]]

    # Combine path (absolute)
    combine = str(PE_DIR / f"L{li}" / f"combine_L{li}_fp16.mlmodelc")

    entry = {
        "layer": li,
        "attn": attn_by_layer[li],
        "n_experts": manifest["n_experts"],
        "top_k": manifest["top_k"],
        "experts": experts,
        "combine": combine,
        "router_proj_bin": manifest["router"]["proj_bin"],
        "router_per_expert_scale_bin": manifest["router"]["per_expert_scale_bin"],
    }
    per_expert_layers.append(entry)

meta["per_expert_layers"] = per_expert_layers

with open(META, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Updated {META} with {len(per_expert_layers)} per_expert_layers")
for e in per_expert_layers[:3]:
    print(f"  L{e['layer']}: {len(e['experts'])} experts, combine={Path(e['combine']).name}")
print(f"  ...")
for e in per_expert_layers[-2:]:
    print(f"  L{e['layer']}: {len(e['experts'])} experts, combine={Path(e['combine']).name}")
