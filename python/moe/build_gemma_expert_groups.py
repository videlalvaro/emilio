"""build_gemma_expert_groups.py — Build expert-GROUP CoreML shards for Gemma-4.

Each group shard packs G_SIZE=16 experts into one CoreML model:
  - Combined gate Conv2d: (D_MODEL → G_SIZE*D_FFN)
  - Combined up Conv2d: (D_MODEL → G_SIZE*D_FFN)
  - GELU·gate·up
  - Grouped down Conv2d: (G_SIZE*D_FFN → G_SIZE*D_MODEL), groups=G_SIZE
  - Reshape + weighted-sum by expert_weights → (1,D_MODEL,1,1)

Inputs: x (1,D_MODEL,1,1), expert_weights (1,G_SIZE,1,1) [0 for inactive]
Output: weighted_sum (1,D_MODEL,1,1)

G=8 groups of 16 → 8 shards/layer × 30 layers = 240 FFN shards loaded at startup.
Per-token: only ~5.4 groups run (those containing active top-8 experts).
Each shard ~190 MB (under 250 MB ANE shard limit).

Run with Xcode python3:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      python/moe/build_gemma_expert_groups.py --layer 0

  All 30 layers:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      python/moe/build_gemma_expert_groups.py --layer-start 0 --layer-end 30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

# ── Constants (must match build_gemma_per_expert.py) ──
D_MODEL = 2816
D_FFN = 704          # moe_intermediate per expert
D_DENSE = 2112       # dense MLP intermediate
N_EXPERTS_ALL = 128
TOP_K = 8
G_SIZE = 16          # experts per group shard (GEMMA_ANE_RESEARCH.md T1.4: G=16→3.65 packs)
N_GROUPS = N_EXPERTS_ALL // G_SIZE  # 8 (ANE_CHAIN_SCHEMA.md: ~250 MB shard limit)

MODEL_DIR = Path("models/gemma-4-26b-a4b")
PE_BASE = Path("python/moe/out/per_expert")   # existing per-expert output
OUT_BASE = Path("python/moe/out/expert_groups")


def _load_index():
    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _read_tensor(idx, key):
    """Read a tensor from HF safetensors, convert bf16→fp32."""
    import torch
    from safetensors import safe_open
    fname = idx[key]
    with safe_open(MODEL_DIR / fname, framework="pt") as f:
        return f.get_tensor(key).to(torch.float32).contiguous().numpy()


def _compile_mlpackage(pkg_path: str) -> str:
    abs_path = str(Path(pkg_path).resolve())
    out_dir = str(Path(pkg_path).parent.resolve())
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", abs_path, out_dir],
        capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed: {result.stderr}")
    return abs_path.replace(".mlpackage", ".mlmodelc")


def _pkg_size_mb(path: str) -> float:
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def _disk_free_gb() -> float:
    st = os.statvfs("/")
    return (st.f_bavail * st.f_frsize) / (1024**3)


def _compute_gate_scale(gate_up_all: np.ndarray, pre_ln2_scale: np.ndarray) -> float:
    """Compute fp16-safety gate_scale (mirrors build_gemma_per_expert.compute_gate_scale)."""
    gate_all = gate_up_all[:, :D_FFN, :]  # (128, 704, 2816)
    ln2_s = pre_ln2_scale.astype(np.float32)
    gate_fused = gate_all * ln2_s[np.newaxis, np.newaxis, :]
    max_norm = float(np.abs(gate_fused).max())
    if max_norm > 16:
        return float(math.sqrt(256.0 / (max_norm * max_norm)))
    return 1.0


# ── Import combine shard builder from per-expert script ──
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_gemma_per_expert import build_combine_shard


def build_expert_group(
    group_idx: int,
    layer_idx: int,
    gate_weights: np.ndarray,     # (G_SIZE, D_FFN, D_MODEL) fp32, pre_ln2 fused
    up_weights: np.ndarray,       # (G_SIZE, D_FFN, D_MODEL) fp32, pre_ln2 fused
    down_weights: np.ndarray,     # (G_SIZE, D_MODEL, D_FFN) fp32
    gate_scale: float,
    shard_name: str,
    out_dir: Path,
):
    """Build one expert-group shard (16 experts in one CoreML model).
    Returns (pkg_path, modelc_path, size_mb)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import coremltools as ct

    g_size = gate_weights.shape[0]  # G_SIZE, captured for trace

    class ExpertGroupModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Gate: (D_MODEL) → (g_size*D_FFN) — all experts' gate projections combined
            self.gate_conv = nn.Conv2d(D_MODEL, g_size * D_FFN, 1, bias=False)
            # Up: (D_MODEL) → (g_size*D_FFN)
            self.up_conv = nn.Conv2d(D_MODEL, g_size * D_FFN, 1, bias=False)
            # Down: grouped conv, each group = one expert's down projection
            self.down_conv = nn.Conv2d(
                g_size * D_FFN, g_size * D_MODEL, 1,
                bias=False, groups=g_size)
            self.register_buffer("_gate_scale",
                                 torch.tensor(gate_scale, dtype=torch.float16))

            # Pack gate weights: (g_size, D_FFN, D_MODEL) → (g_size*D_FFN, D_MODEL, 1, 1)
            self.gate_conv.weight = nn.Parameter(
                torch.tensor(gate_weights.reshape(g_size * D_FFN, D_MODEL),
                             dtype=torch.float16).reshape(g_size * D_FFN, D_MODEL, 1, 1),
                requires_grad=False)
            self.up_conv.weight = nn.Parameter(
                torch.tensor(up_weights.reshape(g_size * D_FFN, D_MODEL),
                             dtype=torch.float16).reshape(g_size * D_FFN, D_MODEL, 1, 1),
                requires_grad=False)
            # Grouped conv weight: (g_size*D_MODEL, D_FFN, 1, 1)
            # Group g uses weight[g*D_MODEL:(g+1)*D_MODEL, :, :, :]
            self.down_conv.weight = nn.Parameter(
                torch.tensor(down_weights.reshape(g_size * D_MODEL, D_FFN),
                             dtype=torch.float16).reshape(g_size * D_MODEL, D_FFN, 1, 1),
                requires_grad=False)

        def forward(self, x, expert_weights):
            """
            x: (1, D_MODEL, 1, 1) fp16
            expert_weights: (1, g_size, 1, 1) fp16 — routing weights (0 for inactive)
            Returns: (1, D_MODEL, 1, 1) fp16 — weighted sum of active expert outputs
            """
            gate = F.gelu(self.gate_conv(x), approximate="tanh") * self._gate_scale
            up = self.up_conv(x)
            h = gate * up                      # (1, g_size*D_FFN, 1, 1)
            h = self.down_conv(h)              # (1, g_size*D_MODEL, 1, 1)

            # Reshape to separate expert dim, weight, and sum
            h = h.reshape(1, g_size, D_MODEL, 1)        # (1, g_size, D_MODEL, 1)
            h = h * expert_weights                        # broadcast (1, g_size, 1, 1)
            h = h.sum(dim=1, keepdim=True)                # (1, 1, D_MODEL, 1)
            h = h.reshape(1, D_MODEL, 1, 1)
            return h

    model = ExpertGroupModel()
    model.half()
    model.eval()

    ex_x = torch.randn(1, D_MODEL, 1, 1, dtype=torch.float16)
    ex_w = torch.zeros(1, g_size, 1, 1, dtype=torch.float16)
    ex_w[0, 0, 0, 0] = 0.2  # one active expert
    with torch.no_grad():
        traced = torch.jit.trace(model, (ex_x, ex_w))

    ct_inputs = [
        ct.TensorType(name="x", shape=(1, D_MODEL, 1, 1), dtype=np.float16),
        ct.TensorType(name="expert_weights", shape=(1, g_size, 1, 1), dtype=np.float16),
    ]
    ct_outputs = [ct.TensorType(name="group_out", dtype=np.float16)]

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    # INT8 per-tensor weight quantization — production baseline for ANE residency
    # (FP16 dense linears risk CPU fallback; INT8 keeps ops in the 12-50 MB sweet spot)
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
    )
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    opt_config = OptimizationConfig(global_config=op_config)
    mlmodel = linear_quantize_weights(mlmodel, config=opt_config)

    pkg_path = str(out_dir / f"{shard_name}.mlpackage")
    mlmodel.save(pkg_path)
    sz = _pkg_size_mb(pkg_path)
    modelc_path = _compile_mlpackage(pkg_path)
    return pkg_path, modelc_path, sz


def process_layer(layer_idx: int, idx: dict, delete_pkg: bool = True):
    """Build 8 expert-group shards + combine + router for one layer."""
    t0 = time.time()

    free = _disk_free_gb()
    print(f"\n  Disk free: {free:.1f} GB")
    if free < 10:
        raise RuntimeError(f"ABORT: only {free:.1f} GB free (need >10 GB safety margin)")

    base = f"model.language_model.layers.{layer_idx}"
    def R(k):
        return _read_tensor(idx, f"{base}.{k}")

    # ── Load expert weights ──
    gate_up_all = R("experts.gate_up_proj")    # (128, 1408, 2816)
    down_all = R("experts.down_proj")          # (128, 2816, 704)

    # ── Norms & scalars ──
    pre_ffn_ln_2_w = R("pre_feedforward_layernorm_2.weight")  # (D_MODEL,)
    pre_ffn_ln_w = R("pre_feedforward_layernorm.weight")
    post_ffn_ln_1_w = R("post_feedforward_layernorm_1.weight")
    post_ffn_ln_2_w = R("post_feedforward_layernorm_2.weight")
    post_ffn_ln_w = R("post_feedforward_layernorm.weight")
    layer_scalar = float(R("layer_scalar").item())
    rms_eps = 1e-6

    # Dense MLP for combine shard
    mlp_gate = R("mlp.gate_proj.weight")       # (2112, 2816)
    mlp_up = R("mlp.up_proj.weight")           # (2112, 2816)
    mlp_down = R("mlp.down_proj.weight")       # (2816, 2112)

    # Router weights
    router_proj = R("router.proj.weight")       # (128, 2816)
    router_scale = R("router.scale")            # (2816,)
    router_per_exp = R("router.per_expert_scale")  # (128,)

    ln2_s = pre_ffn_ln_2_w.astype(np.float32)
    gate_scale = _compute_gate_scale(gate_up_all, pre_ffn_ln_2_w)

    # Fuse router scale into proj
    scalar_root = D_MODEL ** -0.5
    r_scale = (router_scale * scalar_root).astype(np.float32)
    router_proj_fused = router_proj * r_scale[np.newaxis, :]  # (128, 2816)

    layer_dir = OUT_BASE / f"L{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Layer {layer_idx}: {N_GROUPS} groups of {G_SIZE} experts (fp16)")
    print(f"  gate_scale={gate_scale:.4f}  layer_scalar={layer_scalar:.4f}")
    print(f"{'='*60}")

    # ── Save router bins (fp16 + fp32) ──
    router_dir = layer_dir / "router"
    router_dir.mkdir(exist_ok=True)
    router_proj_fused.astype(np.float16).tofile(router_dir / "proj_fp16.bin")
    router_per_exp.astype(np.float16).tofile(router_dir / "per_expert_scale_fp16.bin")
    router_proj_fused.astype(np.float32).tofile(router_dir / "proj_fp32.bin")
    router_per_exp.astype(np.float32).tofile(router_dir / "per_expert_scale_fp32.bin")
    print(f"  Router: saved fp16+fp32 bins")

    # ── Build 8 expert-group shards ──
    group_results = []
    for g in range(N_GROUPS):
        expert_start = g * G_SIZE
        expert_end = expert_start + G_SIZE

        gate_up_group = gate_up_all[expert_start:expert_end]  # (16, 1408, 2816)
        gate_group = gate_up_group[:, :D_FFN, :]              # (16, 704, 2816)
        up_group = gate_up_group[:, D_FFN:, :]                # (16, 704, 2816)
        down_group = down_all[expert_start:expert_end]         # (16, 2816, 704)

        # Fuse pre_ln2 scale into gate/up
        gate_fused = gate_group * ln2_s[np.newaxis, np.newaxis, :]
        up_fused = up_group * ln2_s[np.newaxis, np.newaxis, :]

        name = f"group_L{layer_idx}_g{g}_q8"
        pkg, modelc, sz = build_expert_group(
            group_idx=g,
            layer_idx=layer_idx,
            gate_weights=gate_fused,
            up_weights=up_fused,
            down_weights=down_group,
            gate_scale=gate_scale,
            shard_name=name,
            out_dir=layer_dir,
        )
        group_results.append({
            "group": g,
            "expert_start": expert_start,
            "expert_end": expert_end,
            "modelc": str(modelc),
            "pkg_mb": round(sz, 1),
        })
        if delete_pkg:
            shutil.rmtree(pkg, ignore_errors=True)
        print(f"    group {g} (e{expert_start:03d}-e{expert_end-1:03d}): {sz:.1f} MB")

        free = _disk_free_gb()
        if free < 5:
            raise RuntimeError(f"ABORT: disk critically low ({free:.1f} GB free)")

    # ── Build combine shard ──
    combine_name = f"combine_L{layer_idx}_q8"
    cpkg, cmodelc, csz = build_combine_shard(
        pre_ffn_ln_w=pre_ffn_ln_w,
        mlp_gate_w=mlp_gate, mlp_up_w=mlp_up, mlp_down_w=mlp_down,
        post_ffn_ln_1_w=post_ffn_ln_1_w,
        post_ffn_ln_2_w=post_ffn_ln_2_w,
        post_ffn_ln_w=post_ffn_ln_w,
        layer_scalar=layer_scalar,
        rms_eps=rms_eps,
        shard_name=combine_name, out_dir=layer_dir,
        quant_bits=8,  # INT8 production baseline
    )
    if delete_pkg:
        shutil.rmtree(cpkg, ignore_errors=True)
    print(f"  Combine: {csz:.1f} MB")

    elapsed = time.time() - t0

    # ── Write manifest ──
    manifest = {
        "layer": layer_idx,
        "n_groups": N_GROUPS,
        "group_size": G_SIZE,
        "n_experts": N_EXPERTS_ALL,
        "top_k": TOP_K,
        "d_model": D_MODEL,
        "d_ffn": D_FFN,
        "gate_scale": gate_scale,
        "layer_scalar": layer_scalar,
        "rms_eps": rms_eps,
        "groups": group_results,
        "combine": {"mlmodelc": str(cmodelc), "pkg_mb": round(csz, 1)},
        "router": {
            "proj_fp32": str(router_dir / "proj_fp32.bin"),
            "per_expert_scale_fp32": str(router_dir / "per_expert_scale_fp32.bin"),
            "n_experts": N_EXPERTS_ALL,
            "d_model": D_MODEL,
        },
        "build_time_s": round(elapsed, 1),
    }
    mpath = layer_dir / "manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    free = _disk_free_gb()
    print(f"  Layer {layer_idx} done in {elapsed:.1f}s  ({free:.1f} GB free)")
    return manifest


def write_global_meta(manifests: list, meta_src_path: str):
    """Write global meta JSON for Swift runtime (references existing attn shards)."""
    # Load existing meta for attention shard paths
    with open(meta_src_path) as f:
        src = json.load(f)

    meta = {
        "mode": "expert_groups",
        "n_groups": N_GROUPS,
        "group_size": G_SIZE,
        "n_experts": N_EXPERTS_ALL,
        "top_k": TOP_K,
        "layers": [],
    }
    # Copy non-layer fields from source
    for k in src:
        if k not in ("layers", "per_expert_layers"):
            meta[k] = src[k]

    src_layers = {l["layer"]: l for l in src.get("layers", [])}

    for m in manifests:
        li = m["layer"]
        entry = {
            "layer": li,
            "groups": m["groups"],
            "combine": m["combine"],
            "router": m["router"],
            "gate_scale": m["gate_scale"],
        }
        # Copy attention shard path from source meta
        if li in src_layers:
            entry["attn"] = src_layers[li].get("attn", "")
        meta["layers"].append(entry)

    out_path = OUT_BASE / "expert_groups_meta.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nGlobal meta: {out_path}")
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=30)
    ap.add_argument("--keep-pkg", action="store_true",
                    help="Keep .mlpackage files (default: delete)")
    ap.add_argument("--meta-src", default="python/moe/out/gemma_swift_head_meta.json",
                    help="Source meta JSON (for attn shard paths)")
    args = ap.parse_args()

    # Disk check
    free = _disk_free_gb()
    layers = 1 if args.layer is not None else (args.layer_end - args.layer_start)
    est_gb = layers * N_GROUPS * 0.10 + layers * 0.04  # INT8 groups (~95 MB) + combines
    print(f"Disk: {free:.1f} GB free, estimated need: {est_gb:.1f} GB")
    if free < est_gb + 10:
        print(f"WARNING: tight on disk ({free:.1f} GB free, need ~{est_gb+10:.0f} GB)")
        if free < est_gb + 2:
            raise RuntimeError("ABORT: not enough disk space")

    idx = _load_index()
    manifests = []

    if args.layer is not None:
        m = process_layer(args.layer, idx, delete_pkg=not args.keep_pkg)
        manifests.append(m)
    else:
        for L in range(args.layer_start, args.layer_end):
            m = process_layer(L, idx, delete_pkg=not args.keep_pkg)
            manifests.append(m)

    # Write global meta if building all layers
    if len(manifests) > 1 or args.layer is None:
        write_global_meta(manifests, args.meta_src)


if __name__ == "__main__":
    main()
