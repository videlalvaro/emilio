"""build_gemma_per_expert.py — Build individual per-expert CoreML shards for Gemma-4.

Each expert: Conv2d gate + Conv2d up + GELU·gate·up + Conv2d down → (1,D,1,1)
Pre-ffn_ln_2 scale fused into gate/up weights. FP16 by default (--quant-bits 8 for INT8).

Also builds a per-layer "combine" shard:
  dense MLP + post-norms + residual + layer_scalar

And extracts router weights as binary files for Swift CPU-side routing.

Run with Xcode python3 (has coremltools 9):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      python/moe/build_gemma_per_expert.py --layer 0
  
  All 30 layers, FP16 (default):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      python/moe/build_gemma_per_expert.py --layer-start 0 --layer-end 30

  REAP-pruned 64 experts, INT8:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \\
      python/moe/build_gemma_per_expert.py --layer 0 --n-experts 64 --quant-bits 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

# ── Constants (must match gemma_to_ane.py) ──
D_MODEL = 2816
D_FFN = 704          # moe_intermediate
D_DENSE = 2112       # dense MLP intermediate
N_EXPERTS_ALL = 128  # total experts per layer
TOP_K = 8
MODEL_DIR = Path("models/gemma-4-26b-a4b")
REAP_MASK = Path("python/moe/out/gemma_reap_mask.npz")
OUT_BASE = Path("python/moe/out/per_expert")

# safe-norm peephole constant (same as build_gemma_lm_head_shards.py)
NORM_K = D_MODEL ** 0.5  


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
    """Compile .mlpackage → .mlmodelc."""
    abs_path = str(Path(pkg_path).resolve())
    out_dir = str(Path(pkg_path).parent.resolve())
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", abs_path, out_dir],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed: {result.stderr}")
    modelc = abs_path.replace(".mlpackage", ".mlmodelc")
    return modelc


def _pkg_size_mb(path: str) -> float:
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def build_single_expert(
    gate_w: np.ndarray,   # (D_FFN, D_MODEL) fp32
    up_w: np.ndarray,     # (D_FFN, D_MODEL) fp32  
    down_w: np.ndarray,   # (D_MODEL, D_FFN) fp32
    pre_ln2_scale: np.ndarray,  # (D_MODEL,) fp32 — fuse into gate/up
    gate_scale: float,    # post-activation scale (from fuse_norm_scales_for_ane)
    shard_name: str,
    out_dir: Path,
    quant_bits: int = 8,
):
    """Build one expert as Conv2d CoreML model. Returns (pkg_path, modelc_path)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
    )

    # Fuse pre_ffn_ln_2 scale into gate/up weights
    # HF safetensor stores γ (the full RMSNorm scale, Google convention init=1).
    # The packed model's _set_ln stores (γ-1) then applies (1+w)=γ.
    # We must use γ directly, NOT (1+γ).
    ln2_s = pre_ln2_scale.astype(np.float32)  # (D_MODEL,)  — γ directly
    gate_fused = gate_w * ln2_s[np.newaxis, :]  # (D_FFN, D_MODEL)
    up_fused = up_w * ln2_s[np.newaxis, :]      # (D_FFN, D_MODEL)

    class SingleExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_conv = nn.Conv2d(D_MODEL, D_FFN, 1, bias=False)
            self.up_conv = nn.Conv2d(D_MODEL, D_FFN, 1, bias=False)
            self.down_conv = nn.Conv2d(D_FFN, D_MODEL, 1, bias=False)
            self.register_buffer("_gate_scale",
                                 torch.tensor(gate_scale, dtype=torch.float16))

            # Set weights: Conv2d weight shape is (out_ch, in_ch, 1, 1)
            self.gate_conv.weight = nn.Parameter(
                torch.tensor(gate_fused, dtype=torch.float16).reshape(D_FFN, D_MODEL, 1, 1),
                requires_grad=False)
            self.up_conv.weight = nn.Parameter(
                torch.tensor(up_fused, dtype=torch.float16).reshape(D_FFN, D_MODEL, 1, 1),
                requires_grad=False)
            self.down_conv.weight = nn.Parameter(
                torch.tensor(down_w, dtype=torch.float16).reshape(D_MODEL, D_FFN, 1, 1),
                requires_grad=False)

        def forward(self, x):
            """x: (1, D_MODEL, 1, 1) fp16 → (1, D_MODEL, 1, 1) fp16"""
            g = F.gelu(self.gate_conv(x), approximate="tanh") * self._gate_scale
            u = self.up_conv(x)
            return self.down_conv(g * u)

    model = SingleExpert()
    model.half()
    model.eval()

    example = torch.randn(1, D_MODEL, 1, 1, dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    ct_inputs = [ct.TensorType(name="x", shape=(1, D_MODEL, 1, 1), dtype=np.float16)]
    ct_outputs = [ct.TensorType(name="expert_out", dtype=np.float16)]

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if quant_bits == 8:
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        opt_config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=opt_config)
    # quant_bits == 0 → FP16, no quantization

    pkg_path = str(out_dir / f"{shard_name}.mlpackage")
    mlmodel.save(pkg_path)
    sz = _pkg_size_mb(pkg_path)

    modelc_path = _compile_mlpackage(pkg_path)
    return pkg_path, modelc_path, sz


def build_combine_shard(
    pre_ffn_ln_w: np.ndarray,      # (D_MODEL,)
    mlp_gate_w: np.ndarray,         # (D_DENSE, D_MODEL)
    mlp_up_w: np.ndarray,           # (D_DENSE, D_MODEL)
    mlp_down_w: np.ndarray,         # (D_MODEL, D_DENSE)
    post_ffn_ln_1_w: np.ndarray,    # (D_MODEL,)
    post_ffn_ln_2_w: np.ndarray,    # (D_MODEL,)
    post_ffn_ln_w: np.ndarray,      # (D_MODEL,)
    layer_scalar: float,
    rms_eps: float,
    shard_name: str,
    out_dir: Path,
    quant_bits: int = 0,
):
    """Build the combine shard using Conv2d (rank-4) for ANE placement.
    
    Inputs: x (1,D_MODEL,1,1), moe_sum (1,D_MODEL,1,1)
    Output: hidden (1,D_MODEL,1,1)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
    )

    K = D_MODEL ** 0.5
    EPS_SCALED = rms_eps / (K * K)

    class RMSNormConv(nn.Module):
        """RMSNorm operating on (1,C,1,1) rank-4 tensors for ANE."""
        def __init__(self, weight_np):
            super().__init__()
            # HF stores γ (full scale, Google convention).
            # Packed model: _set_ln stores (γ-1), module applies (1+w)=γ.
            # Here we use γ directly as the scale.
            self.weight = nn.Parameter(
                torch.tensor(weight_np.astype(np.float32), dtype=torch.float16
                             ).reshape(1, D_MODEL, 1, 1),
                requires_grad=False)

        def forward(self, x):
            # x: (1,C,1,1). Scale then RMS on channel dim.
            xs = x * (1.0 / K)
            # var over channel dim: mean of squares
            var = (xs * xs).mean(dim=1, keepdim=True)
            return (xs * torch.rsqrt(var + EPS_SCALED)) * self.weight

    class CombineShardConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_ffn_ln = RMSNormConv(pre_ffn_ln_w)
            self.gate_conv = nn.Conv2d(D_MODEL, D_DENSE, 1, bias=False)
            self.up_conv = nn.Conv2d(D_MODEL, D_DENSE, 1, bias=False)
            self.down_conv = nn.Conv2d(D_DENSE, D_MODEL, 1, bias=False)
            self.post_ffn_ln_1 = RMSNormConv(post_ffn_ln_1_w)
            self.post_ffn_ln_2 = RMSNormConv(post_ffn_ln_2_w)
            self.post_ffn_ln = RMSNormConv(post_ffn_ln_w)
            self.register_buffer("_layer_scalar",
                                 torch.tensor(layer_scalar, dtype=torch.float16
                                              ).reshape(1, 1, 1, 1))

            self.gate_conv.weight = nn.Parameter(
                torch.tensor(mlp_gate_w, dtype=torch.float16).reshape(D_DENSE, D_MODEL, 1, 1),
                requires_grad=False)
            self.up_conv.weight = nn.Parameter(
                torch.tensor(mlp_up_w, dtype=torch.float16).reshape(D_DENSE, D_MODEL, 1, 1),
                requires_grad=False)
            self.down_conv.weight = nn.Parameter(
                torch.tensor(mlp_down_w, dtype=torch.float16).reshape(D_MODEL, D_DENSE, 1, 1),
                requires_grad=False)

        def forward(self, x, moe_sum):
            """x, moe_sum: (1,D_MODEL,1,1). Returns hidden (1,D_MODEL,1,1)."""
            pre = self.pre_ffn_ln(x)
            g = F.gelu(self.gate_conv(pre), approximate="tanh")
            u = self.up_conv(pre)
            dense = self.down_conv(g * u)
            h1 = self.post_ffn_ln_1(dense)
            h2 = self.post_ffn_ln_2(moe_sum)
            x_ffn = self.post_ffn_ln(h1 + h2)
            return (x + x_ffn) * self._layer_scalar

    model = CombineShardConv()
    model.half()
    model.eval()

    ex_x = torch.randn(1, D_MODEL, 1, 1, dtype=torch.float16)
    ex_m = torch.randn(1, D_MODEL, 1, 1, dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model, (ex_x, ex_m))

    ct_inputs = [
        ct.TensorType(name="x", shape=(1, D_MODEL, 1, 1), dtype=np.float16),
        ct.TensorType(name="moe_sum", shape=(1, D_MODEL, 1, 1), dtype=np.float16),
    ]
    ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if quant_bits == 8:
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        opt_config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=opt_config)

    pkg_path = str(out_dir / f"{shard_name}.mlpackage")
    mlmodel.save(pkg_path)
    sz = _pkg_size_mb(pkg_path)
    modelc_path = _compile_mlpackage(pkg_path)
    return pkg_path, modelc_path, sz


def compute_gate_scale(gate_up_all: np.ndarray, pre_ln2_scale: np.ndarray,
                       keep_idx: np.ndarray) -> float:
    """Compute the gate_scale for an expert pack to prevent fp16 overflow.
    
    Mirrors fuse_norm_scales_for_ane() from gemma_to_ane.py.
    Returns scale factor S_g such that max(gate_fused(x) * S_g) < fp16 range.
    """
    import math
    # After fusing pre_ln2 into gate, the max gate weight magnitude increases.
    # We need: max|gelu(gate_fused(x))| * max|up_fused(x)| < 65504
    # Heuristic: compute max weight norms and set S_g = 1/sqrt(max_gate_norm)
    gate_up_keep = gate_up_all[keep_idx]  # (n_experts, 1408, 2816)
    gate_keep = gate_up_keep[:, :D_FFN, :]  # (n_experts, 704, 2816)
    ln2_s = pre_ln2_scale.astype(np.float32)  # γ directly (HF stores full scale)
    gate_fused = gate_keep * ln2_s[np.newaxis, np.newaxis, :]
    max_norm = float(np.abs(gate_fused).max())
    # Target: keep intermediate values well within fp16 range
    # GELU saturates around ±3 to identity, so max output ≈ max_norm * typical_input
    # Conservative: scale = sqrt(256/max_norm²) if max_norm > 16, else 1.0
    if max_norm > 16:
        return float(math.sqrt(256.0 / (max_norm * max_norm)))
    return 1.0


def process_layer(layer_idx: int, idx: dict, delete_pkg: bool = True,
                  n_experts: int = 128, quant_bits: int = 0):
    """Extract weights for one layer, build experts + 1 combine shard."""
    t0 = time.time()
    base = f"model.language_model.layers.{layer_idx}"
    
    def R(k):
        return _read_tensor(idx, f"{base}.{k}")

    use_reap = (n_experts < N_EXPERTS_ALL)

    if use_reap:
        # Load REAP mask
        mask = np.load(REAP_MASK)
        keep = mask["keep_idx"][layer_idx].astype(np.int64)
    else:
        keep = np.arange(N_EXPERTS_ALL, dtype=np.int64)

    # Load expert weights
    gate_up_all = R("experts.gate_up_proj")    # (128, 1408, 2816)
    down_all = R("experts.down_proj")          # (128, 2816, 704)
    
    gate_up_keep = gate_up_all[keep]           # (n_experts, 1408, 2816)
    down_keep = down_all[keep]                 # (n_experts, 2816, 704)

    # Load norms for combine shard
    pre_ffn_ln_w = R("pre_feedforward_layernorm.weight")
    post_ffn_ln_1_w = R("post_feedforward_layernorm_1.weight")
    post_ffn_ln_2_w = R("post_feedforward_layernorm_2.weight")
    post_ffn_ln_w = R("post_feedforward_layernorm.weight")
    pre_ffn_ln_2_w = R("pre_feedforward_layernorm_2.weight")
    layer_scalar = float(R("layer_scalar").item())

    # Dense MLP
    mlp_gate = R("mlp.gate_proj.weight")       # (2112, 2816)
    mlp_up = R("mlp.up_proj.weight")           # (2112, 2816)
    mlp_down = R("mlp.down_proj.weight")       # (2816, 2112)

    # Router weights (for Swift-side CPU routing)
    router_proj = R("router.proj.weight")       # (128, 2816)
    router_scale = R("router.scale")            # (2816,)
    router_per_exp = R("router.per_expert_scale")  # (128,)

    # REAP-slice router (or keep all)
    router_proj_k = router_proj[keep]           # (n_experts, 2816)
    router_per_exp_k = router_per_exp[keep]     # (n_experts,)

    # Fuse router scale into proj (same as fuse_norm_scales_for_ane)
    rms_eps = 1e-6
    scalar_root = D_MODEL ** -0.5
    r_scale = (router_scale * scalar_root).astype(np.float32)
    router_proj_fused = router_proj_k * r_scale[np.newaxis, :]  # (n_experts, 2816)

    # Compute gate_scale for fp16 safety
    gate_scale = compute_gate_scale(gate_up_all, pre_ffn_ln_2_w, keep)

    # Create output directory
    q_tag = "fp16" if quant_bits == 0 else f"q{quant_bits}"
    layer_dir = OUT_BASE / f"L{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Layer {layer_idx}: {n_experts} experts + 1 combine ({q_tag})")
    print(f"  gate_scale={gate_scale:.4f}  layer_scalar={layer_scalar:.4f}")
    print(f"{'='*60}")

    # Save router weights as fp16 binary for Swift
    router_dir = layer_dir / "router"
    router_dir.mkdir(exist_ok=True)
    router_proj_fused.astype(np.float16).tofile(router_dir / "proj_fp16.bin")
    router_per_exp_k.astype(np.float16).tofile(router_dir / "per_expert_scale_fp16.bin")
    print(f"  Router: proj ({router_proj_fused.shape}) + per_exp_scale ({router_per_exp_k.shape})")

    # Build individual experts
    expert_results = []
    for eid in range(n_experts):
        gate_e = gate_up_keep[eid, :D_FFN, :]    # (704, 2816)
        up_e = gate_up_keep[eid, D_FFN:, :]       # (704, 2816)
        down_e = down_keep[eid]                    # (2816, 704)

        name = f"expert_L{layer_idx}_e{eid:03d}_{q_tag}"
        pkg, modelc, sz = build_single_expert(
            gate_w=gate_e, up_w=up_e, down_w=down_e,
            pre_ln2_scale=pre_ffn_ln_2_w,
            gate_scale=gate_scale,
            shard_name=name, out_dir=layer_dir,
            quant_bits=quant_bits,
        )
        expert_results.append({"eid": eid, "modelc": modelc, "pkg_mb": sz})
        if delete_pkg:
            import shutil
            shutil.rmtree(pkg, ignore_errors=True)
        if (eid + 1) % 16 == 0:
            print(f"    experts {eid-15}–{eid}: {sz:.1f} MB each")

    # Build combine shard
    combine_name = f"combine_L{layer_idx}_{q_tag}"
    cpkg, cmodelc, csz = build_combine_shard(
        pre_ffn_ln_w=pre_ffn_ln_w,
        mlp_gate_w=mlp_gate, mlp_up_w=mlp_up, mlp_down_w=mlp_down,
        post_ffn_ln_1_w=post_ffn_ln_1_w,
        post_ffn_ln_2_w=post_ffn_ln_2_w,
        post_ffn_ln_w=post_ffn_ln_w,
        layer_scalar=layer_scalar,
        rms_eps=rms_eps,
        shard_name=combine_name, out_dir=layer_dir,
        quant_bits=quant_bits,
    )
    if delete_pkg:
        import shutil
        shutil.rmtree(cpkg, ignore_errors=True)
    print(f"  Combine: {csz:.1f} MB")

    elapsed = time.time() - t0
    print(f"  Layer {layer_idx} done in {elapsed:.1f}s")

    # Write layer manifest
    manifest = {
        "layer": layer_idx,
        "n_experts": n_experts,
        "top_k": TOP_K,
        "d_model": D_MODEL,
        "d_ffn": D_FFN,
        "gate_scale": gate_scale,
        "layer_scalar": layer_scalar,
        "rms_eps": rms_eps,
        "router": {
            "proj_bin": str(router_dir / "proj_fp16.bin"),
            "per_expert_scale_bin": str(router_dir / "per_expert_scale_fp16.bin"),
            "n_experts": n_experts,
            "d_model": D_MODEL,
        },
        "experts": [
            {"eid": r["eid"], "mlmodelc": r["modelc"]}
            for r in expert_results
        ],
        "combine": {"mlmodelc": cmodelc, "pkg_mb": csz},
    }
    manifest_path = layer_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None,
                    help="Build a single layer")
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=30)
    ap.add_argument("--n-experts", type=int, default=128,
                    help="Number of experts (128=all, 64=REAP-pruned)")
    ap.add_argument("--quant-bits", type=int, default=0,
                    help="Weight quantization bits (0=FP16, 8=INT8)")
    ap.add_argument("--keep-pkg", action="store_true",
                    help="Keep .mlpackage files (default: delete to save disk)")
    args = ap.parse_args()

    idx = _load_index()

    if args.layer is not None:
        process_layer(args.layer, idx, delete_pkg=not args.keep_pkg,
                      n_experts=args.n_experts, quant_bits=args.quant_bits)
    else:
        for L in range(args.layer_start, args.layer_end):
            process_layer(L, idx, delete_pkg=not args.keep_pkg,
                          n_experts=args.n_experts, quant_bits=args.quant_bits)


if __name__ == "__main__":
    main()
