"""_probe_sg2.py — isolate global-layer INT4 noise.

Builds a 2L mixed stack: [sliding(layer4), global(layer5)] and runs both
torch fp16 reference and CoreML INT4 candidate forward on identical inputs.
Reports cos after each layer (using k_new/v_new + a one-layer reference).

Step 1: torch reference using GemmaMixedStackWrap with layers 4,5.
Step 2: convert + INT4 quant, save as gemma4_sg2_real.mlpackage.
Step 3: also run a SLIDING-only torch + a GLOBAL-only torch with REAL layer-4
        output as global input, to attribute cos drop per layer.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct
import coremltools.optimize.coreml as cto

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaMixedStackWrap, _load_layer_weights,
    GemmaSlidingLayer, GemmaGlobalLayer, _state_shape,
    D_MODEL, SLD_D_HEAD, SLD_N_KV, GLB_ROT_DIM, GLB_N_KV, GLB_D_HEAD,
)

OUT_DIR = Path("python/moe/out")
PKG = OUT_DIR / "gemma4_sg2_real.mlpackage"
MLC = OUT_DIR / "gemma4_sg2_real.mlmodelc"
MAX_CTX = 1024
SEED = 0xA1E
LAYER_TYPES = ["sliding", "global"]
NPZS = [OUT_DIR / "gemma_layer4_packed.npz",
        OUT_DIR / "gemma_layer5_packed.npz"]


def _real_rope(theta: float, dh: int, pos: int):
    half = dh // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half) / half))
    freqs = pos * inv_freq
    full = np.concatenate([freqs, freqs])
    return np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)


def cos(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_torch_ref():
    ref = GemmaMixedStackWrap(MAX_CTX, LAYER_TYPES)
    ref.half().eval()
    for i, p in enumerate(NPZS):
        _load_layer_weights(ref.layers[i], p)
    return ref


def convert_and_quant(ref):
    print("  tracing...")
    dh_s = SLD_D_HEAD
    x_ex   = torch.randn(1, 1, D_MODEL,    dtype=torch.float16) * 0.1
    cos_s  = torch.randn(1, 1, dh_s,        dtype=torch.float16)
    sin_s  = torch.randn(1, 1, dh_s,        dtype=torch.float16)
    cos_g  = torch.randn(1, 1, GLB_ROT_DIM, dtype=torch.float16)
    sin_g  = torch.randn(1, 1, GLB_ROT_DIM, dtype=torch.float16)
    am_ex  = torch.full((1, 1, 1, MAX_CTX), -1e4, dtype=torch.float16); am_ex[..., 0] = 0.0
    wm_ex  = torch.zeros(1, 1, MAX_CTX, 1, dtype=torch.float16); wm_ex[0, 0, 0, 0] = 1.0
    traced = torch.jit.trace(ref, (x_ex, cos_s, sin_s, cos_g, sin_g, am_ex, wm_ex))
    with torch.no_grad():
        for i in range(2):
            getattr(ref, f"k_cache_{i}").zero_()
            getattr(ref, f"v_cache_{i}").zero_()

    print("  ct.convert...")
    ct_inputs = [
        ct.TensorType(name="x",             shape=(1, 1, D_MODEL),    dtype=np.float16),
        ct.TensorType(name="cos_s",         shape=(1, 1, dh_s),       dtype=np.float16),
        ct.TensorType(name="sin_s",         shape=(1, 1, dh_s),       dtype=np.float16),
        ct.TensorType(name="cos_g",         shape=(1, 1, GLB_ROT_DIM),dtype=np.float16),
        ct.TensorType(name="sin_g",         shape=(1, 1, GLB_ROT_DIM),dtype=np.float16),
        ct.TensorType(name="attn_mask",     shape=(1, 1, 1, MAX_CTX), dtype=np.float16),
        ct.TensorType(name="kv_write_mask", shape=(1, 1, MAX_CTX, 1), dtype=np.float16),
    ]
    ct_outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
        ct.TensorType(name="k_new",  dtype=np.float16),
        ct.TensorType(name="v_new",  dtype=np.float16),
    ]
    ct_states = []
    for i, t in enumerate(LAYER_TYPES):
        shp = _state_shape(t == "global", MAX_CTX)
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=shp, dtype=np.float16),
            name=f"k_cache_{i}"))
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=shp, dtype=np.float16),
            name=f"v_cache_{i}"))
    m = ct.convert(
        traced, inputs=ct_inputs, outputs=ct_outputs, states=ct_states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print("  INT4 per_block/16 quant...")
    m = cto.linear_quantize_weights(
        m, config=cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int4",
                granularity="per_block", block_size=16,
                weight_threshold=0)))
    if PKG.exists(): shutil.rmtree(PKG)
    m.save(str(PKG))
    if MLC.exists(): shutil.rmtree(MLC)
    Path(ct.utils.compile_model(str(PKG), str(MLC)))
    print(f"  saved {PKG.name} + {MLC.name}")


def main():
    rng = np.random.default_rng(SEED)
    x_np = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)
    cos_s_f, sin_s_f = _real_rope(10000.0,    SLD_D_HEAD, 0)
    cos_g_f, sin_g_f = _real_rope(1_000_000., GLB_ROT_DIM, 0)
    cos_s = cos_s_f.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    sin_s = sin_s_f.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    cos_g = cos_g_f.astype(np.float16).reshape(1, 1, GLB_ROT_DIM)
    sin_g = sin_g_f.astype(np.float16).reshape(1, 1, GLB_ROT_DIM)
    am = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16); am[..., 0] = 0.0
    wm = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16); wm[0, 0, 0, 0] = 1.0

    print("=== _probe_sg2: layers [sliding=4, global=5] ===")
    ref = build_torch_ref()

    if not MLC.exists():
        convert_and_quant(ref)
    else:
        print(f"  reusing existing {MLC.name}")

    # Reset KV
    with torch.no_grad():
        for i in range(2):
            getattr(ref, f"k_cache_{i}").zero_()
            getattr(ref, f"v_cache_{i}").zero_()

    # 1) Torch full ref through both layers
    with torch.no_grad():
        full_hid, full_k, full_v = ref(
            torch.from_numpy(x_np),
            torch.from_numpy(cos_s), torch.from_numpy(sin_s),
            torch.from_numpy(cos_g), torch.from_numpy(sin_g),
            torch.from_numpy(am), torch.from_numpy(wm),
        )
    full_hid_np = full_hid.float().numpy().reshape(D_MODEL)

    # Read intermediate after layer 0 (sliding) from KV cache + recompute
    # We'll re-run just layer 0 alone.
    sld_solo = GemmaSlidingLayer(MAX_CTX).half().eval()
    _load_layer_weights(sld_solo, NPZS[0])
    kc0 = torch.zeros(1, SLD_N_KV, MAX_CTX, SLD_D_HEAD, dtype=torch.float16)
    vc0 = torch.zeros(1, SLD_N_KV, MAX_CTX, SLD_D_HEAD, dtype=torch.float16)
    with torch.no_grad():
        ref_l0_hid, _, _ = sld_solo(
            torch.from_numpy(x_np),
            torch.from_numpy(cos_s), torch.from_numpy(sin_s),
            kc0, vc0, torch.from_numpy(am), torch.from_numpy(wm))
    ref_l0_hid_np = ref_l0_hid.float().numpy().reshape(D_MODEL)

    # 2) CoreML run
    print("  loading mlmodelc...")
    m = ct.models.CompiledMLModel(str(MLC), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = m.make_state()
    inputs = {
        "x": x_np, "cos_s": cos_s, "sin_s": sin_s,
        "cos_g": cos_g, "sin_g": sin_g,
        "attn_mask": am, "kv_write_mask": wm,
    }
    _ = m.predict(inputs, state=state)
    state = m.make_state()
    out = m.predict(inputs, state=state)
    cml_hid = np.asarray(out["hidden"]).astype(np.float32).reshape(D_MODEL)
    cml_k   = np.asarray(out["k_new"]).astype(np.float32)
    cml_v   = np.asarray(out["v_new"]).astype(np.float32)

    print()
    print(f"  cos(layer1 hidden, full ref vs INT4 full)  = {cos(full_hid_np, cml_hid):.6f}")
    print(f"  cos(k_global last, ref vs INT4)             = {cos(full_k.float().numpy(), cml_k):.6f}")
    print(f"  cos(v_global last, ref vs INT4)             = {cos(full_v.float().numpy(), cml_v):.6f}")
    print(f"  ‖full_ref‖={np.linalg.norm(full_hid_np):.3f}  ‖cml‖={np.linalg.norm(cml_hid):.3f}")
    print(f"  ‖layer0 ref hidden‖={np.linalg.norm(ref_l0_hid_np):.3f}")

    # 3) Solo INT4 layer-0 to get sliding-only floor: just measure how much
    # cos(layer0_ref, layer1_ref) drops to bound the global layer's contribution.
    print()
    print("  --- attribution (random N(0,0.5^2) input) ---")
    print(f"  cos(input          , layer0_ref) = {cos(x_np.astype(np.float32).reshape(D_MODEL), ref_l0_hid_np):.6f}")
    print(f"  cos(layer0_ref     , layer1_ref) = {cos(ref_l0_hid_np, full_hid_np):.6f}")
    print("  (sliding-only INT4 cos floor from T4.1.2 = 0.992)")
    print("  expected mixed cos = 0.992 * (global INT4 floor)")


if __name__ == "__main__":
    main()
