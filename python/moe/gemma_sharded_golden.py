"""gemma_sharded_golden.py — structural golden gate for the SHARDED Gemma-4
mixed30 stack.

Chains N shards (each is a compiled mlmodelc) by feeding the hidden output of
shard k as the input x of shard k+1. Compares end-to-end CoreML output vs a
single full PyTorch fp16 reference over all 30 layers.

C3 hard abort on NaN/Inf. Sentinel:
  python/moe/out/.gemma_sharded_{TOTAL}_golden_PASS

Usage (Xcode python only — coremltools 9):
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_sharded_golden.py \
        --total 30 \
        --shard 0:15 --shard 15:22 --shard 22:30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))
from gemma_to_ane import (  # noqa: E402
    GemmaMixedStackWrap, _load_layer_weights,
    _layer_types_from_config,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM,
)
from gemma_mixedN_golden import (  # noqa: E402
    _cos_floor_h, _cos_floor_k, _topk_floor, _real_rope,
)

OUT_DIR = Path("python/moe/out")
MAX_CTX = 1024
SEED = 0xA1E


def _parse_shard(s: str) -> tuple[int, int]:
    a, b = s.split(":")
    a, b = int(a), int(b)
    assert b > a, f"bad shard range {s}"
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, required=True,
                    help="total number of layers across all shards (e.g. 30)")
    ap.add_argument("--shard", action="append", required=True,
                    help="shard range start:end (exclusive); repeat for each shard")
    ap.add_argument("--topk-floor", type=float, default=None,
                    help="override depth-aware top-K floor")
    args = ap.parse_args()

    shards = [_parse_shard(s) for s in args.shard]
    # Validate full coverage and contiguity.
    assert shards[0][0] == 0, f"first shard must start at 0, got {shards[0]}"
    assert shards[-1][1] == args.total, \
        f"last shard must end at {args.total}, got {shards[-1]}"
    for (a1, b1), (a2, b2) in zip(shards, shards[1:]):
        assert b1 == a2, f"shards not contiguous: {(a1,b1)} -> {(a2,b2)}"

    N = args.total
    layer_types = _layer_types_from_config(N)
    cos_floor = _cos_floor_h(N)
    k_floor = _cos_floor_k(N)
    topk_floor = args.topk_floor if args.topk_floor is not None else _topk_floor(N)

    print(f"=== sharded structural golden  total={N}  shards={shards} ===")
    print(f"  layer_types ({N}): {layer_types}")
    print(f"  cos_floor_h={cos_floor:.3f}  cos_floor_k={k_floor:.3f}  topk_floor={topk_floor:.2f}")

    # Locate compiled shard mlmodelcs.
    shard_paths = []
    for a, b in shards:
        p = OUT_DIR / f"gemma4_shard{a}_{b}_real.mlmodelc"
        if not p.exists():
            print(f"FATAL missing shard mlmodelc: {p}", file=sys.stderr)
            sys.exit(2)
        shard_paths.append(p)

    # Layer pack files for the PyTorch reference.
    npz_paths = [OUT_DIR / f"gemma_layer{i}_packed.npz" for i in range(N)]
    for p in npz_paths:
        if not p.exists():
            print(f"FATAL missing pack: {p}", file=sys.stderr)
            sys.exit(2)

    # ---- Inputs ----
    rng = np.random.default_rng(SEED)
    x_np = (rng.standard_normal((1, 1, D_MODEL)) * 0.5).astype(np.float16)
    cos_s_f, sin_s_f = _real_rope(theta=10000.0, dh=SLD_D_HEAD, pos=0)
    cos_s = cos_s_f.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    sin_s = sin_s_f.astype(np.float16).reshape(1, 1, SLD_D_HEAD)
    cos_g_f, sin_g_f = _real_rope(theta=1_000_000.0, dh=GLB_ROT_DIM, pos=0)
    cos_g = cos_g_f.astype(np.float16).reshape(1, 1, GLB_ROT_DIM)
    sin_g = sin_g_f.astype(np.float16).reshape(1, 1, GLB_ROT_DIM)
    attn_mask_np = np.full((1, 1, 1, MAX_CTX), -1e4, dtype=np.float16)
    attn_mask_np[..., 0] = 0.0
    wmask_np = np.zeros((1, 1, MAX_CTX, 1), dtype=np.float16)
    wmask_np[0, 0, 0, 0] = 1.0

    # ---- Full PyTorch reference (all N layers) ----
    print(f"  building PyTorch reference ({N} layers)...")
    ref = GemmaMixedStackWrap(MAX_CTX, layer_types)
    ref.half().eval()
    for i, p in enumerate(npz_paths):
        _load_layer_weights(ref.layers[i], p)
    with torch.no_grad():
        for i in range(N):
            getattr(ref, f"k_cache_{i}").zero_()
            getattr(ref, f"v_cache_{i}").zero_()
        t0 = time.perf_counter()
        ref_hidden, ref_k, ref_v = ref(
            torch.from_numpy(x_np),
            torch.from_numpy(cos_s), torch.from_numpy(sin_s),
            torch.from_numpy(cos_g), torch.from_numpy(sin_g),
            torch.from_numpy(attn_mask_np),
            torch.from_numpy(wmask_np),
        )
        ref_t = time.perf_counter() - t0
    ref_hidden = ref_hidden.float().numpy().reshape(D_MODEL)
    ref_k_np = ref_k.float().numpy()
    ref_v_np = ref_v.float().numpy()
    print(f"  pytorch fp16 fwd: {ref_t*1e3:.1f} ms, ‖hidden‖={np.linalg.norm(ref_hidden):.3f}")
    # Free reference model RAM before loading CoreML shards.
    del ref

    # ---- Chain CoreML shards ----
    cur_x = x_np
    last_k = last_v = None
    cml_t = 0.0
    for (a, b), pth in zip(shards, shard_paths):
        print(f"  loading shard [{a},{b})  {pth.name} ...")
        m = ct.models.CompiledMLModel(str(pth), compute_units=ct.ComputeUnit.CPU_AND_NE)
        # Warm + clear state.
        warm_state = m.make_state()
        inputs = {
            "x": cur_x,
            "cos_s": cos_s, "sin_s": sin_s,
            "cos_g": cos_g, "sin_g": sin_g,
            "attn_mask": attn_mask_np, "kv_write_mask": wmask_np,
        }
        _ = m.predict(inputs, state=warm_state)
        state = m.make_state()
        t0 = time.perf_counter()
        out = m.predict(inputs, state=state)
        dt = time.perf_counter() - t0
        cml_t += dt
        next_x = np.asarray(out["hidden"]).astype(np.float16).reshape(1, 1, D_MODEL)
        # k_new/v_new no longer emitted as outputs (state buffers are the cache);
        # validate finiteness on hidden only.
        if not np.all(np.isfinite(next_x)):
            n_nan = int(np.isnan(next_x).sum()); n_inf = int(np.isinf(next_x).sum())
            print(f"FATAL non-finite in shard[{a},{b}].hidden: "
                  f"nan={n_nan} inf={n_inf}", file=sys.stderr)
            sys.exit(2)
        print(f"    shard[{a},{b}]  {dt*1e3:.1f} ms  "
              f"‖hidden‖={float(np.linalg.norm(next_x)):.3f}")
        cur_x = next_x
        # Drop model + state to release memory before loading next shard.
        del m, state, warm_state, out

    cml_hidden = cur_x.astype(np.float32).reshape(D_MODEL)
    print(f"  coreml int4 chain: {cml_t*1e3:.1f} ms total, "
          f"‖hidden‖={np.linalg.norm(cml_hidden):.3f}")

    def cos(a, b):
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cos_h = cos(ref_hidden, cml_hidden)
    # cos_k/cos_v removed: shards no longer emit k_new/v_new (state-only cache).
    K = 32
    ref_top = set(np.argsort(-np.abs(ref_hidden))[:K].tolist())
    cml_top = set(np.argsort(-np.abs(cml_hidden))[:K].tolist())
    overlap = len(ref_top & cml_top) / K

    print()
    print(f"  cos(hidden) = {cos_h:.6f}  (floor {cos_floor:.3f})")
    print(f"  top-{K} overlap = {overlap:.3f}  (floor {topk_floor:.2f})")

    ok = (cos_h >= cos_floor) and (overlap >= topk_floor)
    if ok:
        print(f"\n# sharded{N} STRUCTURAL GOLDEN: PASS")
        sentinel = OUT_DIR / f".gemma_sharded_{N}_golden_PASS"
        sentinel.write_text(
            f"cos_h={cos_h:.6f} overlap={overlap:.3f} "
            f"shards={shards}\n")
        print(f"  sentinel: {sentinel}")
        sys.exit(0)
    else:
        print(f"\n# sharded{N} STRUCTURAL GOLDEN: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
