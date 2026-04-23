"""T3.5a — INT4 weight quantization of per-expert cached artifacts.

Workaround for coremltools 9 limitation: `linear_quantize_weights` rejects
multifunction models. We bypass by quantizing each cached single-function
artifact in `_pf_expert_cache/` BEFORE assembly. Output goes to
`_pf_expert_cache_int4/`.

Phases:
  1) smoke: quantize ONLY expert_000 → cosine vs FP16 reference
  2) full:  quantize all 128 + reassemble into PF_experts128_layer0_int4.mlpackage

Flags:
  --smoke-only    only do expert_000
  --skip-existing don't re-quantize artifacts that already exist
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

import coremltools as ct
import coremltools.optimize.coreml as cto

ROOT = Path(__file__).resolve().parents[2]
SRC_CACHE = ROOT / "emilio" / "conv-ane" / "_pf_expert_cache"
DST_CACHE = ROOT / "emilio" / "conv-ane" / "_pf_expert_cache_int4"
SMOKE_PACK = ROOT / "emilio" / "conv-ane" / "PF_expert000_int4_smoke.mlpackage"
FULL_PACK = ROOT / "emilio" / "conv-ane" / "PF_experts128_layer0_int4.mlpackage"
GOLDEN_NPZ = ROOT / "python" / "privacy" / "pf_layer0_moe.npz"

D_MODEL = 640
N_EXPERTS = 128


def pkg_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


VARIANTS = {
    "linear_int4_per_tensor": lambda m: cto.linear_quantize_weights(
        m, config=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", weight_threshold=0, dtype="int4"))),
    "linear_int4_per_channel": lambda m: cto.linear_quantize_weights(
        m, config=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", weight_threshold=0, dtype="int4",
            granularity="per_channel"))),
    "linear_int4_grouped_g32": lambda m: cto.linear_quantize_weights(
        m, config=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", weight_threshold=0, dtype="int4",
            granularity="per_block", block_size=32))),
    "palette_4bit_grouped_g16": lambda m: cto.palettize_weights(
        m, config=cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
            nbits=4, mode="kmeans", granularity="per_grouped_channel",
            group_size=16, weight_threshold=0))),
    "linear_int8_per_channel": lambda m: cto.linear_quantize_weights(
        m, config=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", weight_threshold=0, dtype="int8",
            granularity="per_channel"))),
}

# Default chosen by smoke sweep; updated by smoke() if needed.
ACTIVE_VARIANT = "linear_int4_per_channel"


def quantize_one(src: Path, dst: Path, variant: str = None) -> float:
    base = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_AND_NE)
    fn = VARIANTS[variant or ACTIVE_VARIANT]
    quant = fn(base)
    if dst.exists():
        shutil.rmtree(dst)
    quant.save(str(dst))
    return pkg_size_mb(dst)


def make_feed():
    g = np.load(GOLDEN_NPZ)
    flat_norm = g["mlp_norm_out"].reshape(-1, D_MODEL)
    pad = np.zeros((64, D_MODEL, 1, 1), dtype=np.float16)
    pad[0] = flat_norm[0].astype(np.float16).reshape(D_MODEL, 1, 1)
    return {"x_in": pad}


def smoke() -> None:
    """Sweep quant variants on expert_000, pick the smallest that hits cosine ≥ 0.99."""
    src = SRC_CACHE / "expert_000.mlpackage"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    fp16_size = pkg_size_mb(src)
    print(f"[smoke] src: {src.name} ({fp16_size:.2f} MB FP16)")

    feed = make_feed()
    m_fp = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_AND_NE)
    out_fp = m_fp.predict(feed)["x_out"][0].astype(np.float32)

    results = []  # (variant, size_mb, cosine)
    for variant in VARIANTS:
        tmp = SMOKE_PACK.parent / f"_smoke_{variant}.mlpackage"
        try:
            t0 = time.perf_counter()
            sz = quantize_one(src, tmp, variant=variant)
            wall = time.perf_counter() - t0
            m_q = ct.models.MLModel(str(tmp), compute_units=ct.ComputeUnit.CPU_AND_NE)
            out_q = m_q.predict(feed)["x_out"][0].astype(np.float32)
            c = cosine(out_fp, out_q)
            print(f"[smoke] {variant:30s}  {sz:5.2f} MB  cos={c:.6f}  ({wall:.1f}s)")
            results.append((variant, sz, c, tmp))
        except Exception as e:
            print(f"[smoke] {variant:30s}  ERROR: {e!r}")

    # Pick smallest variant with cos >= 0.99
    passing = [(v, s, c, p) for (v, s, c, p) in results if c >= 0.99]
    if not passing:
        # Fall back to highest cosine
        best = max(results, key=lambda r: r[2])
        print(f"\n[smoke] FAIL: no variant >= 0.99. Best was {best[0]} cos={best[2]:.6f}")
        for _, _, _, p in results:
            if p.exists():
                shutil.rmtree(p)
        raise SystemExit(2)

    chosen = min(passing, key=lambda r: r[1])
    print(f"\n[smoke] PASS: chosen variant = {chosen[0]}  size={chosen[1]:.2f} MB  cos={chosen[2]:.6f}")
    print(f"[smoke] compression: {fp16_size/chosen[1]:.2f}×")

    # Save the chosen smoke artifact, clean up the others
    if SMOKE_PACK.exists():
        shutil.rmtree(SMOKE_PACK)
    chosen[3].rename(SMOKE_PACK)
    for _, _, _, p in results:
        if p.exists() and p != chosen[3]:
            shutil.rmtree(p)
    print(f"[smoke] artifact: {SMOKE_PACK.name}")

    global ACTIVE_VARIANT
    ACTIVE_VARIANT = chosen[0]


def quantize_all(skip_existing: bool) -> None:
    DST_CACHE.mkdir(parents=True, exist_ok=True)
    feed = make_feed() if GOLDEN_NPZ.exists() else None
    sizes, cosines = [], []
    t_start = time.perf_counter()
    for eid in range(N_EXPERTS):
        src = SRC_CACHE / f"expert_{eid:03d}.mlpackage"
        dst = DST_CACHE / f"expert_{eid:03d}.mlpackage"
        if skip_existing and dst.exists():
            sizes.append(pkg_size_mb(dst))
            continue
        t0 = time.perf_counter()
        sz = quantize_one(src, dst)
        sizes.append(sz)
        if feed is not None and eid % 16 == 0:
            m_fp = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_AND_NE)
            m_q = ct.models.MLModel(str(dst), compute_units=ct.ComputeUnit.CPU_AND_NE)
            of = m_fp.predict(feed)["x_out"][0].astype(np.float32)
            oq = m_q.predict(feed)["x_out"][0].astype(np.float32)
            c = cosine(of, oq)
            cosines.append((eid, c))
            print(f"  [{eid+1:3d}/{N_EXPERTS}] {sz:.2f} MB cos={c:.6f} ({time.perf_counter()-t0:.1f}s)")
        else:
            print(f"  [{eid+1:3d}/{N_EXPERTS}] {sz:.2f} MB ({time.perf_counter()-t0:.1f}s)")
    print(f"\n[full] quantize: {time.perf_counter()-t_start:.1f}s")
    print(f"[full] mean expert size: {np.mean(sizes):.2f} MB")
    if cosines:
        cs = np.array([c for _, c in cosines])
        print(f"[full] cosine probes: worst={cs.min():.6f} mean={cs.mean():.6f}")

    print(f"\n[full] reassembling 128-fn MFD...")
    t0 = time.perf_counter()
    desc = ct.utils.MultiFunctionDescriptor()
    for eid in range(N_EXPERTS):
        src = DST_CACHE / f"expert_{eid:03d}.mlpackage"
        desc.add_function(str(src), src_function_name="main",
                          target_function_name=f"expert_{eid:03d}")
    desc.default_function_name = "expert_000"
    if FULL_PACK.exists():
        shutil.rmtree(FULL_PACK)
    ct.utils.save_multifunction(desc, str(FULL_PACK))
    print(f"[full] assemble: {time.perf_counter()-t0:.1f}s")
    print(f"[full] artifact: {FULL_PACK.name} ({pkg_size_mb(FULL_PACK):.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    smoke()
    if args.smoke_only:
        print("\n[t3.5a] --smoke-only set, stopping")
        return
    quantize_all(skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
