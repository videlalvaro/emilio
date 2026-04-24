"""Probe A': what low-precision storage format does the ANE actually stream?

We rebuild the same Gemma-shape SwiGLU expert (d_model=2304, d_ffn=9216, ~127 MB
fp16) and apply each of coremltools' weight-compression transforms. We then
measure latency on CPU_AND_NE and verify (via MLComputePlan) which device each
op landed on.

Hypothesis: palettization (4-bit LUT) is the format Apple's compiler maps
natively to ANE. linear quantize int4/int8 may silently rehydrate to fp16.

Output: table of {format, mlpackage_size_MB, latency_ms, on_ane?}.
"""
import shutil
import time
from pathlib import Path

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb

OUT_DIR = Path(__file__).parent.parent / "tmp" / "format_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL = 2304
D_FFN = 9216


def build_baseline(out_path: Path) -> ct.models.MLModel:
    rng = np.random.default_rng(0)
    Wg = rng.standard_normal((D_FFN, D_MODEL)).astype(np.float16)
    Wu = rng.standard_normal((D_FFN, D_MODEL)).astype(np.float16)
    Wd = rng.standard_normal((D_MODEL, D_FFN)).astype(np.float16)
    bg = np.zeros((D_FFN,), dtype=np.float16)
    bu = np.zeros((D_FFN,), dtype=np.float16)
    bd = np.zeros((D_MODEL,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, D_MODEL),
                                            dtype=ct.converters.mil.mil.types.fp16)])
    def prog(x):
        g = mb.linear(x=x, weight=Wg, bias=bg, name="gate")
        u = mb.linear(x=x, weight=Wu, bias=bu, name="up")
        sg = mb.silu(x=g, name="silu")
        h = mb.mul(x=sg, y=u, name="mul")
        return mb.linear(x=h, weight=Wd, bias=bd, name="down")

    m = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    m.save(str(out_path))
    return m


def package_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def time_predict(m, x, in_name, n_iter=100, warmup=20):
    for _ in range(warmup):
        m.predict({in_name: x})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        m.predict({in_name: x})
    return (time.perf_counter() - t0) / n_iter


def device_summary(pkg_path: Path) -> str:
    """Compile and inspect MLComputePlan; return short summary like 'ANE/ANE/CPU'."""
    try:
        from coremltools.models.compute_plan import MLComputePlan
        compiled = pkg_path.with_suffix(".mlmodelc")
        if compiled.exists():
            shutil.rmtree(compiled)
        compiled = Path(ct.utils.compile_model(str(pkg_path), str(compiled)))
        plan = MLComputePlan.load_from_path(
            path=str(compiled),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        program = plan.model_structure.program
        if program is None:
            return "?"
        devs = []
        for fn_name, fn in program.functions.items():
            for op in fn.block.operations:
                if op.operator_name in ("ios18.linear", "linear"):
                    try:
                        d = plan.get_compute_device_usage_for_mlprogram_operation(op)
                        if d is None:
                            devs.append("?")
                        else:
                            n = d.preferred_compute_device.__class__.__name__
                            devs.append("ANE" if "Neural" in n else ("GPU" if "GPU" in n else "CPU"))
                    except Exception:
                        devs.append("?")
        return "/".join(devs) if devs else "?"
    except Exception as e:
        return f"err:{type(e).__name__}"


def measure(label: str, transform):
    print(f"\n=== {label} ===")
    base_path = OUT_DIR / "baseline.mlpackage"
    if not base_path.exists():
        build_baseline(base_path)

    out_path = OUT_DIR / f"{label}.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)

    base = ct.models.MLModel(str(base_path),
                             compute_units=ct.ComputeUnit.CPU_AND_NE)
    if transform is None:
        # baseline: just copy by re-saving
        base.save(str(out_path))
        compressed = ct.models.MLModel(str(out_path),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    else:
        try:
            compressed = transform(base)
            compressed.save(str(out_path))
        except Exception as e:
            print(f"  transform failed: {e!r}")
            return None

    size_mb = package_size_mb(out_path)
    in_name = list(compressed.input_description)[0]
    x = np.zeros((1, D_MODEL), dtype=np.float32)
    try:
        lat = time_predict(compressed, x, in_name, n_iter=80, warmup=15)
    except Exception as e:
        print(f"  predict failed: {e!r}")
        return None
    devs = device_summary(out_path)
    bw = (size_mb * 1e6) / 1e9 / lat
    print(f"  size={size_mb:6.1f} MB   lat={lat*1e3:6.2f} ms   {bw:5.1f} GB/s   linears: {devs}")
    return (label, size_mb, lat * 1e3, bw, devs)


def main():
    rows = []
    rows.append(measure("baseline_fp16", None))

    rows.append(measure("linear_w_int8",
                         lambda m: cto.linear_quantize_weights(
                             m, config=cto.OptimizationConfig(
                                 global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric",
                                                                            weight_threshold=0,
                                                                            dtype="int8")))))

    rows.append(measure("linear_w_int4",
                         lambda m: cto.linear_quantize_weights(
                             m, config=cto.OptimizationConfig(
                                 global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric",
                                                                            weight_threshold=0,
                                                                            dtype="int4")))))

    rows.append(measure("palette_4bit_per_tensor",
                         lambda m: cto.palettize_weights(
                             m, config=cto.OptimizationConfig(
                                 global_config=cto.OpPalettizerConfig(nbits=4, mode="kmeans",
                                                                       weight_threshold=0)))))

    rows.append(measure("palette_4bit_per_channel",
                         lambda m: cto.palettize_weights(
                             m, config=cto.OptimizationConfig(
                                 global_config=cto.OpPalettizerConfig(
                                     nbits=4, mode="kmeans",
                                     granularity="per_grouped_channel",
                                     group_size=16,
                                     weight_threshold=0)))))

    rows.append(measure("palette_8bit_per_tensor",
                         lambda m: cto.palettize_weights(
                             m, config=cto.OptimizationConfig(
                                 global_config=cto.OpPalettizerConfig(nbits=8, mode="kmeans",
                                                                       weight_threshold=0)))))

    print("\n=== SUMMARY ===")
    print(f"{'format':28s}  {'size_MB':>8s}  {'lat_ms':>7s}  {'GB/s':>6s}  {'devices':s}")
    for r in rows:
        if r is None:
            continue
        label, size_mb, lat_ms, bw, devs = r
        print(f"{label:28s}  {size_mb:8.1f}  {lat_ms:7.2f}  {bw:6.1f}  {devs}")


if __name__ == "__main__":
    main()
