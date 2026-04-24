"""Probe A' (Gemma-real): format sweep with TRUE Gemma-4-26B-A4B expert dims.

Real architecture (from config.json):
  d_model = 2816
  moe_intermediate_size = 704     <-- 13x smaller than our earlier guess
  hidden_activation = "gelu_pytorch_tanh"
  3 projections per expert: gate, up, down (GeGLU)

Expected expert size at fp16: 3 * 2816 * 704 * 2 = ~11.9 MB
At INT4: ~3 MB. We expect this to be COMPLETELY ANE-cache-resident, so
latency should be far below the 0.56 ms we measured on the larger 127 MB
expert.
"""
import shutil
import time
from pathlib import Path

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb

OUT_DIR = Path(__file__).parent.parent / "tmp" / "gemma_format_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL = 2816
D_FFN   = 704   # moe_intermediate_size


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
        # Gemma uses gelu_pytorch_tanh (approximate gelu). MIL `gelu` with
        # mode="TANH_APPROXIMATION" matches the PyTorch tanh-approx formula.
        gelu_g = mb.gelu(x=g, mode="TANH_APPROXIMATION", name="gelu")
        h = mb.mul(x=gelu_g, y=u, name="mul")
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


def time_predict(m, x, in_name, n_iter=200, warmup=30):
    for _ in range(warmup):
        m.predict({in_name: x})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        m.predict({in_name: x})
    return (time.perf_counter() - t0) / n_iter


def device_summary(pkg_path: Path) -> str:
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
        lat = time_predict(compressed, x, in_name, n_iter=200, warmup=30)
    except Exception as e:
        print(f"  predict failed: {e!r}")
        return None
    devs = device_summary(out_path)
    bw = (size_mb * 1e6) / 1e9 / lat if lat > 0 else 0
    print(f"  size={size_mb:6.2f} MB   lat={lat*1e3:6.3f} ms   {bw:5.1f} GB/s   linears: {devs}")
    return (label, size_mb, lat * 1e3, bw, devs)


def main():
    rows = []
    rows.append(measure("baseline_fp16", None))

    rows.append(measure("linear_w_int8",
                         lambda m: cto.linear_quantize_weights(
                             m, config=cto.OptimizationConfig(
                                 global_config=cto.OpLinearQuantizerConfig(
                                     mode="linear_symmetric",
                                     weight_threshold=0, dtype="int8")))))

    rows.append(measure("linear_w_int4",
                         lambda m: cto.linear_quantize_weights(
                             m, config=cto.OptimizationConfig(
                                 global_config=cto.OpLinearQuantizerConfig(
                                     mode="linear_symmetric",
                                     weight_threshold=0, dtype="int4")))))

    rows.append(measure("palette_4bit_per_channel",
                         lambda m: cto.palettize_weights(
                             m, config=cto.OptimizationConfig(
                                 global_config=cto.OpPalettizerConfig(
                                     nbits=4, mode="kmeans",
                                     granularity="per_grouped_channel",
                                     group_size=16,
                                     weight_threshold=0)))))

    print("\n=== SUMMARY (Gemma-4 expert: d=2816, ffn=704, GeGLU) ===")
    print(f"{'format':28s}  {'size_MB':>8s}  {'lat_ms':>7s}  {'GB/s':>6s}  devices")
    for r in rows:
        if r is None:
            continue
        label, size_mb, lat_ms, bw, devs = r
        print(f"{label:28s}  {size_mb:8.2f}  {lat_ms:7.3f}  {bw:6.1f}  {devs}")

    # Per-token MLP projection: 8 experts * 30 layers
    print("\n=== Per-token MLP cost projection (8 experts * 30 layers) ===")
    for r in rows:
        if r is None:
            continue
        label, _, lat_ms, _, _ = r
        print(f"  {label:28s}  {8*30*lat_ms:6.1f} ms/token")


if __name__ == "__main__":
    main()
