"""Probe: did the model actually execute on the ANE?

Use Core ML compute plan (macOS 14.4+) to find out which device each op
actually ran on. The key API is MLComputePlan.estimatedCost(of:) and the
per-op compute device assignment.

Easier: ct.models.MLModel.compute_plan accessor may surface this.
Failing that, we read the compiled `.mlmodelc/model.mil`
and look for `cpuOnly`/`cpuAndGpu`/`cpuAndNeuralEngine` annotations, or
use `ct.models.compute_plan.MLComputePlan`.

Run with the same shapes as the expert probe.
"""
import shutil
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

OUT_DIR = Path(__file__).parent.parent / "tmp" / "device_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_expert(d_model, d_ffn, out_path):
    rng = np.random.default_rng(0)
    Wg = rng.standard_normal((d_ffn, d_model)).astype(np.float16)
    Wu = rng.standard_normal((d_ffn, d_model)).astype(np.float16)
    Wd = rng.standard_normal((d_model, d_ffn)).astype(np.float16)
    bg = np.zeros((d_ffn,), dtype=np.float16)
    bu = np.zeros((d_ffn,), dtype=np.float16)
    bd = np.zeros((d_model,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, d_model), dtype=ct.converters.mil.mil.types.fp16)])
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


def inspect_one(label, d_model, d_ffn):
    print(f"\n=== {label}: d_model={d_model} d_ffn={d_ffn} ===")
    path = OUT_DIR / f"{label}.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    build_expert(d_model, d_ffn, path)
    compiled = OUT_DIR / f"{label}.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)
    compiled = Path(ct.utils.compile_model(str(path), str(compiled)))

    # Use MLComputePlan if available
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(
            path=str(compiled),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        program = plan.model_structure.program
        if program is None:
            print("  (no program structure - not an mlprogram?)")
            return
        # iterate over functions
        for fn_name, fn in program.functions.items():
            print(f"  function {fn_name}:")
            for op in fn.block.operations:
                try:
                    dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if dev is None:
                        dev_name = "?"
                    else:
                        dev_name = dev.preferred_compute_device.__class__.__name__
                    print(f"    op '{op.operator_name}' -> {dev_name}")
                except Exception as e:
                    print(f"    op '{op.operator_name}' -> error: {e!r}")
    except Exception as e:
        print(f"  MLComputePlan failed: {e!r}")


def main():
    inspect_one("tiny",   1024, 4096)
    inspect_one("medium", 2048, 4096)
    inspect_one("gemma",  2304, 9216)


if __name__ == "__main__":
    main()
