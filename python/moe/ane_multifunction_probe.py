"""Build a tiny multi-function .mlpackage to verify the path works end-to-end.

Path under test:
  coremltools.utils.MultiFunctionDescriptor + save_multifunction
    -> .mlpackage with N named functions (constants deduped across functions)
    -> compile to .mlmodelc
    -> load each via MLModelConfiguration.functionName

This is the "MoE expert" authoring contract: each expert is a named function,
they share weights for the parts that are identical, and dispatch picks one
function per call. The private `_ANEModelInstanceParameters._procedureArray`
is what this becomes inside the daemon.
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

OUT_DIR = Path(__file__).parent.parent / "tmp" / "mfn_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_one_expert(expert_id: int, out_path: Path):
    """One tiny dense MLP: y = relu(x @ W) where W is constant.
    Different expert_id -> different W so we can tell them apart at runtime."""
    rng = np.random.default_rng(seed=expert_id)
    W = rng.standard_normal((8, 8)).astype(np.float16)
    b = np.zeros((8,), dtype=np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 8), dtype=ct.converters.mil.mil.types.fp16)])
    def prog(x):
        y = mb.linear(x=x, weight=W, bias=b, name="lin")
        z = mb.relu(x=y, name="out")
        return z

    mlmodel = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.save(str(out_path))


def main():
    n_experts = 4
    parts = []
    for i in range(n_experts):
        p = OUT_DIR / f"expert_{i}.mlpackage"
        if p.exists():
            shutil.rmtree(p)
        print(f"[+] building {p.name}")
        build_one_expert(i, p)
        parts.append(p)

    multi_path = OUT_DIR / "experts_multi.mlpackage"
    if multi_path.exists():
        shutil.rmtree(multi_path)

    desc = MultiFunctionDescriptor(str(parts[0]))
    desc.default_function_name = "expert_0"
    # Rename the default 'main' function inside each part to expert_<i>
    # First part's default already there as 'main' -> we will rename via add_function:
    # add_function(source_pkg, source_function_name, target_function_name)
    # Re-add expert_0 explicitly so it's named expert_0 (not 'main')
    desc = MultiFunctionDescriptor()
    for i, p in enumerate(parts):
        desc.add_function(str(p), "main", f"expert_{i}")
    desc.default_function_name = "expert_0"

    print(f"[+] saving multifunction package to {multi_path.name}")
    save_multifunction(desc, str(multi_path))

    # Load each named function directly from the .mlpackage; CoreML compiles internally.
    x = np.arange(8, dtype=np.float16).reshape(1, 8).astype(np.float32)
    print(f"[+] loading & running each function from {multi_path.name}")
    for i in range(n_experts):
        m = ct.models.MLModel(
            str(multi_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            function_name=f"expert_{i}",
        )
        # In multi-function models the spec.description.input is empty;
        # use input_description (per-function) instead.
        in_name = list(m.input_description)[0] if list(m.input_description) else "x"
        out = m.predict({in_name: x})
        out_name = next(iter(out))
        out_arr = np.array(out[out_name]).reshape(-1)
        print(f"    expert_{i}: out[:4] = {out_arr[:4].tolist()}")

    print("[+] done. multi-function .mlpackage works end-to-end.")


if __name__ == "__main__":
    main()
