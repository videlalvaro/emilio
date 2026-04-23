"""ANE gather residency probe.

Question: does mb.gather on a constant (weight) tensor stay on ANE?
If yes -> dynamic per-row expert selection is feasible (49× ceiling).
If no  -> shard-dispatch (4× ceiling) is the only path.

Builds 3 toy models (~tiny) and reports per-op compute device:
  M1: gather K rows from a [N, F] weight by [B, K] int indices, then conv
  M2: gather + reduce-sum (mimics expert combine pattern)
  M3: dense conv baseline (sanity check that the framework lands on ANE)
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path
from collections import Counter
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TMP = REPO_ROOT / "emilio" / "conv-ane" / "_gather_probe"

# Realistic-ish dimensions (smaller than full MoE so build is fast).
N_EXPERTS = 128
D_FF = 640
D_MODEL = 640
B = 64
K = 4   # top-k


def _check_interp() -> None:
    import coremltools as ct  # noqa: F401
    assert "Xcode.app" in sys.executable, sys.executable


def report_residency(label: str, mlpackage: Path) -> dict:
    import coremltools as ct
    compiled = mlpackage.with_suffix(".mlmodelc")
    if compiled.exists():
        shutil.rmtree(compiled)
    compiled = Path(ct.utils.compile_model(str(mlpackage), str(compiled)))
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts: Counter = Counter()
    op_dev: list[tuple[str, str]] = []
    for op in plan.model_structure.program.functions["main"].block.operations:
        info = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if info is None:
            continue
        dev = type(info.preferred_compute_device).__name__
        tag = "ANE" if "Neural" in dev else ("GPU" if "GPU" in dev else "CPU")
        counts[tag] += 1
        op_dev.append((op.operator_name or "?", tag))
    print(f"\n=== {label} ===")
    print(f"  package: {mlpackage.name}")
    print(f"  totals: {dict(counts)}")
    for n, t in op_dev:
        print(f"    {n:32s} -> {t}")
    return dict(counts)


def build_m1_gather_then_conv(out: Path) -> None:
    """Gather K weight rows per batch element, project with a conv."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    rng = np.random.RandomState(0)
    weights = rng.randn(N_EXPERTS, D_FF).astype(np.float16)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(B, K), dtype=types.int32),
        mb.TensorSpec(shape=(B, D_FF, 1, 1), dtype=types.fp16),
    ], opset_version=ct.target.iOS18)
    def prog(idx, x):
        # gather K weight rows per batch; result [B, K, D_FF]
        w_const = mb.const(val=weights, name="W")
        gathered = mb.gather(x=w_const, indices=idx, axis=0,
                             validate_indices=False, name="gather_W")
        # reduce: take mean over K -> [B, D_FF]
        mean = mb.reduce_mean(x=gathered, axes=[1], keep_dims=False, name="mean_K")
        # reshape to NCHW for conv: [B, D_FF, 1, 1]
        r = mb.reshape(x=mean, shape=(B, D_FF, 1, 1), name="reshape")
        # tiny pointwise conv to give the graph some compute
        conv_w = mb.const(val=rng.randn(D_MODEL, D_FF, 1, 1).astype(np.float16),
                          name="cw")
        return mb.conv(x=r, weight=conv_w, name="conv_out")

    m = ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if out.exists():
        shutil.rmtree(out)
    m.save(str(out))


def build_m2_gather_reduce_combine(out: Path) -> None:
    """Mimic MoE: gather K full expert weight slabs, combine with weights, conv."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    rng = np.random.RandomState(1)
    # Each expert is a [D_MODEL, D_FF] slab packed as [N_EXPERTS, D_MODEL, D_FF]
    expert_w = rng.randn(N_EXPERTS, D_MODEL, D_FF).astype(np.float16)

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(B, K), dtype=types.int32),
        mb.TensorSpec(shape=(B, K), dtype=types.fp16),
        mb.TensorSpec(shape=(B, D_FF), dtype=types.fp16),
    ], opset_version=ct.target.iOS18)
    def prog(idx, gw, x):
        w_const = mb.const(val=expert_w, name="EW")
        # [B, K, D_MODEL, D_FF]
        gathered = mb.gather(x=w_const, indices=idx, axis=0,
                             validate_indices=False, name="gather_EW")
        # x: [B, D_FF] -> [B, 1, D_FF, 1]; matmul against [B, K, D_MODEL, D_FF]
        x_r = mb.reshape(x=x, shape=(B, 1, D_FF, 1), name="x_r")
        # batched matmul: (B,K,D_MODEL,D_FF) x (B,1,D_FF,1) -> (B,K,D_MODEL,1)
        out_per_k = mb.matmul(x=gathered, y=x_r, name="mm")
        # squeeze to [B, K, D_MODEL]
        sq = mb.reshape(x=out_per_k, shape=(B, K, D_MODEL), name="sq")
        # weight by gw [B, K] -> [B, K, 1]
        gw_r = mb.reshape(x=gw, shape=(B, K, 1), name="gw_r")
        weighted = mb.mul(x=sq, y=gw_r, name="weighted")
        # sum over K -> [B, D_MODEL]
        return mb.reduce_sum(x=weighted, axes=[1], keep_dims=False, name="combine")

    m = ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if out.exists():
        shutil.rmtree(out)
    m.save(str(out))


def build_m3_dense_baseline(out: Path) -> None:
    """Sanity: dense conv on same shape; should be all ANE.

    Use torch.jit.trace path to match how PF packs are built (which DO land
    on ANE). Direct mb.program construction may use different placement.
    """
    import coremltools as ct
    import torch
    import torch.nn as nn

    rng = np.random.RandomState(2)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(D_FF, D_MODEL, 1, bias=False)
            self.conv.weight = nn.Parameter(
                torch.from_numpy(rng.randn(D_MODEL, D_FF, 1, 1).astype(np.float32)),
                requires_grad=False)
        def forward(self, x):
            return self.conv(x)

    mod = M().eval()
    x_ex = torch.zeros(B, D_FF, 1, 1)
    with torch.no_grad():
        traced = torch.jit.trace(mod, (x_ex,))
    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(B, D_FF, 1, 1), dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    if out.exists():
        shutil.rmtree(out)
    m.save(str(out))


def main() -> int:
    _check_interp()
    TMP.mkdir(parents=True, exist_ok=True)
    import coremltools as ct
    print(f"coremltools {ct.__version__}")

    m1 = TMP / "M1_gather_then_conv.mlpackage"
    m2 = TMP / "M2_gather_reduce_combine.mlpackage"
    m3 = TMP / "M3_dense_baseline.mlpackage"

    print("\n[probe] building M1 (gather rows + conv)...")
    build_m1_gather_then_conv(m1)
    r1 = report_residency("M1: gather + reduce_mean + conv", m1)

    print("\n[probe] building M2 (gather expert slabs + matmul + combine)...")
    build_m2_gather_reduce_combine(m2)
    r2 = report_residency("M2: MoE-like gather + matmul + reduce_sum", m2)

    print("\n[probe] building M3 (dense conv baseline)...")
    build_m3_dense_baseline(m3)
    r3 = report_residency("M3: dense conv baseline", m3)

    print("\n=== VERDICT ===")
    def verdict(label, counts):
        ane = counts.get("ANE", 0)
        cpu = counts.get("CPU", 0)
        gpu = counts.get("GPU", 0)
        total = ane + cpu + gpu
        print(f"  {label}: ANE={ane}/{total}  CPU={cpu}  GPU={gpu}  "
              f"-> {'ANE-OK' if cpu == 0 and gpu == 0 else 'FALLS OFF ANE'}")
    verdict("M1 (toy gather)", r1)
    verdict("M2 (MoE-like)  ", r2)
    verdict("M3 (dense)     ", r3)

    return 0


if __name__ == "__main__":
    sys.exit(main())
