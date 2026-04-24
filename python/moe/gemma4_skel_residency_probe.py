"""ane-validator probe for gemma4_skeleton_1L (T4.1.0).

Artifact: python/moe/out/gemma4_skeleton_1L.mlmodelc (already compiled).
Inputs:
  x: (1,1,2816) fp16
  attn_mask: (1,1,1,1024) fp16
  kv_write_mask: (1,1,1024,1) fp16
  States: k_cache_0 (1,8,1024,256) fp16, v_cache_0 same.
"""
import sys, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import coremltools as ct

PKG = Path("/Users/alvarovidela/Code/em2/python/moe/out/gemma4_skeleton_1L.mlpackage")
COMPILED = Path("/Users/alvarovidela/Code/em2/python/moe/out/gemma4_skeleton_1L.mlmodelc")


def main():
    print(f"package : {PKG}")
    print(f"compiled: {COMPILED}")

    # 1) Load + predict
    t0 = time.time()
    m = ct.models.MLModel(str(PKG), compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"LOAD: OK ({time.time()-t0:.2f}s)")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 2816)).astype(np.float16) * 0.1
    attn_mask = np.zeros((1, 1, 1, 1024), dtype=np.float16)
    kv_write_mask = np.zeros((1, 1, 1024, 1), dtype=np.float16)
    kv_write_mask[..., 0, 0] = 1.0

    feed = {"x": x, "attn_mask": attn_mask, "kv_write_mask": kv_write_mask}

    state = None
    try:
        state = m.make_state()
    except Exception as e:
        print(f"make_state: not available ({e!r})")

    try:
        t0 = time.time()
        if state is not None:
            y = m.predict(feed, state=state)
        else:
            y = m.predict(feed)
        dt = (time.time() - t0) * 1000
        print(f"PREDICT: OK  latency={dt:.2f} ms  outputs={list(y.keys())}")
    except Exception as e:
        print(f"PREDICT: FAIL ({e!r})")
        return 2

    lat = []
    for _ in range(5):
        t0 = time.time()
        m.predict(feed, state=state) if state is not None else m.predict(feed)
        lat.append((time.time() - t0) * 1000)
    print(f"PREDICT (5x warm): min={min(lat):.2f} ms  mean={sum(lat)/len(lat):.2f} ms")

    # 2) Compute plan
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(
        path=str(COMPILED),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    if program is None:
        print("COMPUTE_PLAN: no mlprogram structure")
        return 3

    # tally
    dev_counts = Counter()
    op_dev = Counter()  # (operator_name, device) -> count
    linear_rows = []  # for big linears we care about
    for fn_name, fn in program.functions.items():
        print(f"\nfunction {fn_name}: {len(fn.block.operations)} ops")
        for op in fn.block.operations:
            try:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                cost = plan.get_estimated_cost_for_mlprogram_operation(op)
                devn = dev.preferred_compute_device.__class__.__name__ if dev else "?"
            except Exception as e:
                devn = f"ERR:{e!r}"
                cost = None
            dev_counts[devn] += 1
            op_dev[(op.operator_name, devn)] += 1

            if op.operator_name in ("linear", "matmul", "conv", "constexpr_lut_to_dense",
                                    "constexpr_affine_dequantize", "constexpr_blockwise_shift_scale"):
                cw = f"{cost.weight:.4f}" if cost is not None else "-"
                # try to extract output shape
                try:
                    out_t = op.outputs[0]
                    shp = tuple(out_t.shape) if hasattr(out_t, "shape") else None
                except Exception:
                    shp = None
                linear_rows.append((op.operator_name, shp, devn, cw))

    print("\n=== device tally (all ops) ===")
    for d, c in dev_counts.most_common():
        print(f"  {d:<24} {c}")

    print("\n=== op x device tally ===")
    for (opn, dev), c in sorted(op_dev.items()):
        print(f"  {opn:<32} {dev:<24} {c}")

    print("\n=== heavy ops (linear/matmul/conv/quant) ===")
    print(f"  {'op':<34} {'shape':<28} {'device':<20} {'cost':>8}")
    for opn, shp, dev, cw in linear_rows:
        print(f"  {opn:<34} {str(shp):<28} {dev:<20} {cw:>8}")

    # verdict
    cpu_count = sum(c for d, c in dev_counts.items() if "CPU" in d)
    gpu_count = sum(c for d, c in dev_counts.items() if "GPU" in d)
    ane_count = sum(c for d, c in dev_counts.items() if "Neural" in d or "ANE" in d)
    print(f"\nANE={ane_count}  GPU={gpu_count}  CPU={cpu_count}")
    if cpu_count == 0 and gpu_count == 0:
        print("VERDICT: PASS — all ops on ANE")
        return 0
    else:
        print("VERDICT: see CPU/GPU rows above")
        return 0


if __name__ == "__main__":
    sys.exit(main())
