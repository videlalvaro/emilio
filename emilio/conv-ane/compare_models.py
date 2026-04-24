#!/usr/bin/env python3
"""Compare all layer weights between GPTQ-woodbury and RTN models.
Focus: find which layers diverge most, and check biases."""
import sys
sys.path.insert(0, "/Users/alvarovidela/Library/Python/3.9/lib/python/site-packages")
import coremltools as ct
from coremltools.optimize.coreml import get_weights_metadata
import numpy as np

print("Loading GPTQ model...")
gptq_m = ct.models.MLModel("QwenANE_28L_stateful_q4_gptq-woodbury.mlpackage",
                            compute_units=ct.ComputeUnit.CPU_ONLY)
print("Loading RTN model...")
rtn_m = ct.models.MLModel("QwenANE_28L_stateful_q4.mlpackage",
                           compute_units=ct.ComputeUnit.CPU_ONLY)

print("Getting GPTQ weights metadata...")
gptq_wm = get_weights_metadata(gptq_m)
print("Getting RTN weights metadata...")
rtn_wm = get_weights_metadata(rtn_m)

def strip_suffix(k):
    """Remove trailing _0 or _1 from op names for matching."""
    if k.endswith("_0") or k.endswith("_1"):
        return k[:-2]
    return k

gptq_dict = {strip_suffix(k): v.val for k, v in gptq_wm.items() if v.val is not None}
rtn_dict = {strip_suffix(k): v.val for k, v in rtn_wm.items() if v.val is not None}

common = sorted(set(gptq_dict.keys()) & set(rtn_dict.keys()))
print(f"\nCommon ops: {len(common)}")
print(f"GPTQ-only: {sorted(set(gptq_dict.keys()) - set(rtn_dict.keys()))[:5]}")
print(f"RTN-only:  {sorted(set(rtn_dict.keys()) - set(gptq_dict.keys()))[:5]}")

print(f"\n{'Op name':<65} {'Shape':>15} {'MaxDiff':>10} {'MeanDiff':>10} {'Match%':>7}")
print("-" * 112)

for k in common:
    g = gptq_dict[k].astype(np.float32).flatten()
    r = rtn_dict[k].astype(np.float32).flatten()
    if g.shape != r.shape:
        print(f"  {k}: SHAPE MISMATCH {g.shape} vs {r.shape}")
        continue
    diff = np.abs(g - r)
    tag = k.replace("_weight_quantized", "").replace("_weight", "")
    shape_str = str(tuple(gptq_dict[k].shape))
    match_pct = (diff == 0).sum() / len(diff) * 100
    print(f"  {tag:<63} {shape_str:>15} {diff.max():>10.6f} {diff.mean():>10.6f} {match_pct:>6.1f}%")

# Check if there are any non-quantized weights (biases, norms)
print("\n\nNon-quantized ops (biases, norms):")
for k in common:
    if "data" not in k and "scale" not in k:
        g = gptq_dict[k].astype(np.float32).flatten()
        r = rtn_dict[k].astype(np.float32).flatten()
        diff = np.abs(g - r)
        if diff.max() > 0:
            print(f"  {k}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
        else:
            print(f"  {k}: IDENTICAL")
