# Linear INT4 Per-Block Shard Bug: Why Your ANE Conv Ops Land on CPU

**Date**: 2026-04-24  
**Severity**: Critical — silently defeats the entire ANE mandate  
**Affects**: Tested small CoreML shards using linear INT4 per-block quantization
(`constexpr_blockwise_shift_scale`), not all 4-bit compression formats

---

## TL;DR

Linear INT4 per-block quantization (`constexpr_blockwise_shift_scale`) caused
**every `ios18.conv` op to fall off ANE to CPU** in our tested small shards
(2–8 transformer layers and isolated Conv probes).
The model compiles without error, runs without error, produces correct output — but
all the expensive matmul-as-conv projections execute on CPU, not ANE. You get ~5×
slower inference and burn CPU power instead of ANE power, with zero visible warning.

**INT8 per-tensor does not have this problem in our current Gemma shards.** It is
the production baseline. **INT4 palettization is a different compression family**
(`constexpr_lut_to_dense`) and must not be condemned by this result; it remains
promising but unvalidated for these Gemma shard patterns.

---

## Evidence

All measurements via `MLComputePlan` (ground-truth compiler scheduling, not heuristic):

| Model | Layers in package | Quant | Conv placement | Total CPU ops |
|-------|-------------------|-------|---------------|---------------|
| Qwen 0.5B monolithic | 24 | fp16 | **97/97 ANE** | 0 |
| Qwen 0.5B monolithic | 24 | INT8 | **97/97 ANE** | 0 |
| Qwen 0.5B monolithic, stateful | 24 | INT4 | **97/97 ANE** | 0 |
| Qwen 0.5B monolithic, non-stateful | 24 | INT4 | **97/97 CPU** | 3059 |
| Qwen 1.5B shard | 3 | INT4 | **12/12 CPU** | 66 |
| Qwen 3B shard | 2 | INT4 | **8/8 CPU** | 48 |
| **Qwen 3B shard** | **2** | **INT8** | **8/8 ANE** | **0** |
| Synthetic single conv (any dim) | — | INT4 | **CPU** | all |

The measured pattern is:
- linear INT4 per-block + small graph → CPU
- INT8 per-tensor + tested shard graphs → ANE
- linear INT4 per-block + large monolithic stateful graph → ANE in one Qwen case
   (compiler has enough context to fuse)

This table does **not** test 4-bit palettization (`constexpr_lut_to_dense`) or
W8A8 activation quantization.

---

## Why This Matters for Gemma

`gemma_to_ane.py` uses `--layer-start` / `--layer-end` sharding and its old
default path used `dtype="int4"` with `granularity="per_block"`.

Every Gemma shard built with that **linear INT4 per-block** path must be assumed
CPU until `MLComputePlan` proves otherwise. The shard will compile, load, and run
— but may run on CPU, not ANE. (Note: the old "96 MB shard cliff" is not the
cause — INT8 shards up to 223 MB compiled work fine on ANE.)

This means:
1. **All existing Gemma linear-INT4-per-block shards must be re-audited**, and
   known bad small shards should be treated as CPU, not ANE
2. **ANE residency probes that only check compilation success are insufficient** —
   you must check `MLComputePlan` placement
3. **Energy benchmarks on these shards measure CPU power, not ANE power**
4. **tok/s numbers from those shards reflect CPU throughput, not ANE throughput**

---

## Root Cause (Hypothesis)

The CoreML ANE compiler uses a cost model to decide whether to schedule an op
subgraph on ANE. Linear INT4 per-block `constexpr_blockwise_shift_scale` creates
a different dequant + conv pattern than our validated INT8 per-tensor path.

For a large monolithic model (24+ layers, stateful KV), the compiler sees enough
downstream ops to justify the ANE scheduling. For a small shard (2–8 layers), the
dequant+conv pattern doesn't meet the threshold, and the compiler falls back to CPU.

Key supporting observation: even a monolithic 24-layer INT4 model falls to CPU if it
is **non-stateful** — the KV cache state ops seem to tip the compiler's cost model
toward ANE scheduling.

---

## Fix

### Immediate (proven)

Use **INT8 per-tensor** for current production sharded models:

```python
# BEFORE (proven risky for tested small shards)
OpLinearQuantizerConfig(
    mode="linear_symmetric", dtype="int4",
    granularity="per_block", block_size=INT4_BLOCK_SIZE,
)

# AFTER (current production baseline)
OpLinearQuantizerConfig(
    mode="linear_symmetric", dtype="int8",
)
```

Tradeoff: 2× weight size vs 4-bit storage. For Gemma 4 26B-A4B with ~1.5 GB active weights at INT4,
INT8 gives ~3 GB active — still fits in 48 GB RAM, still bandwidth-feasible on M4 Max.

Keep INT8 as the production baseline until an alternative passes both:
- ANE residency via `MLComputePlan`
- golden quality via `golden-validator` / layer golden gates

### Future (unproven, needs investigation)

1. **Larger shards**: Pack more layers per shard to cross the compiler's fusion
   threshold. ~~Risk: may hit the 96 MB compiled-weight cliff.~~
   UPDATE (2026-04-24): The 96 MB cliff is wrong. INT8 shards up to 223 MB
   compiled are validated on ANE (Qwen 7B 1-layer probe, M4 Max).
2. **`compute_precision` hints**: Experiment with `ct.precision.FLOAT16` vs
   `ct.precision.FLOAT32` to see if the compiler changes scheduling.
3. **Different INT4 encoding**: `constexpr_affine_dequantize` (per-channel INT4)
   tested identically to blockwise on isolated probes — also CPU. But untested
   on medium-sized shards (8–12 layers).
4. **4-bit palettization** (`constexpr_lut_to_dense`): Different encoding path.
   Apple docs and CoreML-LLM point here for Neural Engine-friendly 4-bit weights.
   Next probe should use `OpPalettizerConfig(nbits=4, granularity="per_grouped_channel", group_size=32)`.
5. **W8A8**: Separate path for int8 activation + weight compute. Requires real
   activation calibration and compiler audit; do not conflate with weight-only INT4.
6. **File a Radar**: This is arguably a CoreML compiler bug — the placement
   should not silently change based on graph size.

---

## How to Check Your Shards

```python
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from collections import Counter

plan = MLComputePlan.load_from_path(
    "path/to/model.mlmodelc",
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
prog = plan.model_structure.program
for fn_name, fn in prog.functions.items():
    for op in fn.block.operations:
        if op.operator_name == "ios18.conv":
            dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = dev.preferred_compute_device.__class__.__name__
            name = "ANE" if "Neural" in d else ("GPU" if "GPU" in d else "CPU")
            print(f"conv → {name}")  # Must say ANE, not CPU
```

Run with Xcode python3 (`/Applications/Xcode.app/Contents/Developer/usr/bin/python3`)
which has coremltools 9.0.

---

## Action Items

- [ ] Add/keep `MLComputePlan` placement check in ane-validator agent (not just compile success)
- [ ] Re-test all existing Gemma shards for actual ANE placement
- [ ] Keep `gemma_to_ane.py` production sharded builds on INT8 per-tensor
- [x] Add a palettization probe for representative Gemma FFN / LM-head shard shapes
- [ ] Re-run energy benchmarks on INT8 shards (previous numbers measured CPU, not ANE)
- [x] Update `ANE_CHAIN_SCHEMA.md` — linear INT4 per-block caveat; palettization is separate
