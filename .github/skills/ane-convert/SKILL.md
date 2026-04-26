---
name: ane-convert
description: 'Convert GGUF models to Apple Neural Engine CoreML shards. Use for: ANE conversion, GGUF to CoreML, ANE feasibility analysis, shard planning, model quantization for ANE, INT4/INT8 ANE placement, build mlpackage, validate ANE residency, check compute unit placement. Triggers: "convert to ANE", "analyze for ANE", "plan shards", "check ANE placement", "GGUF to CoreML".'
argument-hint: 'Path to GGUF model or conversion task description'
---

# ANE Conversion Skill

Convert GGUF language models to Apple Neural Engine CoreML shards using the `ane_sdk` framework.

## When to Use

- User wants to convert a GGUF model to run on Apple Neural Engine
- User asks about ANE feasibility for a model
- User wants to plan how a model should be sharded for ANE
- User needs to validate ANE residency (compute unit placement) of built shards
- User asks about INT4 vs INT8 quantization for ANE

## Tools

### CLI (`python -m ane_sdk`)

All commands require `PYTHONPATH="$PWD/python:$PYTHONPATH"` and use `.venv/bin/python`.

```bash
# Analyze feasibility (quick, no conversion)
PYTHONPATH="$PWD/python:$PYTHONPATH" .venv/bin/python -m ane_sdk analyze <model.gguf> --quant int8

# Plan sharding (show shard boundaries)
PYTHONPATH="$PWD/python:$PYTHONPATH" .venv/bin/python -m ane_sdk plan <model.gguf> --seq-len 2048 --quant int8

# Full conversion pipeline (long-running — needs gatekeeper approval)
PYTHONPATH="$PWD/python:$PYTHONPATH" .venv/bin/python -m ane_sdk convert <model.gguf> --seq-len 2048 --quant int8 --output-dir <dir>
```

### Python API

```python
from ane_sdk import GGUFReader, ANEAnalyzer, ShardPlanner, ANEConverter

reader = GGUFReader("model.gguf")
report = ANEAnalyzer.analyze(reader, max_seq_len=2048, target_quant="int8")
plan = ShardPlanner.plan(reader, report, quant_bits=8)
ANEConverter(reader, plan, output_dir="./out").convert()
```

### Standalone scripts

- `emilio/conv-ane/gguf_to_ane.py` — Low-level single-shard builder (Xcode python3 only)
- `emilio/conv-ane/build_lm_head_shards.py` — LM head shard builder (accepts `--quant-bits {0,4,8}`)

## CRITICAL: Compression Families Are Not Interchangeable

The measured shard bug is **linear INT4 per-block quantization**
(`OpLinearQuantizerConfig(dtype="int4", granularity="per_block")`, emitted as
`constexpr_blockwise_shift_scale`) on small sharded Conv/Linear graphs. Those
models compile and run without error, but the expensive projections can execute
on CPU silently. See [INT4_SHARD_ANE_BUG.md](../../../docs/INT4_SHARD_ANE_BUG.md).

Do not turn that into "INT4 never works on ANE." Apple docs and CoreML-LLM point
to a separate 4-bit path: **palettization**
(`OpPalettizerConfig(nbits=4, granularity="per_grouped_channel", group_size=32)`,
emitted as `constexpr_lut_to_dense`). That path is promising but not production
for this repo until it passes both residency and golden-quality gates.

**Rules:**
1. Use **INT8 per-tensor** as the production baseline for current sharded Gemma builds.
2. Treat **linear INT4 per-block** as proven risky on small sharded Conv/Linear graphs.
3. Treat **INT4 per-grouped-channel palettization** as experimental until the exact shard pattern passes `ane-validator` and `golden-validator`.
4. Treat **W8A8** as a separate future path requiring real activation calibration and compiler audit.
5. After ANY conversion, validate ANE placement — compilation success is NOT sufficient.

## Procedure

### 1. Analyze

Run `analyze` to check feasibility before committing to a long conversion:

```bash
PYTHONPATH="$PWD/python:$PYTHONPATH" .venv/bin/python -m ane_sdk analyze models/<model>.gguf --quant int8
```

Check: `can_run_sharded: true`, `theoretical_tok_s`, and `total_weight_mb`.

### 2. Get Gatekeeper Approval

**Conversion is a long-running task (> 60 s).** Per project policy, invoke the
`optimality-gatekeeper` agent before starting. Provide:
- Model name, param count, active weight size
- Number of shards, estimated time
- Quantization: INT8 per-tensor production baseline, or the exact experimental
  compression family plus its planned residency/quality gates

### 3. Convert

```bash
PYTHONPATH="$PWD/python:$PYTHONPATH" .venv/bin/python -m ane_sdk convert models/<model>.gguf \
  --seq-len 2048 --quant int8 --output-dir emilio/conv-ane/<output_dir> \
  2>&1 | tee tmp/<model>_conversion.log
```

Run in async mode — each shard takes ~2 min. Monitor with:
```bash
ls -d <output_dir>/*.mlpackage | wc -l   # count built shards
ps aux | grep -E 'gguf_to_ane|build_lm_head' | grep -v grep  # check running process
```

### 4. Validate ANE Residency

After conversion, check that conv ops land on ANE using `MLComputePlan`:

```python
# Run with Xcode python3 (has coremltools 9)
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

plan = MLComputePlan.load_from_path("path/to/shard.mlmodelc", compute_units=ct.ComputeUnit.CPU_AND_NE)
prog = plan.model_structure.program
for fn_name, fn in prog.functions.items():
    for op in fn.block.operations:
        if op.operator_name == "ios18.conv":
            dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = dev.preferred_compute_device.__class__.__name__
            name = "ANE" if "Neural" in d else ("GPU" if "GPU" in d else "CPU")
            print(f"conv → {name}")  # MUST say ANE
```

Or use the `ane-validator` agent on a representative shard.

**PASS criteria:** All `ios18.conv` ops on ANE, zero CPU compute ops.

### 5. Quality Validation (if golden reference exists)

Use `golden-validator` agent to check cosine similarity ≥ 0.97 vs reference logits
before benchmarking or shipping.

## Python Environments

Three environments, NOT interchangeable:

| Env | Python | Use for |
|-----|--------|---------|
| `.venv` | 3.9.6 + torch | `ane_sdk` CLI, analysis, planning |
| `.venv313` | 3.13 + HF/transformers | Golden reference generation |
| Xcode `python3` | System + coremltools 9 | CoreML conversion, `gguf_to_ane.py`, `build_lm_head_shards.py`, ANE placement checks |

The `ane_sdk convert` command automatically delegates to Xcode python3 for CoreML operations.

## ANE Constraints Reference

- **~250 MB compiled shard limit** — empirically validated to 223 MB (7B 1L INT8 on M4 Max). Old "96 MB cliff" was wrong.
- **INT8 per-tensor is the production baseline for current shards** — linear INT4 per-block caused silent CPU fallback in tested small shards; palettization is a separate unvalidated path.
- **Stateful models** must use `.all` compute mode at runtime
- **Conv2d(1×1) pattern** for all matmuls, fp16 activations throughout
- **Per-op weight sweet spot**: 12–50 MB (group_size=8 ≈ 24 MB ideal)
- **ANE bandwidth**: ~100 GB/s → `tok/s ≈ 100 GB/s ÷ active_weight_bytes × (1/1.2)`

## File Locations

| What | Path |
|------|------|
| SDK source | `python/ane_sdk/` (6 modules) |
| CLI entry | `python/ane_sdk/__main__.py` |
| Low-level converter | `emilio/conv-ane/gguf_to_ane.py` |
| LM head builder | `emilio/conv-ane/build_lm_head_shards.py` |
| ANE empirical laws | `emilio/conv-ane/ANE_CHAIN_SCHEMA.md` |
| INT4 bug writeup | `docs/INT4_SHARD_ANE_BUG.md` |
| GGUF models | `models/*.gguf` |
| Conversion outputs | `emilio/conv-ane/qwen_*_ane*/` |

## Proven Models

| Model | Quant | Shards | tok/s | Status |
|-------|-------|--------|-------|--------|
| Qwen 2.5-0.5B | fp16 | 1 (mono) | 167 | Ship-ready |
| Qwen 2.5-1.5B | linear INT4 per-block | 7+2 | 9.9 | Not production; re-test with INT8 or palettization |
| Qwen 2.5-3B | INT8 | 18+4 | ~20 est | Converted, ANE validated |
