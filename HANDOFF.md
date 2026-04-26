# Agent Handoff — Gemma-4 ANE Inference (April 2026)

## Branch & Repo

- **Repo**: `git@github.com:videlalvaro/emilio.git`
- **Branch**: `experimental/gemma4-ane-full` (pushed 2026-04-26)
- **Parent branch**: `feat/gemma-ane-decode` (the working base before this session)
- **8 grouped commits** on top of parent — see `git log --oneline` for the full list

## What This Project Is

Running **Gemma-4-26B-A4B** (Google's MoE model: 26B total params, 4B active) entirely on **Apple Neural Engine** via CoreML. ALL compute-heavy inference (matmuls, norms, activations, projections) MUST run on ANE — CPU/GPU fallback is NOT acceptable (see `.github/copilot-instructions.md`).

The project has successfully converted all 30 transformer layers into 90 ANE shards (3 per layer: 1 attn + 2 FFN partials) with INT8 quantization, validated quality via golden cosine gates, and built a Swift runtime driver. It is NOT yet generating correct long-form text end-to-end.

## Current State (What Works)

### Proven & Validated
- **90/90 CoreML ANE shards** compile and run 100% on ANE (zero CPU fallback)
- **Architecture**: 1 attn shard (33-47 MB) + 1 FFN partial (182 MB) + 1 merged FFN partial+combiner (227 MB, mixed INT8/fp16) per layer
- **Golden validation** on 7 sampled layers: cos(hidden) 0.9555–0.9999, cos(attn) >0.9997
- **Swift runtime** (`emilio/conv-ane/gemma_ane.swift`): expert-groups dispatch, sliding/global window, per-expert and packed modes
- **FP16 variant** also proven: 270 shards (30 attn + 240 FFN), min cos=0.9994, generates coherent text ("The capital of France is a city of romance, art, and")
- **Per-expert MoE dispatch** validated on privacy-filter model: 35× vs dense, 100% public CoreML API

### Key Empirical Laws (Validated)
- **~250 MB compiled shard limit** for ANE (empirically validated to 223 MB on M4 Max)
- **INT8 per_tensor** is production baseline for shards
- **INT8 per_block (`constexpr_blockwise_shift_scale`)** poisons entire graph to 0% ANE — NEVER use
- **Merged combiner** pattern eliminates CPU fallback (standalone small shards fall to CPU)
- **`weight_threshold=10_000_000`** keeps dense MLP weights fp16 while INT8-quantizing expert packs
- **Rank-2 tensors** → 0% ANE (need rank-3 minimum)
- ANE driver weight cache holds **1024+ distinct .mlmodelc** with no eviction

## What Doesn't Work Yet (Open Tasks)

### T4.1.5 — Decode loop with KV state read/write
- Gate: needs a low-entropy long-decode oracle before checking off
- Current: bounded 2-step decode smoke passes, but full-horizon decode diverges by row 7

### T4.3 — Full prefill+decode validation
- Gate: full prefill+decode logit cosine ≥ 0.999 vs golden, MMLU within 2pp
- **Root cause identified**: diffuse gamma-weighted amplification in final norm. Small distributed hidden-state drift across 30 INT8 layers gets amplified by the final gamma vector until near-tie token orderings invert (e.g., token 506 vs 9405 margin flips at decode step 5)
- The drift is NOT a single-layer or single-head bug — top 12 contributing dimensions explain only 9.89% of margin contribution
- **Active investigation**: the harmful drift is already in the residual carry feeding layer 28; attention delta is mostly fine; FFN partially cancels rather than causes the drift
- **Possible fix paths**: (a) FP16 shards (proven to work but 270 shards), (b) mixed INT8/FP16 for sensitive layers, (c) W8A8 activation quantization (future), (d) INT4 palettization (`constexpr_lut_to_dense` — promising but unvalidated)

### T4.4 — Energy/perf benchmarking
- Blocked on T4.3 (quality gate must pass first)

### T3 — MoBE (Mixture of Basis Experts)
- Not started. Would reduce expert storage via shared basis matrices

## Disk State (IMPORTANT — Artifacts Deleted)

The previous session crashed the machine (OOM + disk full from loading 300+ CoreML models). **All derived artifacts and model weights were deleted** to recover 204 GB. The source code and rebuild scripts are intact.

### What Was Deleted
- `models/gemma-4-26b-a4b/` (48 GB) — re-download from HuggingFace
- `models/qwen3.6-35b-a3b/` (67 GB) — re-download from HuggingFace
- `python/moe/out/` contents — expert_groups, lm_head_shards, all compiled shards
- `python/tmp/` — all probe artifacts
- `emilio/conv-ane/*.mlmodelc` — all compiled CoreML models
- `weights/` — intermediate weight dumps

### How to Rebuild Everything

1. **Download base model**:
   ```bash
   # In .venv313 (HF/transformers env)
   huggingface-cli download google/gemma-4-26b-a4b-it --local-dir models/gemma-4-26b-a4b
   ```

2. **Pack layers** (extracts per-layer weights from HF checkpoint):
   ```bash
   # In .venv313
   python python/moe/pack_gemma_layer.py --layer 0  # repeat for 0-29
   ```

3. **Convert to ANE shards** (INT8, 3 shards per layer):
   ```bash
   # In Xcode python3 (coremltools 9 ONLY works here)
   python3 python/moe/gemma_to_ane.py --layer 0 --max-ctx 128 --ffn-shards 2 --quant int8
   # Or batch: tmp/convert_all_30.sh
   ```

4. **Compile shards**:
   ```bash
   xcrun coremlcompiler compile python/moe/out/gemma4_shard0_1_real_attn_q8.mlpackage python/moe/out/
   # Repeat for all .mlpackage files
   ```

5. **Build Swift runtime**:
   ```bash
   cd emilio/conv-ane
   swiftc -O -o gemma_ane gemma_ane.swift -framework CoreML -framework Foundation
   ```

6. **Build expert groups** (optional, for G=8 dispatch):
   ```bash
   python3 python/moe/build_gemma_expert_groups.py --layer 0 --group-size 8
   ```

7. **Build LM head shards** (for >250 MB vocab split):
   ```bash
   python3 python/moe/build_gemma_lm_head_shards.py
   ```

## Python Environments (NOT interchangeable)

| Env | Activate | Purpose |
|-----|----------|---------|
| `.venv` | `source .venv/bin/activate` | PyTorch only (probes, golden capture) |
| `.venv313` | `source .venv313/bin/activate` | HF transformers (model download, reference inference) |
| Xcode `python3` | `/usr/bin/python3` or just `python3` | **coremltools 9** — ONLY env that can convert to CoreML |

## Key Files

### Conversion Pipeline
| File | Purpose |
|------|---------|
| `python/moe/gemma_to_ane.py` | Core converter: HF weights → CoreML .mlpackage (attn/ffn split, INT8, mixed quant) |
| `python/moe/pack_gemma_layer.py` | Extracts per-layer weights from HF checkpoint into standalone format |
| `python/moe/export_gemma_swift_head.py` | Exports LM head + metadata for Swift runtime |
| `python/moe/build_gemma_expert_groups.py` | Builds G=8 expert group shards |
| `python/moe/build_gemma_per_expert.py` | Builds individual per-expert shards |
| `python/moe/build_gemma_lm_head_shards.py` | Splits LM head into vocab-half ANE shards |

### Swift Runtime
| File | Purpose |
|------|---------|
| `emilio/conv-ane/gemma_ane.swift` | Main Swift inference driver (1265 lines) |
| `emilio/conv-ane/gguf_to_ane.py` | Qwen GGUF→ANE converter (separate model family) |

### Validation & Quality Gates
| File | Purpose |
|------|---------|
| `python/moe/gemma_swift_logit_gate.py` | Prefill-time logit cosine gate vs golden |
| `python/moe/gemma_swift_decode_logit_gate.py` | Decode-time logit validation |
| `python/moe/gemma_split_golden.py` | Per-layer golden logit capture |
| `python/moe/gemma_hf_logits_capture_reap.py` | REAP-aware golden capture from HF reference |

### ANE Residency Probes
| File | Purpose |
|------|---------|
| `python/moe/ane_residency_check_l5.py` | Single-layer ANE placement check |
| `python/moe/gemma_full_residency_sweep.py` | Full 30-layer sweep |
| `python/moe/gemma4_combiner_residency_probe.py` | Merged combiner probe |
| `python/moe/gemma_palettization_residency_probe.py` | INT4 palettization probe |
| `python/moe/validate_ffn_subsplit_ane.py` | FFN sub-split validation |

### Research Docs
| File | Purpose |
|------|---------|
| `python/moe/GEMMA_ANE_RESEARCH.md` | Master research plan with all TODOs and status |
| `emilio/conv-ane/ANE_CHAIN_SCHEMA.md` | Empirical ANE laws (shard limits, quant rules) |
| `docs/INT4_SHARD_ANE_BUG.md` | INT4 per-block quantization bug documentation |
| `.github/copilot-instructions.md` | Project policy (ANE-only mandate, hard rules) |

## Agent Team (`.github/agents/`)

| Agent | Role |
|-------|------|
| `optimality-gatekeeper` | MUST approve any long-running task (>60s) before execution |
| `ane-validator` | Verifies CoreML op pattern lands on ANE before scaling |
| `golden-validator` | Checks model output quality vs saved golden logits |
| `compiler` | Handles CoreML/Swift/Xcode builds |
| `tester` | Runs unit/smoke tests |
| `energy-bencher` | Power/energy measurements (requires gatekeeper approval) |
| `historian` | Records decisions/findings for the book |
| `doc-writer` | Updates research docs |

**Default workflow**: historian (record intent) → optimality-gatekeeper (review) → ane-validator + golden-validator (cheap gates) → compiler → tester → energy-bencher → doc-writer → historian (record outcome)

## Critical Gotchas

1. **ANE output strides**: ANE `MLMultiArray` outputs have `stride[1] = 32` (not 1). Must use stride-aware pointer access. Reading naively yields cos≈0.
2. **ANE execution plan limit**: ~120 concurrent plans. Loading 300+ shards simultaneously OOMs the machine.
3. **INT8 per_block is BROKEN on ANE**: `constexpr_blockwise_shift_scale` poisons entire graph to CPU. Use `per_tensor` only.
4. **Golden validator context**: must use `--max-ctx 128` to match batch conversion's `--max-ctx 128`.
5. **coremltools 9**: ONLY available in Xcode's system `python3`. The `.venv` and `.venv313` envs cannot convert models.
6. **OOM risk**: Never load all 90 shards simultaneously. The Swift runtime should load/unload per-layer.
7. **Attention scaling**: Gemma-4 uses `scaling=1.0` (q_norm/k_norm provide magnitude control). Do NOT add `1/√d` scaling — it was a prior bug that took 3× cos improvement to fix.

## Recommended Next Steps (Priority Order)

1. **Re-download Gemma-4 weights** from HuggingFace (~48 GB, ~30 min)
2. **Rebuild the 90 ANE shards** using `tmp/convert_all_30.sh` (~2h total)
3. **Attack T4.1.5** (decode loop with KV state) — this is the main blocker
4. **Attack T4.3** (full decode quality gate) — may require mixed INT8/FP16 for layers 25-29 where quantization drift compounds most
5. **Consider FP16 path** as the pragmatic shipping configuration (270 shards, proven correct text generation, ~3× more shards but quality gate passes cleanly)

## Knowledge Sources

- `BOOK_ANALYSIS.md` — 9 classic CS books with optimization patterns
- `python/moe/GEMMA_ANE_RESEARCH.md` — full research plan + arxiv synthesis
- `emilio/conv-ane/ANE_CHAIN_SCHEMA.md` — empirical ANE laws
- `memory:repo/ane-scaling-research.md` — detailed experiment log
- `memory:repo/gol-ane.md` — Game of Life ANE work (separate track)
