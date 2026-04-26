# Project policy: ANE / MoE inference work

This repo runs expensive, hardware-bound experiments (CoreML conversions, calibrations,
powermetrics sweeps, ANE residency probes). Bad code wastes hours and burns laptop battery.

## ANE-ONLY mandate

**ALL compute-heavy inference MUST run on the Apple Neural Engine.**
CPU/GPU fallback for matmuls, attention, FFN, norms, or LM head projection is NOT acceptable.
If something currently runs on CPU (like the host-side LM head), the next step is ALWAYS to
move it to ANE — never to optimise the CPU path. People who want CPU/GPU inference already
have llama.cpp, MLX, and dozens of other options.

Permitted CPU-side exceptions (non-compute, unavoidable host work):
- Tokenizer encode/decode (string processing, BPE merges)
- Embedding table lookup (index → vector, not matmul)
- Sampling (argmax / top-p / temperature — trivial)
- Meta JSON parsing, file I/O, model loading orchestration
- KV cache position bookkeeping (write-mask construction)

Everything else — every matmul, every norm, every activation, every projection including
the LM head — MUST be in a CoreML `.mlpackage` targeting ANE. The current
production baseline for Gemma shards is INT8 per-tensor weight quantisation.
Do not generalize the old INT4 shard bug: the measured failure is linear INT4
per-block quantisation (`constexpr_blockwise_shift_scale`) on small sharded
Conv/Linear graphs. INT4 per-grouped-channel palettization
(`constexpr_lut_to_dense`) is a separate, promising but unvalidated path and
must pass ANE residency plus golden quality before scale-out. W8A8 is a separate
future path requiring real activation calibration and compiler audit.
The compiled shard limit is ~250 MB (empirically validated to 223 MB on M4 Max).
When a single op is too large for one ANE shard (>250 MB compiled), split it into
multiple ANE shards (e.g. vocab-half LM head shards), do NOT fall back to CPU/Accelerate.

## Hard rules

1. **Never start a long-running task without the optimality-gatekeeper agent's approval.**
   "Long-running" = anything > 60 seconds, anything that loads the full Gemma-4 model,
   any CoreML conversion, any calibration pass, any `powermetrics` capture, any model download.
2. **Quality gate before perf**: any model artifact that will be benchmarked or shipped MUST
   pass `golden-validator` (cosine ≥ 0.97 vs `python/moe/out/gemma_golden.npz`) first.
3. **ANE residency gate before scale**: any new CoreML op pattern MUST pass `ane-validator`
   on the smallest representative shape before being applied to all 30 layers.
4. **No silent destructive ops**: never delete `.mlpackage`, `.npz` calibration data, or
   anything under `python/moe/out/` or `models/` without explicit user confirmation.
5. **Three Python envs are not interchangeable** — see `python/moe/GEMMA_ANE_RESEARCH.md`.
   `.venv` (torch only), `.venv313` (HF/transformers), Xcode `python3` (coremltools 9 only).
6. **No CPU/GPU compute shortcuts**: never optimise a host-side matmul/norm/projection as a
   substitute for moving it to ANE. The Accelerate/vDSP path is a temporary stopgap only —
   the real fix is always an ANE shard.

## Knowledge sources optimization decisions must reference

- `BOOK_ANALYSIS.md` — 9 classic CS books with optimization patterns (Knuth, Iverson,
  Dragon Book, Concrete Math, etc.). Cite a book chapter when proposing an optimization.
- `python/moe/GEMMA_ANE_RESEARCH.md` — current Gemma-on-ANE plan + arxiv synthesis
  (REAP, MoBA, MoBE, FlashMoE, VEQ).
- `emilio/conv-ane/ANE_CHAIN_SCHEMA.md` — empirical ANE laws (INT8 production
   baseline for shards, linear INT4 per-block fallback risk, G=8 sweet spot,
   ~250 MB shard limit).
- `memory:repo/ane-scaling-research.md` and `memory:repo/gol-ane.md`.

## Team

See `.github/agents/`. The default flow for any new technique:
`historian` (record intent) → `optimality-gatekeeper` (review) →
`ane-validator` + `golden-validator` (cheap gates) → `compiler` → `tester` →
`energy-bencher` (only if perf-relevant) → `doc-writer` → `historian` (record outcome).
