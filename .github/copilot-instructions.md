# Project policy: ANE / MoE inference work

This repo runs expensive, hardware-bound experiments (CoreML conversions, calibrations,
powermetrics sweeps, ANE residency probes). Bad code wastes hours and burns laptop battery.

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

## Knowledge sources optimization decisions must reference

- `BOOK_ANALYSIS.md` — 9 classic CS books with optimization patterns (Knuth, Iverson,
  Dragon Book, Concrete Math, etc.). Cite a book chapter when proposing an optimization.
- `python/moe/GEMMA_ANE_RESEARCH.md` — current Gemma-on-ANE plan + arxiv synthesis
  (REAP, MoBA, MoBE, FlashMoE, VEQ).
- `emilio/conv-ane/ANE_CHAIN_SCHEMA.md` — empirical ANE laws (INT4 mandatory,
  G=8 sweet spot, 96 MB cliff).
- `memory:repo/ane-scaling-research.md` and `memory:repo/gol-ane.md`.

## Team

See `.github/agents/`. The default flow for any new technique:
`historian` (record intent) → `optimality-gatekeeper` (review) →
`ane-validator` + `golden-validator` (cheap gates) → `compiler` → `tester` →
`energy-bencher` (only if perf-relevant) → `doc-writer` → `historian` (record outcome).
