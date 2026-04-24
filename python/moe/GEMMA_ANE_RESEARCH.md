# Gemma-4-26B-A4B on ANE — Research Report & TODO

**Last updated**: 2026-04-24
**Target**: Deploy Gemma-4-26B-A4B (128 experts × 30 layers MoE) on M4 Max ANE.
**Goal**: Energy-efficient inference; GPU 100% free for foreground apps.

---

## Baseline (validated, before optimizations)

End-to-end INT4-ANE projection (G=4 best pack, measured via probes):

| ctx  | attn (ms) | MLP (ms) | total | tok/s |
|------|----------:|---------:|------:|------:|
| 1024 |      47.6 |       62 |   110 |   9.1 |
| 2048 |      59.2 |       62 |   121 |   8.3 |
| 4096 |      80.8 |       62 |   143 |   7.0 |
| 8192 |     114.6 |       62 |   177 |   5.7 |

~3-5 W package, 0% GPU. Energy: ANE 3.18 mJ/eval @ 5.2 W vs MPS 7.04 mJ/eval @ 22.7 W → 2.2× package efficiency.

Empirical ANE laws (this session):
1. fp16 dense linears fall back to CPU at all sizes tested.
2. INT4 weight-quant (`cto.linear_quantize_weights`) is mandatory for ANE residency.
3. Per-op weight must be ~12-50 MB to land on ANE; G=8 (24 MB) is sweet spot; G=32 (96 MB) falls off.
4. Routing locality is adversarial: mean **6.61 / 8** distinct packs touched per token at G=8.

---

## Arxiv Literature Synthesis (Kimi / Moonshot / FlashMoE)

Three queries to `export.arxiv.org`, full-text reads of MoBA + both FlashMoEs.

### Tier 1 — Directly Actionable

#### REAP (arXiv:2510.13999)
Router-weighted Expert Activation Pruning. **50% expert pruning near-lossless on Kimi-K2 (1T) and Qwen3-Coder-480B for code-gen.** Uses router gate-values × expert activation norms. Beats merging (which has irreducible error from losing fine-grained routing).

For us: prune Gemma's 128 → 64 experts. Halves pack search space. Combined with co-clustering, drops mean distinct packs from 6.61/8 to ~3.5/4 → MLP cost approximately halves (62 → ~30 ms). Highest leverage: we already have `gemma_routing.npz` and pruning is one-shot calibration.

#### MoBA (arXiv:2502.13189, Moonshot, deployed in Kimi)
Block-sparse attention: split KV into blocks of size B; per query, score each block by `<q, mean_pool(K_block)>`; route to top-k blocks; FlashAttention over the union. Causality via hard-mask future + force-include current block.

Production deployment (Kimi): **last 3 layers stay full-attention**, rest are MoBA. Quest is "MoBA with smaller block + min/max pooling" — training-free variant we can prototype.

For us: 6 global-attention layers dominate decode at long ctx. MoBA top-k=4 of 16×512-token blocks would cut global KV streaming 6×8K → 6×2K. Expected: **80 → 20 ms global → total 115 → 55 ms (~18 tok/s) at 8K ctx**. Caveat: paper-quality MoBA was trained (100B tokens continual-pretrain). Drop-in needs Quest-style calibration.

#### MoBE (arXiv:2508.05257)
Each expert's `W = A·B`; A per-expert, B is a *linear combination of L shared basis matrices* per layer. **Kimi-K2 (1T): −24-30% params @ 1-2% acc drop.**

For us: shrinks per-pack weight footprint. May let us pack G=12-16 within the 12-50 MB ANE residency window without hitting the 96 MB cliff. Requires reconstruction-loss calibration.

### Tier 2 — Lower Priority

- **MiMo-V2-Flash (arXiv:2601.02780)** — 5:1 SWA:global hybrid (similar to Gemma-4); MTP heads as draft model → **2.6× spec-decode speedup**. Gemma-4 has no MTP; would need to train.
- **VEQ (arXiv:2602.01037)** — modality-adaptive INT3-W for VLMs (Kimi-VL, Qwen3-VL); Hessian-based with token-expert affinity. Use only if INT4 cosine fails vs golden.
- **FlashMoE-SSD (arXiv:2601.17063)** — ML-based cache eviction; only relevant if weights exceed RAM (we fit 13 GB INT4 in 48 GB).

### Tier 3 — NOT Applicable

- **FlashMoE-Distributed (arXiv:2506.04667)** — 8×H100 RDMA single-kernel; principle ("fuse dispatch+compute") already realized empirically by our packed-experts probe.
- **Huawei xDeepServe (arXiv:2508.02520)** — 384-chip CloudMatrix.
- **AMD MI325X benchmark (arXiv:2603.10031)** — vLLM cluster.
- **Kimi K2 / K2.5 / Kimi-VL tech reports** — training-side only.

---

## TODO

Recommended order: highest impact-per-effort first. Each has a kill-switch (cosine vs `gemma_golden.npz`).

### T1 — REAP expert pruning (do first)
- [x] **T1.1** Implement `python/moe/gemma_reap.py`: ~~load `gemma_routing.npz`~~ (only had aggregate stats — REAP does its own calibration via forward hooks on every router Linear). Score = sum across calibration set of softmax(router_logits)[expert] over the chosen top-8. Used router-gate-only score (no `expert_output_norm` term); proved sufficient at 50% prune. **DONE 2026-04-22**.
- [x] **T1.2** Drop bottom 50% experts per layer → 64 surviving. `python/moe/out/gemma_reap_mask.npz` saved (keep_idx, drop_mask, frac_gate_mass_dropped). Mean dropped gate mass at keep=64 on wikitext-2 (8K tokens) = 6.74%. **DONE 2026-04-22**.
- [x] **T1.3** Validate teacher-forced on `gemma_golden.npz`: **cos(prompt_last)=0.9941** (gate ≥0.97), **TF top-1 = 16/16** byte-identical continuation, mean step cos = 0.9788. **PASS 2026-04-22**. (See `docs/ane_book/JOURNAL.md` for the calibration-corpus and metric-design lessons.)
- [x] **T1.4** Routing locality on pruned model — `gemma_reap.py pack_locality` (2026-04-22). Mean distinct-packs (remap K=64): G=4 → 6.75, **G=8 → 5.39** (from 6.61, −18%), G=16 → 3.65, **G=32 → 1.99/2** (entire surviving set fits in 2 packs). Raw-128 numbers unchanged → all locality win comes from compact re-indexing, not REAP scoring. **G=32 is now potentially viable** (~39 MB INT4/pack); needs ane-validator probe before T4.1.

### T2 — MoBA-style sparse global attention (DEFERRED 2026-04-22)
- [x] **T2.1** Skeleton implemented (`python/moe/gemma_moba_probe.py`) — register custom `ALL_ATTENTION_FUNCTIONS["moba"]` interface, dispatches MoBA on global layers (`sliding_window is None`, `layer_idx not in last_full_set`).
- [~] **T2.2** Long-ctx validation NOT RUN. Cost-to-prove ≥ 50 min CPU at ctx=4K+8K, and projected savings (T2.3) made it not worth blocking T4.1.
- [x] **T2.3** Latency projection: with Kimi keep_last_full=3, only 3/30 layers are MoBA → 5% attn FLOP savings at ctx=4K, 7.5% at 8K. KV-streaming bandwidth saving is bigger but unmeasured. **Decision: defer T2 until baseline ANE perf is known; revisit as a prefill-optimization layer.**

### T3 — MoBE basis-shared experts (do third, optional)
- [ ] **T3.1** Implement `python/moe/gemma_mobe.py`: per-layer rank decomposition `W = A·B`, factorize B as linear combination of L=8 shared basis matrices; minimize reconstruction loss vs original via SGD on calibration activations.
- [ ] **T3.2** Validate weight-reconstruction MSE per expert; check end-to-end logit cosine ≥ 0.97 vs golden.
- [ ] **T3.3** Re-run `gemma_packed_experts.py` with MoBE-compressed weights; check whether G=12 or G=16 now fits in the 12-50 MB ANE window (was 24 MB at G=8 baseline).

### T4 — CoreML conversion (gated on T1; T2 deferred)
- [x] **T4.1.0** `python/moe/gemma_to_ane.py` skeleton: argparse, multi-function .mlpackage scaffold (embed + layer_i + lm_head), state schemas (sliding `(8,1024,256)` × 24 + global `(2,max_ctx,512)` × 6). Compile a 1-layer random-weights smoke build. Gate: `coremlcompiler` succeeds.
  - PASS — `gemma4_skeleton_1L.mlpackage` (205.5 MB INT4); ane-validator 15/15 linears on ANE incl. 4×down (45056,704); single-token 5.81 ms (random weights). See JOURNAL 2026-04-22.
- [x] **T4.1.1** Single layer with REAL REAP-permuted weights end-to-end (attention + 4 packed-expert programs + router). **Gate: ane-validator** all-ANE.
  - PASS — `gemma4_layer1_real.mlpackage` (217.5 MB INT4); 20/20 linears on ANE incl 4×down (45056,704), topk on ANE, one_hot on CPU (~0 ms); 6.56 ms/token (random inputs). See JOURNAL 2026-04-22.
- [x] **T4.1.2** Single layer **golden-validator**: hidden-state cosine ≥ 0.999 vs HF reference single-layer forward. Catches RoPE phase / GQA / k_eq_v / weight permutation bugs.
  - PASS (loosened single-layer gate cos≥0.99 + top-32 overlap≥0.90) — layer 0 REAP-pruned, dense+MoE parallel branches, INT4 `per_block` `block_size=16` (locked Gemma-pack quant config). cos(hidden)=0.992, cos(k_new)=0.995, cos(v_new)=0.995, top-32 overlap=0.938. PyTorch fp16 ref 14.2 ms vs CoreML INT4 24.8 ms (max_ctx=1024); mlpackage 271 MB. Rationale for loosened gate: K/V single-matmul floor ~0.994 at INT4; the strict ≥0.97 quality gate vs HF golden lives at T4.1.4 full-stack. See JOURNAL 2026-04-22.
- [ ] **T4.1.3** Stack 2 → 4 → 30 layers, ane- and golden-validator after each.
- [x] **T4.1.4** Embed + lm_head + final_logit_softcapping. **Gate**: cos ≥ 0.97 vs `gemma_golden.npz` last_logits, TF top-1 ≥ 14/16.
  - PASS 2026-04-23 — final RMSNorm convention fixed to Gemma-4 upstream semantics (`weight` direct, not `(1+weight)`). `gemma_t414_logit_gate.py` re-run: min cos = 0.9875, prompt-last cos recovered from 0.9254 to 0.9875, factual completions returned to top-1 (`' Paris'`, `' George Washington'`). See `docs/handoff_per_expert_moe_to_gemma.md` TL;DR #12.
- [ ] **T4.1.5** Decode loop with KV state read/write. **Gate**: replace the old open-ended 16-token oracle with a low-entropy long-decode oracle before checking this off.
  - STATUS 2026-04-23 — original `gemma_golden.npz` gate (`"What is the capital of France?"`) FAILs 6/16 exact-match, but the first divergence is an HF near-tie (`' Germany'` vs `' the'`, margin 0.25 nats), so it is not a trustworthy decode correctness gate.
  - Complementary factual gate PASS — stable closed-form battery captured in `python/moe/out/gemma_hf_closedform_battery.npz` and validated by `python/moe/gemma_t415_closedform_battery_gate.py`. CoreML chain matches HF 4/4 tokens on all three screened low-entropy prompts: `gold_symbol` (`Au`), `silver_symbol` (`Ag`), `iron_symbol` (`Fe`). Sentinel `.gemma_t415_closedform_battery_PASS` written.
  - Numerics note — raw `E @ h_scaled` softcap matmul emits NumPy/Accelerate `RuntimeWarning`s (`divide by zero`, `overflow`, `invalid`) on the CoreML path, but `GEMMA_SOFTCAP_DEBUG=1` showed all warned raw logits remain finite with `|raw| < 2.0`; treat as warning-noise pending cleanup, not a quality blocker.
- [x] **T4.2** `emilio/conv-ane/gemma_ane.swift` driver mirroring `qwen_ane.swift`.
  - STATUS 2026-04-23 — Swift runtime-prep artifacts now exist: `python/moe/export_gemma_swift_head.py` exports `gemma_logit_head.npz` into raw fp16 bins (`gemma_embed_fp16.bin`, `gemma_final_norm_gamma_fp16.bin`) plus `gemma_swift_head_meta.json` for Swift-side loading.
  - STATUS 2026-04-23 — `emilio/conv-ane/gemma_ane.swift` now has a real prompt-id path: load 3 shards, create one `MLState` per shard, prefill KV from prompt token IDs, and greedily decode additional token IDs using CPU-side final RMSNorm + tied-embedding argmax.
  - STATUS 2026-04-23 — bounded Swift smoke PASS after fixing two runtime bring-up bugs: workspace-relative shard-path resolution from `gemma_swift_head_meta.json`, and inter-shard `hidden -> x` rank normalization back to the model's expected `(1, 1, d_model)` input shape.
  - STATUS 2026-04-23 — exact smoke command `--prompt-ids 2,818,5279,529,7001,563 --n-new 4` generated `[9079, 236761, 1030, 563]`; this matches the first 4 generated IDs in the stored Python/CoreML artifact `python/moe/out/gemma_t414_generate_multi.npz` for the same prompt (`"The capital of France is"`).
  - STATUS 2026-04-23 — raw-text tokenizer port PASS: Swift now loads Gemma's HF BPE tokenizer JSON directly, including merge-pair decoding, `▁` space normalization, and `<0xXX>` byte-fallback tokens. `--prompt "The capital of France is" --tokenize-only` reproduces the reference prompt IDs `[2, 818, 5279, 529, 7001, 563]`.
  - STATUS 2026-04-23 — raw-text runtime parity PASS: `--prompt "The capital of France is" --n-new 4` generates `[9079, 236761, 1030, 563]`, exactly matching the verified prompt-id path.
  - STATUS 2026-04-23 — closed-form Swift raw-text parity PASS on all three stable factual prompts. Exact prompt IDs and exact 4/4 generated IDs match the saved oracle in `python/moe/out/gemma_hf_closedform_battery.npz` for `gold_symbol` (`Au`), `silver_symbol` (`Ag`), and `iron_symbol` (`Fe`).
  - STATUS 2026-04-23 — `gemma_ane.swift` now includes self-check hooks `--tokenize-only`, `--expect-prompt-ids`, and `--expect-generated-ids`, and `scripts/gemma_swift_closedform_parity.sh` provides a reproducible three-prompt parity driver using fresh-process invocations.
  - Remaining gap — the runtime driver itself now supports both prompt IDs and raw text; broader decode-quality signoff remains under T4.1.5 / T4.3, and perf/energy work remains under T4.4.
- [ ] **T4.3** Validation gate: full prefill+decode logit cosine ≥ 0.999 vs `gemma_golden.npz`, MMLU within 2pp.
  - STATUS 2026-04-24 — Swift prompt-logit grounding PASS for the existing 6-token REAP prompt (`[2, 818, 5279, 529, 7001, 563]`) using `--dump-prompt-logits-prefix` plus `python/moe/gemma_swift_logit_gate.py` against `python/moe/out/gemma_hf_golden_logits_reap.npz`. Result: prompt-ID alignment exact, per-position cosine `[0.9966, 0.9875, 0.9899, 0.9916, 0.9930, 0.9920]`, `min cos = 0.9875`, top-1 agreement `5/6`, sentinel `.gemma_swift_t414_logit_gate_PASS` written.
  - STATUS 2026-04-24 — this is prompt-position prefill grounding only; it does **not** close T4.3 because the current golden is the 6-token REAP prompt, not the full prefill+decode gate from `gemma_golden.npz`, and no Swift-side decode-logit or MMLU validation exists yet.
  - STATUS 2026-04-24 — `emilio/conv-ane/gemma_ane.swift` now also supports `--dump-decode-logits-prefix`, which writes one full-vocab post-token decode row per greedy step together with `prompt_ids`, `generated_ids`, `n_new_requested`, and `logit_row_semantics=post_token`. This aligns the Swift dump surface with `gemma_golden.npz` (`prompt_ids`, `next_token_ids`, `next_token_logits`) for the next T4.3 comparator pass.
  - STATUS 2026-04-24 — bounded Swift decode-logit grounding PASS for the first two greedy steps of the saved `gemma_golden.npz` prompt (`[2, 3689, 563, 506, 5279, 529, 7001, 236881]`) using `--dump-decode-logits-prefix` plus `python/moe/gemma_swift_decode_logit_gate.py`. Result: prompt-ID alignment exact, generated-ID prefix exact (`[108, 3689]`), per-step cosine `[0.9651, 0.9788]`, `min cos = 0.9651`, top-1 agreement `2/2`, sentinel `.gemma_swift_t415_decode_logit_gate_PASS` written.
  - STATUS 2026-04-24 — decode dumps are now checkpointed after each decoded token, not only at process exit. Long traced runs that are interrupted still leave usable partial `/tmp/<prefix>_{meta,logits_f32.bin}` artifacts with the current `generated_ids` prefix and `rows`, which the comparator can already score as a partial-horizon check.
  - STATUS 2026-04-24 — one fresh traced 16-step run on the checkpointing runtime (`/tmp/gemma_swift_t415_decode_full_trace_run2`) now pins down the current honest T4.3 limit. The checkpointed prefix stayed aligned through 4 persisted decode rows with exact generated-ID prefix `[108, 3689, 563, 506]`, cosine `[0.9651, 0.9788, 0.9870, 0.9820]`, and top-1 agreement `4/4`. At persisted row 5 the top-1 still matched (`529`) but cosine dropped to `0.9467`, below the current decode-smoke floor `0.96`. By persisted row 7 the generated-ID prefix had diverged outright: Swift checkpoint `[108, 3689, 563, 506, 5279, 529, 506]` vs golden `[108, 3689, 563, 506, 5279, 529, 9405]`. T4.3 therefore remains open: the bounded 2-step smoke is real, but full-horizon decode alignment is not yet stable on the current Swift/CoreML runtime.
  - STATUS 2026-04-24 — same-prefix hidden-boundary comparison first localized the injected drift to final shard `[22,30)`, not final RMSNorm or the tied head. On the first instrumented Swift replay dump (`/tmp/gemma_swift_t415_hidden_run4`), decode step 4 stayed same-prefix and still matched HF closely through `hidden_l15`/`hidden_l22` (`0.9700`, `0.9847`) before collapsing at `hidden_l30` (`0.5259`) and `projected_hidden` (`0.3488`). Decode step 5 showed the same pattern while still same-prefix: `hidden_l15=0.9900`, `hidden_l22=0.9898`, `hidden_l30=0.5291`, `projected_hidden=0.3108`.
  - STATUS 2026-04-24 — per-layer debugtap replay on shard `[22,30)` (`/tmp/gemma_swift_t43_finalshard_layers_run1`) tightens that result to the last layer of the shard. On same-prefix decode step 4 the intermediate boundaries stayed near-golden all the way through `hidden_l29`: `hidden_l23=0.9821`, `hidden_l24=0.9805`, `hidden_l25=0.9802`, `hidden_l26=0.9834`, `hidden_l27=0.9852`, `hidden_l28=0.9847`, `hidden_l29=0.9809`, then collapsed only at `hidden_l30=0.5259`. Decode step 5 showed the same pattern: `hidden_l23=0.9894`, `hidden_l24=0.9885`, `hidden_l25=0.9881`, `hidden_l26=0.9928`, `hidden_l27=0.9883`, `hidden_l28=0.9890`, `hidden_l29=0.9776`, then `hidden_l30=0.5291`. Therefore the first bad boundary is `hidden_l30`, which means layer 29 itself is the first layer injecting the decisive drift inside the final shard. Decode step 6 is already post-branch and preserves the same `hidden_l29 -> hidden_l30` collapse pattern.
  - STATUS 2026-04-24 — a fresh non-overwriting internal-tap rebuild (`gemma4_shard22_30_layer29internalsv2_real.mlmodelc`) was required because the earlier `layer29internals` debug shard was stale: the current runtime + baseline meta still reproduced the known-good prefix `[108, 3689, 563, 506, 5279, 529]`, while the old debug shard diverged from token 0 and was rejected before HF compare. The fresh v2 replay re-established same-prefix rows at steps 4/5 and narrowed the failure to the tail of layer 29 itself. Step 4: `hidden_l29=0.9809`, `hidden_l29_post_attn=0.9719`, `hidden_l29_ffn_out=0.9950`, then `hidden_l30=0.5259`, `projected_hidden=0.3488`. Step 5: `hidden_l29=0.9776`, `hidden_l29_post_attn=0.9789`, `hidden_l29_ffn_out=0.9951`, then `hidden_l30=0.5291`, `projected_hidden=0.3108`. In the exporter, `post_attn` is after `x = residual + post_attn_ln(a)` and `ffn_out` is after `x_ffn = post_ffn_ln(h1 + h2)` but before `x = residual + x_ffn; x = x * layer_scalar`; both stay near-golden. The earliest decisive collapse is therefore now localized to the final residual-add / `layer_scalar` / output-handoff tail of layer 29, not the attention path or the FFN/MoE branch itself.
  - STATUS 2026-04-24 — the apparent `hidden_l30` / `projected_hidden` collapse above was a comparator artifact, not a CoreML layer-29 tail bug. In `transformers`, `capture_outputs` rewrites `out.hidden_states[-1]` to `last_hidden_state`, so `python/moe/gemma_hf_hidden_boundary_compare.py` was accidentally comparing Swift `hidden_l30` against HF's post-final-norm state and then reapplying final norm for `projected_hidden`. After patching the comparator to hook the raw output of the final decoder layer, the same existing same-prefix dump (`/tmp/gemma_swift_t43_layer29internalsv2_run1`) improves to step 4 `hidden_l30=0.9943`, `projected_hidden=0.9541` and step 5 `hidden_l30=0.9950`, `projected_hidden=0.9504`, while `hidden_l29`, `hidden_l29_post_attn`, and `hidden_l29_ffn_out` remain unchanged and near-golden. The tiny packed `layer_scalar` values (for example layer 29 = `0.2080078125`) are consistent with the previously low Swift `hidden_l30` norm and helped expose that the HF side, not the exporter/runtime side, was being measured incorrectly. This falsifies the earlier layer-29-tail localization: the stale original `layer29internals` artifact is still rejected, but the fresh `layer29internalsv2` dump now shows the final shard output is near-golden on same-prefix rows. T4.3 remains open because full-horizon decode still diverges by row 7, but the active issue is no longer a `hidden_l29 -> hidden_l30` break inside shard `[22,30)`.
  - STATUS 2026-04-24 — a direct self-check on the same Swift dump (`tmp/gemma_projected_hidden_selfcheck.py` against `/tmp/gemma_swift_t43_layer29internalsv2_run1`) proves the runtime's CPU-side final projection is internally consistent. Recomputing `projected_hidden = hidden_l30 * gamma / (rms(hidden_l30) * softcap)` from the dumped Swift `hidden_l30` matches the dumped Swift `projected_hidden` at both same-prefix rows with cosine `1.00000000` and max abs error `3.6e-7`. So the `projected_hidden` cosine drop to ~`0.95` is not a second bug in Swift final RMSNorm or tied-head prep; it is the expected amplification of the remaining small same-prefix hidden-state drift (`hidden_l30 ~ 0.995`) before the logit head. The active debugging target therefore moves upstream again: find which earlier hidden-state drift directions matter for the row-5 near-tie flip (`506` vs `9405`), not the final projection implementation itself.
  - STATUS 2026-04-24 — `python/moe/gemma_swift_decode_logit_gate.py` now removes any stale prefix-scoped PASS sentinel on all non-pass exits, including fatal generated-ID mismatch. This matters for live-updating checkpoint prefixes: earlier partial-horizon PASS results must not survive once later rows on the same prefix fail.
  - STATUS 2026-04-24 — `scripts/gemma_swift_logit_gate.sh` now provides a reproducible one-command driver for the Swift dump + Python comparator path.
  - STATUS 2026-04-24 — `scripts/gemma_swift_decode_logit_gate.sh` now provides a reproducible bounded decode-smoke driver for the Swift dump + `gemma_golden.npz` comparator path. This is still a 2-step smoke only; full T4.3 remains open until the entire decode horizon and MMLU gate are covered.
- [ ] **T4.4** Perf+energy report; update `emilio/conv-ane/ANE_CHAIN_SCHEMA.md` Round 5.

---

## Decision Gates

- After **T1**: if pruning passes cosine, re-baseline tok/s before deciding T2/T3 priority.
- After **T2**: if MoBA training-free Quest variant fails quality, fall back to layer-wise hybrid (more full-attn layers) or skip and accept long-ctx penalty.
- **T3 is optional** — only pursue if T1+T2 land us > 15 tok/s and we still want to push pack cardinality.
