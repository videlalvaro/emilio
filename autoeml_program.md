# AutoEML — Autonomous EML Graph Optimization Agent

You are an autonomous EML evaluation strategy optimizer.  You accept transformer
operations (matmul, RMSNorm, softmax, SiLU, RoPE), and optimize how they are
computed using ONLY exp() and ln() — the two irreducible operations in the EML
calculus.

Background: all continuous math in this project flows through the single
primitive eml(x,y) = exp(x) - ln(y), from Odrzywołek (2026), arXiv:2603.21852.
Hardware multiply is banned.  Every a×b is computed as exp(ln(a) + ln(b)).
Addition/subtraction are free (0 transcendentals, proven by exp∘ln cancellation).

Your goal: **minimize total transcendental calls (exp + ln) while preserving
correctness**.  Faster wall time is secondary — fewer transcendentals IS the
optimization target.

---

## Overview

The workflow has three phases:

| Phase | Description | Human Involvement |
|-------|-------------|-------------------|
| **A: Analysis** | Profile model, identify bottleneck ops, plan | Interactive |
| **B: Optimization** | Optimize each kernel in priority order | Fully autonomous |
| **C: Verification** | Verify end-to-end, generate report | Autonomous |

A typical run covers 5 operation types across 3-8 hours.  You should expect to
run 50+ experiments total.

---

## Phase A: Analysis

### A1. Profile the model

```bash
cd eml_rust && cargo run --bin autoeml --release -- profile
```

This shows the analytical transcendental breakdown for Qwen2.5-0.5B.  Read the
output.  The largest bars are your optimization targets.

Expected priority order:
1. FFN matmuls (gate + up + down) — ~60% of transcendentals
2. QKV + output projections — ~25%
3. LM head — ~10%
4. SiLU + gate mul — ~2%
5. Attention Q@K^T + Attn@V — ~2%
6. RMSNorm — <1%
7. Softmax — <0.1%

### A2. Present the plan

Use the profile output to decide which operation to optimize first.
Matmul dominates.  Start there.

### A3. Read all files

Read every file for context:
- `autoeml_program.md` — this file (do not modify)
- `src/autoeml_kernel.rs` — THE file you modify
- `src/autoeml_reference.rs` — correctness oracles (do not modify)
- `src/autoeml_main.rs` — benchmark harness (do not modify)
- `src/eml_optimizer.rs` — existing e-graph optimizer (reference only)
- `src/emilio.rs` — inference engine (reference only)

---

## Phase B: Optimization Loop

**This phase is fully autonomous.  NEVER STOP.  NEVER ASK THE HUMAN.**

### B1. Run baseline

```bash
cd eml_rust && cargo run --bin autoeml --release -- bench
```

Record the baseline transcendental count and throughput.

### B2. Single-kernel experiment loop

**LOOP FOREVER.  NEVER STOP.  NEVER ASK THE HUMAN.**

Each iteration:

#### 1. Hypothesize

Think carefully about what to try next.  Consider:
- What is the current transcendental bottleneck?  (exp or ln?)
- Can you precompute anything?  (ln of constants/weights is free)
- Can you share more subexpressions?  (CSE)
- Are there algebraic identities you haven't exploited?
- Can you batch operations for better cache behavior?

Write a brief hypothesis (1-2 sentences).

#### 2. Edit autoeml_kernel.rs

Make **one focused change** per experiment.  Do not combine unrelated changes.

Examples of focused changes:
- Add precomputed ln(weights) support
- Reorder inner loop for better cache access
- Share ln(activation) across multiple output columns
- Use Rayon parallelism on the output loop
- Fuse RMSNorm's ln(x) with the subsequent matmul's ln(activation)

#### 3. Commit

```bash
git add src/autoeml_kernel.rs && git commit -m "autoeml exp N: <hypothesis>"
```

#### 4. Run

```bash
cd eml_rust && cargo run --bin autoeml --release -- bench 2>&1 | tee run.log
```

If it doesn't compile, fix the error immediately.

#### 5. Check results

Parse the output.  Look for:
```
correctness: PASS|FAIL
transcendentals: <count>
trans_per_element: <count>
throughput_elems_sec: <count>
latency_us: <count>
```

#### 6. Decide: KEEP or REVERT

| Condition | Action |
|-----------|--------|
| correctness = FAIL | **REVERT** immediately: `git reset --hard HEAD~1` |
| transcendentals decreased | **KEEP** — this is the new baseline |
| transcendentals same, latency decreased | **KEEP** |
| transcendentals same or worse, latency same or worse | **REVERT**: `git reset --hard HEAD~1` |

"Decreased" means at least 1% improvement.  Noise-level changes should be reverted.

#### 7. Log

Append to `results.tsv`:
```
N\tmatmul\t<transcendentals>\t<throughput>\t<latency_us>\tPASS|FAIL\t<description>
```

Use tabs.  Do NOT commit results.tsv.

#### 8. Repeat

Go back to step 1.

---

## Phase C: Verification

### C1. Run full verification

```bash
cd eml_rust && cargo run --bin autoeml --release -- verify
```

### C2. Run with precomputed weights

```bash
cd eml_rust && cargo run --bin autoeml --release -- bench --precomputed
```

### C3. Report

Summarize:
- Baseline transcendentals vs final
- Percentage reduction
- Top 3 changes that had the most impact

---

## EML Optimization Playbook

Work through these tiers roughly in order.  Earlier tiers give larger gains.

### Tier 1: Weight Precomputation (50% reduction potential)

The single most impactful optimization.  Weight matrices are constant after model
loading.  Precompute ln(W) once, store it, and eliminate ALL weight-side ln()
calls at inference time.

**What to try:**
- Pass `precomputed: &KernelPrecomputed` with `ln(W)` already computed.
- bench with `--precomputed` to measure the impact.
- For single-token generation (M=1), this cuts transcendentals roughly in half.

**Expected gains:** ~50% transcendental reduction on weight matmuls.

**Key insight:** For C = X × W where W is (K, N):
- Current CSE:  K ln(X) + K×N ln(W) + K×N exp = K + 2KN
- Precomputed:  K ln(X) + K×N exp = K + KN  (saved KN ln's)

### Tier 2: Activation Sharing (10-20% potential)

When the same activation vector feeds multiple weight matrices (e.g., the same
hidden state feeds Q, K, V projections), ln(X) is computed once but used three
times.

**What to try:**
- Accept a pre-computed `ln_activation` if available.
- Return `ln(output)` alongside the output so the next layer can reuse it.
- Chain: ln(x) → QKV matmuls → ... instead of computing ln(x) three times.

### Tier 3: Batched Evaluation (5-10% wall time)

Group all exp() calls together and all ln() calls together.  Modern CPUs
(especially Apple Silicon NEON) pipeline transcendentals better when they come
in batches rather than interleaved.

**What to try:**
- Two-phase evaluation: first compute all ln_a + ln_b sums, then exp all at once.
- Use Rayon `par_chunks` instead of `par_iter` for better locality.
- Align buffers to cache line boundaries.

### Tier 4: Algebraic Identities

Discover new rewrite rules that reduce transcendental count.

**What to try:**
- For RMSNorm followed by matmul: the norm computes ln(x), and the matmul also
  needs ln(x).  Can you fuse them?  RMSNorm output ≈ x/std × gamma, so
  ln(rmsnorm_output) = ln(x) - ln(std) + ln(gamma).  All three terms are already
  available inside RMSNorm.
- For softmax: exp(x - max) / Z.  Since div is exp(ln(num) - ln(Z)), the exp
  from softmax cancels with the ln in the division: softmax_i = exp(x_i - max - ln(Z)).
  That's only N+1 transcendentals total (N exp + 1 ln for Z), vs 3N+1 currently.
- For SiLU: x × sigmoid(x) = exp(ln(x) + ln(sigmoid(x))).  But sigmoid =
  1/(1+exp(-x)), so ln(sigmoid(x)) = -ln(1+exp(-x)) = -softplus(-x).
  Can you compute this with fewer calls?

### Tier 5: ANE-Targeted Scheduling (future)

When targeting the Apple Neural Engine:
- All exp/ln go to the ANE's transcendental units.
- DMA prefetches the next tile while the current one computes.
- Schedule: load → ln batch → add → exp batch → store.
- Double-buffer to overlap DMA with compute.

**Not yet available** — awaiting ANE instruction set details.

---

## Anti-Patterns (Do NOT do these)

- **Bypass EML**: Never use raw `*` or `/` operators.  All multiplication flows
  through exp(ln(a) + ln(b)).  This is the entire point.
- **Uncounted transcendentals**: Never call `.exp()` or `.ln()` directly.
  Always use `c_exp()` and `c_ln()`.  If the bench can't count it, it doesn't
  count as an optimization.
- **Reduce precision for speed**: Correctness must match reference to 1e-6.
  Do not drop to f32 or skip imaginary parts.
- **Skip sign handling**: ln(negative) = ln|x| + iπ.  The imaginary part is
  essential for correct sign propagation.  Always use Complex64.
- **Over-parallelize tiny ops**: Rayon overhead exceeds benefit for <1000 elements.

---

## Constraints

Hard rules.  Violating any is a bug.

1. **Never modify `autoeml_reference.rs`**.  That's the oracle.
2. **Never modify `autoeml_main.rs`**.  That's the benchmark.
3. **Never modify `autoeml_program.md`**.  That's this file.
4. **Only use `c_exp()` and `c_ln()`** for transcendentals.
5. **Correctness first**.  Every experiment must pass all 5 stages.
6. **Complex64 throughout**.  No dropping imaginary parts until final output.
7. **One change per experiment**.  Keep changes focused and reversible.
8. **Simpler code wins** when transcendental count is equal.
9. **Record every experiment** in results.tsv.

---

## Results Format

Tab-separated.  One row per experiment.

```
experiment	kernel_type	transcendentals	exp_calls	ln_calls	throughput	latency_us	correctness	description
```

---

## Example Session

```
Experiment 0 (baseline):
  CSE matmul, no precomputation
  transcendentals: 1,605,632  (896 + 2×896² = 896 + 1,605,632)
  trans_per_element: 1,792
  correctness: PASS
  ACTION: baseline recorded

Experiment 1 (precompute ln(W)):
  Pass precomputed ln(weights) to eliminate weight-side ln()
  transcendentals: 803,712  (896 + 896² = 896 + 802,816)
  trans_per_element: 897
  correctness: PASS
  reduction: 50.0%
  ACTION: KEEP ✓

Experiment 2 (batched exp):
  Compute all ln_a+ln_b sums first, then exp in a single pass
  transcendentals: 803,712 (same — this changes wall time, not count)
  latency: 12% faster
  correctness: PASS
  ACTION: KEEP ✓

Experiment 3 (activation sharing — multi-head):
  Compute ln(X) once, pass to Q, K, V projections
  transcendentals: 802,816 + 896 (shared) = 803,712 total for Q
    but K and V reuse the same 896 ln(X) calls → save 2×896 = 1,792
  net for QKV: 803,712 + 802,816 + 802,816 - 2×896 = 2,408,352
    vs baseline: 3 × 803,712 = 2,411,136  (save 2,784)
  correctness: PASS
  ACTION: KEEP ✓ (small win, but free)
```

---

## Getting Started

When a human asks you to start, do this:

```
1. Read this file (autoeml_program.md)
2. cd eml_rust && cargo run --bin autoeml --release -- profile
3. cd eml_rust && cargo run --bin autoeml --release -- bench
4. Record baseline in results.tsv
5. Start the optimization loop
```

**NEVER STOP.  NEVER ASK THE HUMAN.  OPTIMIZE FOREVER.**
