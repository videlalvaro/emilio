# AutoEML — Autonomous EML Graph Optimization

AutoEML is an autonomous optimization loop for EML (exp-minus-ln) evaluation
strategies, inspired by [AutoKernel](https://github.com/RightNow-AI/autokernel).
It optimizes the hot inner loops of EML-native matrix operations by iteratively
editing a single kernel file, benchmarking, and keeping or reverting changes.

## How It Works

### Architecture

```
autoeml_kernel.rs   ← THE file the agent edits (one kernel at a time)
autoeml_reference.rs ← Correctness oracles (DO NOT MODIFY)
autoeml_main.rs      ← CLI: profile / bench / verify (DO NOT MODIFY)
autoeml_program.md   ← Agent playbook (optimization tiers, constraints)
results.tsv          ← Experiment log
```

### The Loop

```
 ┌─────────────┐
 │  PROFILE    │  Analytical breakdown: which ops dominate?
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │  HYPOTHESIZE│  Pick an optimization from the playbook
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │  EDIT       │  Modify autoeml_kernel.rs
 └─────┬───────┘
       ▼
 ┌─────────────┐
 │  BENCH      │  5-stage correctness + throughput measurement
 └─────┬───────┘
       ▼
 ┌──────┴──────┐
 │  KEEP or    │  Faster & correct → git commit
 │  REVERT     │  Slower or wrong → git checkout
 └─────────────┘
       │
       └──→ repeat
```

### Constraints

- Every multiply **must** go through `exp(ln(a) + ln(b))` — no raw float ops.
- `c_exp()` and `c_ln()` are the **only** transcendental primitives.
- Results must match the reference implementation to within 1e-6 relative error.
- We don't cheat: no replacing EML with GPU/Metal ops.

### Benchmarking

The bench harness runs 5 stages before performance measurement:

1. **Smoke test** — 4×4 matmul
2. **Shape sweep** — 8×8 through model-size
3. **Numerical stability** — near-zero, large, negative, mixed-sign values
4. **Determinism** — same input → same output
5. **EML purity** — transcendental count matches expectation

Then it measures median latency over N iterations on a model-sized matmul
`(1, 896) × (896, 896)` — the QKV projection shape for Qwen2.5-0.5B.

## Profiling Results

Target model: **Qwen2.5-0.5B-Instruct** (single-token generation)

| Operation | % of Budget |
|-----------|-------------|
| FFN matmuls (gate+up+down) | 63.5% |
| LM head | 27.5% |
| QKV + output projections | 8.9% |
| SiLU + gate mul | ~0.1% |

Matmul dominates. All optimization effort targeted the CSE matmul kernel.

## Experiments

15 experiments were run. 7 were kept, 8 were reverted.

Then 3 more book-driven experiments (15–17) were run, 2 kept, 1 reverted.
**Total: 18 experiments, 9 kept, 9 reverted.**

### Kept (cumulative, in order applied)

| # | Optimization | Latency (μs) | Δ from baseline | Key insight |
|---|-------------|-------------|-----------------|-------------|
| 0 | Baseline CSE matmul | 17,238 | — | ln(A) + ln(B) shared across dot products |
| 2 | Transpose ln(B) | 16,494 | −4.3% | Cache-friendly sequential k-loop access |
| 4 | Activation sharing | — | −1,792 ln/QKV | Precompute ln(X) once for Q, K, V projections |
| 5 | Pre-transposed weights | 7,082 | −59.7% | One-time transpose at model load, not per call |
| 6 | Real-valued exp bypass | 4,264 | −75.3% | f64 exp + sign from im/π parity, skip Complex64 trig |
| 9 | 4-wide loop unroll | 4,065 | −76.4% | Independent accumulators for instruction-level parallelism |
| 10 | Batched atomic counter | 3,995 | −76.8% | One atomic add per matmul instead of per element |
| 11 | Branchless sign | 3,917 | −77.3% | `1.0 - 2.0*(n&1)` instead of if/else |
| 16 | Rayon j-parallelism | ~710 | −95.9% | Work-stealing over 896 columns [Iverson APL] |
| 17 | Zero-copy borrow + par_iter_mut | ~456 | −97.4% | Eliminate 12MB clone; write directly to result |

### Reverted

| # | Optimization | Latency (μs) | Why reverted |
|---|-------------|-------------|-------------|
| 1 | Rayon parallel outer loop | ~52,000 | 3× slower — thread pool overhead on M=1 |
| 3 | Two-phase batched eval | 16,913 | Slight regression — per-element Vec allocation |
| 7 | SoA re/im split | 5,485 | 28% slower — extra allocation + cache pressure from 4 arrays |
| 8 | Precomputed signs | 4,675 | 9.6% slower — sign extraction overhead > savings |
| 12 | Truncation vs round | ~3,921 | Noise — not worth the complexity |
| 13 | 8-wide loop unroll | 4,235 | Register pressure on Apple Silicon ARM |
| 14 | Pre-extracted re/sign arrays | 5,719 | 46% slower — 4-array cache pressure dominates |
| 15 | Compact f64+u8 format | ~3,834 | No improvement — compute-bound on exp(), not memory-bound |

## Final Results

Benchmark: `(1, 896) × (896, 896)` matmul with transposed precomputed weights.

| Metric | Baseline | After Exp 11 | After Exp 17 | Total Improvement |
|--------|----------|-------------|-------------|-------------------|
| **Latency** | 17,238 μs | 3,917 μs | **~456 μs** | **37.8× faster** |
| **Throughput** | 51,978 elem/s | ~225,000 elem/s | **~1,970,000 elem/s** | **37.9× higher** |
| **Transcendentals** | 1,606,528 | 803,712 | **803,712** | **50% reduction** |
| **ln calls** | 803,712 | 896 | **896** | **99.9% reduction** |

At ~456 μs we are at the theoretical hardware limit:
802,816 exp calls × ~4.7 ns / 8 cores ≈ 472 μs.

All 11 verification tests pass. Correctness confirmed across all shapes and
numerical edge cases.

## Key Learnings

1. **The biggest single win was avoiding Complex64 in the hot loop** (exp 6).
   Since ln(real) produces imaginary parts of exactly 0 or π, we replace
   `Complex64::exp()` (which computes cos+sin) with `f64::exp()` + a sign bit.
   This alone was ~40% faster.

2. **Memory layout beats algorithmic cleverness.** Transposing weight matrices
   at load time (exp 5) gave a larger speedup than any loop-level trick.

3. **SoA hurts when both components are accessed together.** Splitting Complex64
   into separate re/im arrays (exp 7) increased cache pressure without benefit —
   the AoS layout is correct when re and im are consumed as a pair.

4. **ARM register pressure limits unrolling.** 4-wide is the sweet spot on
   Apple Silicon; 8-wide causes register spills and is 8% slower.

5. **Diminishing returns at the loop level.** After exp 6, each remaining
   single-threaded improvement was single-digit percent.

6. **Parallelism unlocked the next frontier** (exp 16–17). Rayon work-stealing
   over the j dimension gave 5.3× on 8 cores. Eliminating the 12MB clone
   (zero-copy borrow of precomputed weights) added another 1.5×. Combined:
   ~8× over the single-threaded optimized kernel. Source: Iverson's APL model
   of treating inner products as atomic parallel operations.

7. **We hit the hardware wall.** At ~456 μs, latency matches the theoretical
   minimum of 802,816 exp calls × 4.7 ns / 8 cores ≈ 472 μs. Further gains
   require reducing exp call count (algebraic pruning of negligible terms) or
   using approximate exp.

## Commit History

```
e663c44 emilio: precomputed ln(weights) + optimized matmul — 0.49 → 4.3 tok/s (8.8×)
02b0dca autoeml exp 15-17: rayon j-parallelism + zero-copy — 456 μs (37.8× faster)
34dd3e7 autoeml exp 11: branchless sign — ~3,917 μs (77.7% faster than baseline)
31072d7 autoeml exp 10: batched atomic counter — ~3,995 μs (77.3% faster than baseline)
cc618f6 autoeml exp 9: 4-wide loop unroll — 4,065 μs (76.9% faster than baseline)
1d7a4e3 autoeml exp 6: real-valued exp bypass — 4,264 μs (75.8% faster than baseline)
8b49923 autoeml exp 5: pre-transposed weight precompute — 7,082 μs (59.7% faster than baseline)
a494232 autoeml exp 4: activation sharing — precompute ln(X) for Q/K/V reuse
a48ff74 autoeml exp 2: transpose ln(B) for cache-friendly k-loop
97e9a58 autoeml: autonomous EML graph optimization agent
```

## Emilio Integration: Kernel → Inference Engine

The autoeml micro-benchmark optimizes a single `(1, 896) × (896, 896)` matmul
in isolation. Emilio is the full Qwen2.5-0.5B inference engine that executes
24 transformer layers per token, each containing 7 matmuls plus RMSNorm, SiLU,
RoPE, and softmax — all through pure EML arithmetic.

### The Gap

Emilio originally used `build_matmul_cse()` from `eml_optimizer.rs`, which:
- Used full `Complex64::exp()` in the inner loop (cos + sin computation)
- Recomputed `ln(B)` for every matmul call (weights never change)
- Transposed `ln(B)` on every call (redundant work)
- Used `Complex64::ln()` element-wise (2× slower than f64 path)

This gave **0.49 tok/s** — each token required ~168 matmuls (7 per layer × 24
layers) plus the LM head (the largest matmul: 896 × 151,936).

### What Changed

**Step 1: Optimized `build_matmul_cse`** — Rewrote the function body to use
all autoeml kernel optimizations without atomic counters (which cause massive
contention under rayon):
- Real-exp bypass: `f64::exp(re)` + branchless sign from `im/π` parity
- 4-wide loop unroll with independent accumulators
- Rayon `par_iter_mut` over the j dimension (column parallelism)
- Parallel `ln(A)`, `ln(B)`, and transpose

**Step 2: Precomputed `ln(weights)` at load time** — Added
`build_matmul_cse_precomp()` that accepts pre-computed `ln(B)` in transposed
layout, skipping both `ln()` and transpose per call. Key insight: GGUF stores
weights as `(out_dim, in_dim)`, and emilio transposes them to `(in_dim, out_dim)`
before matmul, which internally transposes `ln(B)` to `(out_dim, in_dim)`.
The double-transpose cancels — so element-wise `ln()` of the original GGUF
layout gives the correct `ln_b_t` directly.

**Step 3: Wired all 13 matmul call sites** — Every matmul in emilio
(QKV projections, output projection, gate/up/down FFN, and LM head) now uses
`build_matmul_cse_precomp` with layer-specific precomputed `ln(W)` fields.

### Results

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| **Token generation** | 0.49 tok/s | **~4.3 tok/s** | **8.8×** |
| **Per-token time** | ~2.05 s | ~0.23 s | 8.8× |
| **Model load time** | 1.24 s | 2.69 s | +1.45 s (one-time) |
| **Output quality** | ✓ correct | ✓ identical | — |

The load-time increase (~1.45s) pays for precomputing `ln()` over ~494M weight
elements (168 weight matrices × avg ~2.9M elements each). This is amortized
over all generated tokens — break-even at ~1 token.

### Where the Time Goes (per token, ~230 ms)

At ~4.3 tok/s, each token takes ~230 ms across 24 layers:
- **FFN matmuls** (gate + up + down): ~63% — 3 matmuls per layer, the gate/up
  projections are (1, 896) × (896, 4864) = ~4.4M exp each
- **LM head**: ~28% — (1, 896) × (896, 151936) = ~136M exp, runs once per token
- **QKV + output projections**: ~9% — 4 matmuls per layer but smaller
- **RMSNorm, SiLU, RoPE, softmax**: <1% — dominated by matmul cost

## Running AutoEML

```bash
cd eml_rust

# Profile transcendental budget
cargo run --bin autoeml --release -- profile

# Benchmark current kernel
cargo run --bin autoeml --release -- bench
cargo run --bin autoeml --release -- bench --precomputed
cargo run --bin autoeml --release -- bench --transposed

# Verify all operations
cargo run --bin autoeml --release -- verify

# Run emilio inference (Qwen2.5-0.5B)
cargo run --bin emilio --release -- ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --generate "The capital of France is"
```
