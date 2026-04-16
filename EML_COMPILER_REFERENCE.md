# EML Compiler Reference

**Version**: 2.0 (v1 + v2 formats)  
**Implementation**: emilio (Rust)  
**Date**: April 2026

---

## 1. Overview

EML is a compiled model format and inference system built on a single algebraic primitive:

$$\text{eml}(x, y) = e^x - \ln(y)$$

From this one binary operator and the constant 1, every elementary function—exponentiation, logarithm, arithmetic, trigonometric, and hyperbolic functions—can be derived (Odrzywołek, 2026). The EML compiler takes a GGUF-format quantized model, precomputes weight transformations, and emits a `.eml` binary that the emilio inference engine executes entirely through the EML primitive.

### Design Principle

> Speed is not the goal. The goal is proving that a single algebraic identity suffices for every operation in a production transformer.

---

## 2. The Primitive

```
eml(x, y) = exp(x) - ln(y)
```

Implemented in Rust (`eml_ops.rs`):

```rust
pub fn eml(x: Complex64, y: Complex64) -> Complex64 {
    x.exp() - y.ln()
}
```

All computation operates in `Complex64` (complex128). The imaginary part of `ln` encodes sign information: `ln(-x) = ln|x| + iπ`.

---

## 3. Derived Operations

Every operation is derived from `eml` via a bootstrapping chain:

```
eml → exp → ln → sub → 0 → neg → add → inv → mul → div → pow → sqrt
```

### 3.1 Scalar Operations

| Operation | Definition | Algebraic Form | Transcendental Cost |
|-----------|-----------|----------------|---------------------|
| `exp(x)` | `eml(x, 1)` | `exp(x)` | 1 exp |
| `ln(z)` | `eml(1, eml(eml(1, z), 1))` | `ln(z)` | 1 ln |
| `sub(a, b)` | `eml(ln(a), exp(b))` | `a - b` | 0 (exp∘ln cancel) |
| `neg(x)` | `sub(0, x)` | `-x` | 0 |
| `add(a, b)` | `sub(a, neg(b))` | `a + b` | 0 |
| `inv(z)` | `exp(-ln(z))` | `1/z` | 0 (hardware division) |
| `mul(a, b)` | `exp(ln(a) + ln(b))` | `exp(ln(a) + ln(b))` | 2 ln + 1 exp |
| `div(a, b)` | `exp(ln(a) - ln(b))` | `exp(ln(a) - ln(b))` | 2 ln + 1 exp |
| `pow(a, b)` | `exp(b * ln(a))` | `exp(b·ln(a))` | 1 ln + 1 exp + 1 mul |
| `sqrt(x)` | `exp(0.5 * ln(x))` | `exp(½·ln(x))` | 1 ln + 1 exp |

**Key insight**: Addition and subtraction are *free* (zero transcendentals) because `exp` and `ln` cancel in the EML derivation. Multiplication costs 3 transcendentals—this is the fundamental cost unit.

### 3.2 Fused vs. Unfused

The compiler provides two implementations of each operation:

- **Unfused** (`eml_*_unfused`): Tree-walking evaluation matching the Python reference. Used for correctness verification.
- **Fused** (`eml_*`): Algebraically simplified. Fewer transcendental calls, same numerical result.

Example—multiplication:

```rust
// Unfused (20 eml() calls):
pub fn eml_mul_unfused(a: Complex64, b: Complex64) -> Complex64 {
    eml_exp_unfused(eml_add_unfused(eml_ln_unfused(a), eml_ln_unfused(b)))
}

// Fused (3 transcendentals: 2 ln + 1 exp):
pub fn eml_mul(a: Complex64, b: Complex64) -> Complex64 {
    (a.ln() + b.ln()).exp()
}
```

### 3.3 Composite Operations

| Operation | Formula | Cost |
|-----------|---------|------|
| `softmax(x)` | `exp(x - max) / Σ exp(xᵢ - max)` | 3K + O(1) per K elements |
| `RMSNorm(x, γ)` | `x / √(mean(x²) + ε) · γ` | Shared `ln(x)` via CSE |
| `SiLU(x)` | `x · σ(x)` where `σ(x) = 1/(1+exp(-x))` | 3 transcendentals |
| `GELU(x)` | `x · σ(1.702x)` | 5 transcendentals |
| `matmul(A, B)` | `Σₖ exp(ln|aₖ| + ln|bₖⱼ|) · sign(aₖ) · sign(bₖⱼ)` | K exp + K ln per element |

All composite operations are parallelized via Rayon.

---

## 4. Compilation Pipeline

```
┌───────────────────────────┐
│  GGUF (Q8_0 quantized)    │  Input: quantized model weights
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│  Load & Dequantize        │  Q8_0 blocks → f64
│  (gguf.rs)                │  Parse metadata, extract tensors
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│  Precompute ln(W)         │  For each weight w:
│  (engine.rs)              │    ln(w) = ln|w| + iπ·(w < 0)
│                           │  Stored as Complex64
└────────────┬──────────────┘
             │
             ├─────────────────────┐
             ▼                     ▼
┌────────────────────┐  ┌──────────────────────┐
│  EML v1 Format     │  │  EML v2 Format       │
│  (eml_format.rs)   │  │  (eml_v2.rs)         │
│                    │  │                      │
│  Complex64 weights │  │  Sign+magnitude      │
│  ~8.4 GB           │  │  Weight fusion       │
│                    │  │  Sparse pruning      │
│                    │  │  Execution graph     │
│                    │  │  ~4.8 GB             │
└────────────────────┘  └──────────────────────┘
             │                     │
             └──────────┬──────────┘
                        ▼
           ┌────────────────────────┐
           │  Inference Engine      │
           │  (engine.rs)           │
           │                       │
           │  Tokenize → Forward   │
           │  → Sample → Decode    │
           └────────────────────────┘
```

### 4.1 GGUF Loading (gguf.rs)

The GGUF parser supports:

- **Format**: GGUF v3, little-endian
- **Tensor types**: F32, F16, Q8_0, Q4_0, BF16
- **Dequantization**: Q8_0 blocks (32 values × int8 + f16 scale) → f64

```
Q8_0 block layout:
  [f16 scale] [int8 × 32]
  
  dequant: value[i] = scale × quant[i]
```

Metadata extraction provides model architecture parameters (`QwenConfig`):
- `vocab_size`, `n_layers`, `n_heads`, `n_kv_heads`
- `d_model`, `d_ff`, `d_head`
- `rope_freq_base`, `rms_norm_eps`
- `eos_token_id`, `eot_token_id`

### 4.2 Weight Precomputation (engine.rs)

The core compile-time transformation:

```
For each weight matrix W (shape: out_dim × in_dim):
    For each element w:
        ln(w) = Complex64 {
            re: ln(|w|),
            im: if w > 0 { 0.0 } else { π }
        }
```

This eliminates all `ln(W)` calls at inference time. The logarithmic homomorphism `ln: (ℝ₊, ×) → (ℝ, +)` is factored out of the inner loop—computed once at compile time, amortized over all generated tokens.

**Weight matrices precomputed per layer** (7 total):
- `ln_q` — Query projection
- `ln_k` — Key projection
- `ln_v` — Value projection
- `ln_o` — Output projection
- `ln_gate` — SwiGLU gate projection
- `ln_up` — SwiGLU up projection
- `ln_down` — SwiGLU down projection

Plus `ln_output` for the LM head.

### 4.3 Sign Encoding

Sign is encoded in the imaginary part of `Complex64`:

```
ln(w) for w > 0:  re = ln(w),   im = 0
ln(w) for w < 0:  re = ln(|w|), im = π
```

Sign recovery in the inner loop:
```rust
sign(w) = 1.0 - 2.0 * round(im / π)
```

Since `im` is exactly 0 or π, the round operation is exact.

---

## 5. Binary Formats

### 5.1 EML v1 Format

**Magic**: `EML1` (4 bytes)  
**Byte order**: Little-endian throughout

```
┌──────────────────────────────────────┐
│ Header                               │
│   magic: [u8; 4] = "EML1"           │
│   version: u32 = 1                   │
│   vocab_size: u32                    │
│   n_layers: u32                      │
│   n_heads: u32                       │
│   n_kv_heads: u32                    │
│   d_model: u32                       │
│   d_ff: u32                          │
│   rope_freq_base: f64                │
│   rms_norm_eps: f64                  │
│   max_seq_len: u32                   │
│   d_head: u32                        │
│   eos_token_id: u32                  │
│   eot_token_id: u32                  │
├──────────────────────────────────────┤
│ Tokenizer                            │
│   vocab_count: u32                   │
│   for each token:                    │
│     len: u32                         │
│     bytes: [u8; len]                 │
│   merges_count: u32                  │
│   for each merge:                    │
│     len: u32                         │
│     bytes: [u8; len]  ("left right") │
├──────────────────────────────────────┤
│ Global Weights                       │
│   token_embd: f64[vocab × d_model]   │
│   output_norm: f64[d_model]          │
│   ln_output: Complex64[vocab × d_model] │
├──────────────────────────────────────┤
│ Per-Layer Weights (× n_layers)       │
│   ln_q: Complex64[n_heads*d_head × d_model]     │
│   ln_k: Complex64[n_kv_heads*d_head × d_model]  │
│   ln_v: Complex64[n_kv_heads*d_head × d_model]  │
│   ln_o: Complex64[d_model × n_heads*d_head]      │
│   ln_gate: Complex64[d_ff × d_model]             │
│   ln_up: Complex64[d_ff × d_model]               │
│   ln_down: Complex64[d_model × d_ff]             │
│   q_bias: f64[n_heads*d_head]                    │
│   k_bias: f64[n_kv_heads*d_head]                 │
│   v_bias: f64[n_kv_heads*d_head]                 │
│   attn_norm: f64[d_model]                        │
│   ffn_norm: f64[d_model]                         │
└──────────────────────────────────────┘
```

**Array encoding**: Every array is length-prefixed with a `u64` element count.

**Complex64 encoding**: Two consecutive `f64` values (real, imaginary) per element = 16 bytes/element.

**File size for Qwen2.5-0.5B**: ~8.4 GB

### 5.2 EML v2 Format

**Magic**: `EML2` (4 bytes)

v2 applies five optimizations over v1:

#### 5.2.1 Sign+Magnitude Encoding

Replaces Complex64 (16 bytes) with:
- **Magnitude**: `f64` — `ln|w|` (8 bytes)
- **Sign**: 1 bit per element, packed 8 per byte

```
Disk layout per tensor:
  magnitudes: f64[N]                    // 8N bytes
  sign_bitmap: u8[ceil(N/8)]            // N/8 bytes

  Bit k of byte k/8:
    0 → positive (sign = +1.0)
    1 → negative (sign = -1.0)
```

**Savings**: 8.125 bytes/element vs 16 bytes — **49% reduction**.

At load time, signs are expanded to `f64` (+1.0/-1.0) for branch-free inner loop execution.

#### 5.2.2 QKV Weight Fusion

Q, K, V projection weights concatenated column-wise into a single `W_qkv`:

```
W_qkv = [W_q | W_k | W_v]    shape: (q_dim + kv_dim + kv_dim) × d_model
```

One matmul replaces three. The `ln(activation)` computation is shared instead of repeated 3×. Saves ~43K ln computations per token across 24 layers.

#### 5.2.3 Gate+Up Fusion

SwiGLU gate and up projection weights concatenated:

```
W_gate_up = [W_gate | W_up]   shape: (2 × d_ff) × d_model
```

One matmul replaces two. Saves ~21.5K ln computations per token.

#### 5.2.4 Sparse Pruning

Weights with `ln|w| < -30.0` (i.e., `|w| < 9.4 × 10⁻¹⁴`) are replaced with `-∞`.

```rust
// IEEE 754: exp(-∞) = 0
// No runtime branching needed — pruned entries contribute nothing.
```

Measured sparsity: **0.8%** at this conservative threshold (Qwen2.5 weights are well-conditioned).

#### 5.2.5 Execution Graph

The inference DAG is serialized as a sequence of tagged operations:

```rust
pub enum ExecOp {
    RmsNorm { layer: i32 },          // -1 = final norm
    FusedQkvMatmul { layer: u32 },
    BiasAdd { layer: u32 },
    RoPE { layer: u32 },
    Attention { layer: u32 },
    OutputProjection { layer: u32 },
    ResidualAdd { layer: u32, stage: u32 },
    FusedGateUpMatmul { layer: u32 },
    SiLU { layer: u32 },
    ElementwiseMul { layer: u32 },
    DownProjection { layer: u32 },
    LmHead,
}
```

314 operations for Qwen2.5-0.5B. Stored for documentation, reproducibility, and future graph-level optimization passes.

**File size for Qwen2.5-0.5B**: ~4.8 GB (43% smaller than v1)

---

## 6. E-Graph Optimizer (eml_optimizer.rs)

The compiler includes an equality saturation optimizer using the `egg` crate.

### 6.1 Term Language

```
S → Const(f64) | Var(str)
S → Eml(S, S)
S → Exp(S) | Ln(S)
S → Add(S, S) | Sub(S, S) | Neg(S)
S → Mul(S, S) | Div(S, S) | Inv(S)
S → Sqrt(S) | Pow(S, S)
S → Gelu(S) | Sigmoid(S)
```

### 6.2 Cost Model

The cost function counts transcendental evaluations (exp + ln):

| Node | Cost |
|------|------|
| `Const`, `Var` | 0 |
| `Eml(x, y)` | 2 + cost(x) + cost(y) |
| `Exp(x)`, `Ln(x)` | 1 + cost(x) |
| `Add`, `Sub`, `Neg` | 0 + children (proven free) |
| `Mul`, `Div` | 3 + children |
| `Inv` | 2 + children |
| `Sqrt` | 2 + children |
| `Pow` | 2 + children |
| `Gelu` | 5 + children |
| `Sigmoid` | 2 + children |

### 6.3 Rewrite Rules

Paper-derived equivalences that the e-graph explores:

**EML ↔ exp/ln**:
```
eml(x, 1)     ⟺  exp(x)
exp(ln(x))    →  x
ln(exp(x))    →  x
```

**Additive group**:
```
add(a, b)     ⟺  add(b, a)
add(a, 0)     →  a
sub(a, a)     →  0
neg(neg(x))   →  x
sub(a, b)     ⟺  add(a, neg(b))
```

**Multiplicative group**:
```
mul(a, b)     ⟺  mul(b, a)
mul(a, 1)     →  a
div(a, a)     →  1
inv(inv(x))   →  x
div(a, b)     ⟺  mul(a, inv(b))
```

**Log-domain equivalences** (core EML insight):
```
mul(a, b)     ⟺  exp(add(ln(a), ln(b)))
div(a, b)     ⟺  exp(sub(ln(a), ln(b)))
inv(x)        ⟺  exp(neg(ln(x)))
sqrt(x)       ⟺  exp(mul(0.5, ln(x)))
```

**Log distribution rules** (CSE enablers):
```
ln(mul(a, b)) ⟺  add(ln(a), ln(b))
ln(div(a, b)) →  sub(ln(a), ln(b))
ln(pow(a, b)) →  mul(b, ln(a))
exp(add(a,b)) ⟺  mul(exp(a), exp(b))
```

### 6.4 Saturation Parameters

```rust
Runner::default()
    .with_iter_limit(30)
    .with_node_limit(50_000)
    .with_time_limit(Duration::from_secs(5))
```

---

## 7. Inference Engine (engine.rs)

### 7.1 Forward Pass Architecture

Target: Pre-norm transformer with RoPE, GQA, SwiGLU (Qwen2.5).

```
Token ID
    │
    ▼
[Embedding Lookup]        discrete indexing (not EML)
    │
    ▼
For each layer (×24):
    ├─ RMSNorm (attention)    CSE: shared ln(x)
    ├─ QKV Projection         matmul via precomputed ln(W)
    ├─ Bias Add               EML add (0-cost)
    ├─ RoPE                   precomputed cos/sin via exp(iθ)
    ├─ GQA Attention           
    │   ├─ Score: Q·K^T / √d  EML dot product + EML div
    │   ├─ Softmax             numerically stable EML softmax
    │   └─ Weighted sum: S·V   EML matmul
    ├─ Output Projection       matmul via precomputed ln(W)
    ├─ Residual Add            EML add (0-cost)
    ├─ RMSNorm (FFN)           CSE: shared ln(x)
    ├─ SwiGLU FFN
    │   ├─ Gate projection     matmul via precomputed ln(W)
    │   ├─ Up projection       matmul via precomputed ln(W)
    │   ├─ SiLU(gate)          EML sigmoid + EML mul
    │   ├─ gate ⊙ up           EML element-wise mul
    │   └─ Down projection     matmul via precomputed ln(W)
    └─ Residual Add            EML add (0-cost)
    │
    ▼
[Final RMSNorm]
    │
    ▼
[LM Head]                    matmul: (1, d_model) × (d_model, vocab)
    │
    ▼
[Argmax]                     discrete comparison (not EML)
    │
    ▼
Token ID
```

### 7.2 EML RMSNorm

The most algebraically dense composite operation. Full CSE derivation:

```
RMSNorm(x, γ, ε):
    // Cache ln(xᵢ) — shared between squaring and normalization
    ln_x[i] = eml_ln(xᵢ)

    // x² via log domain: exp(2 · ln(x))
    sq_sum = Σᵢ eml_exp(eml_mul(2, ln_x[i]))

    // mean(x²) = sq_sum / N via log domain
    mean_sq = eml_div(sq_sum, N)

    // std = √(mean_sq + ε) = exp(0.5 · ln(mean_sq + ε))
    ln_std = eml_mul(0.5, eml_ln(eml_add(mean_sq, ε)))

    // Final: x · γ / std = exp(ln(x) + ln(γ) - ln(std))
    result[i] = eml_exp(eml_sub(eml_add(ln_x[i], ln_γ[i]), ln_std))
```

`ln(x)` is computed once per element and reused in both the squaring step and the final normalization—this is the monoid-morphism factoring.

### 7.3 EML RoPE

Rotary position embeddings use Euler's formula for cos/sin:

```
cos(θ) + i·sin(θ) = exp(iθ)
```

The rotation frequencies are precomputed at initialization:

```rust
// freq = 1 / base^(2i/d_head) via EML
let freq = eml_exp(eml_neg(eml_mul(exponent, eml_ln(base))));
// angle = pos × freq via EML
let angle = eml_mul(pos, freq);
// cos + i·sin = exp(i·angle) via EML exp on complex argument
let rot = eml_exp(Complex64::new(0.0, angle));
```

The rotation itself uses EML mul/add/sub:

```
x'[i]        = eml_sub(eml_mul(x[i], cos), eml_mul(x[i+d/2], sin))
x'[i+d/2]   = eml_add(eml_mul(x[i], sin), eml_mul(x[i+d/2], cos))
```

### 7.4 EML SiLU

```
sigmoid(x) = inv(add(1, exp(neg(x))))
           = 1 / (1 + exp(-x))

silu(x) = x · sigmoid(x)
        = eml_mul(x, sigmoid(x))
```

### 7.5 KV Cache

Standard per-layer KV cache for autoregressive generation:

```rust
pub struct LayerKVCache {
    pub k: Vec<f64>,    // k_cache[pos * kv_dim + h * d_head + d]
    pub v: Vec<f64>,    // v_cache[pos * kv_dim + h * d_head + d]
    pub len: usize,
    pub kv_dim: usize,
}
```

Single-token forward pass uses the cache: new Q attends against all cached K/V.

### 7.6 Non-EML Operations

Three operations are exempt from EML purity:

1. **Token embedding lookup** — Discrete integer indexing
2. **Argmax / sampling** — Integer comparison for token selection
3. **RoPE angle tables** — Precomputed constants (not data-dependent)

Everything else—every multiply, every division, every norm, every softmax—flows through the EML primitive.

---

## 8. Matmul Kernel

The matmul kernel is the performance-critical inner loop. It accounts for >99% of transcendental calls.

### 8.1 CSE Matmul with Precomputed Weights

```
C[i,j] = Σₖ exp(ln|aₖ| + ln|Wₖⱼ|) · sign(aₖ) · sign(Wₖⱼ)
```

Where `ln|W|` and `sign(W)` are precomputed at compile time.

At inference time:
1. **Phase 1**: Compute `ln|a|` and `sign(a)` for the activation vector (once per call)
2. **Phase 2**: Parallel matmul with 4-wide loop unrolling

```rust
// Inner loop (4-wide unrolled, Rayon j-parallel):
for c in 0..chunks {
    let k = c * 4;
    let e0 = (la_mags[a_off+k] + w.magnitudes[b_off+k]).exp();
    acc0 += e0 * la_signs[a_off+k] * w.signs[b_off+k];
    // ... e1, e2, e3 similarly
}
*out = acc0 + acc1 + acc2 + acc3;
```

### 8.2 Optimization History

18 experiments, 9 kept. From 17,238 μs to 456 μs (**37.8× speedup**).

| # | Optimization | Latency (μs) | Speedup | Status |
|---|-------------|-------------|---------|--------|
| 0 | Baseline CSE matmul | 17,238 | — | baseline |
| 2 | Transpose ln(B) for cache locality | 16,494 | 4.3% | kept |
| 5 | Pre-transposed weights at load | 7,082 | 59% | kept |
| 6 | **Real-valued exp bypass** (f64 exp + sign bit) | 4,264 | **75%** | kept |
| 9 | 4-wide loop unroll (ILP) | 4,065 | 76% | kept |
| 10 | Batched atomic counter | 3,995 | 77% | kept |
| 11 | Branchless sign: `1.0 - 2.0*(n&1)` | 3,917 | 77% | kept |
| 16 | **Rayon j-parallelism** (Iverson) | ~710 | **96%** | kept |
| 17 | **Zero-copy par_iter_mut** | ~456 | **97%** | kept |

**Biggest single wins**:
- **Exp 6 (real-valued bypass)**: Avoiding `Complex64::exp()` (which computes cos+sin) — since imaginary parts are exactly 0 or π, use `f64::exp()` + sign bit. 40% relative improvement.
- **Exp 16 (j-parallelism)**: Parallelizing over columns (not rows) with Rayon work-stealing. 5.3× on 8 cores.

**Theoretical limit**: `802,816 exp × 4.7 ns/exp ÷ 8 cores ≈ 472 μs`. Final result (456 μs) is within 3%.

### 8.3 Transcendental Budget

After optimization:
- **exp calls**: 802,816 (one per element, irreducible)
- **ln calls**: 896 (one per activation element, shared across all columns)
- **Reduction**: 50% fewer total transcendentals vs baseline

---

## 9. Metal GPU Backend (metal_eml.rs)

### 9.1 Architecture

```
CPU                              GPU
─────────                        ─────
Tokenize                         
Embedding lookup                 
                    ──────►      ln_split (activation → mag + sign)
                    ──────►      eml_matmul_v4 (QKV)
Bias add, RoPE                   
Attention (Q·K^T, softmax, ·V)   
                    ──────►      eml_matmul_v4 (O projection)
                    ──────►      eml_residual_rms_norm_ln_split
                    ──────►      eml_matmul_v4 (gate+up)
                    ──────►      eml_silu_mul_ln
                    ──────►      eml_matmul_v4 (down)
                    ◄──────      Read result back
Residual add                     
...next layer...                 
```

### 9.2 GPU Kernels

Four pure-EML Metal compute shaders:

#### eml_matmul_v4

SIMD-cooperative matmul with packed sign bits.

- One SIMD group (32 threads) per output element
- 4-way unrolled, SIMD-strided inner loop
- Weight magnitudes in half-precision (f16) — 50% bandwidth reduction
- Weight signs packed as bits: 32 per `uint32` — 16× sign bandwidth reduction
- Inner loop: `exp(ln_a_mag + ln_b_mag) × sign_a × sign_b`
- Final reduction: `simd_sum(acc0 + acc1 + acc2 + acc3)`

#### eml_ln_split

GPU-side ln decomposition of raw activations:

```metal
out_mag[tid]  = log(abs(x) + 1e-30f);   // ln|x|
out_sign[tid] = (x >= 0.0f) ? 1.0f : -1.0f;
```

Eliminates CPU→GPU activation copy for the ln computation.

#### eml_silu_mul_ln

Fused SiLU(gate) × up, output in log domain:

```metal
// ln|silu(g)| = ln|g| - ln(1 + exp(-g))
// result_mag = ln|silu(g)| + ln|u|
// result_sign = sign(g) × sign(u)
```

Output is in (magnitude, sign) form—ready for the next matmul without reconstruction.

#### eml_residual_rms_norm_ln_split

Fused 3-in-1 kernel: residual add + RMSNorm + ln_split.

```metal
// Pass 1: z = a + b (free), accumulate Σz² via EML squaring
// Pass 2: SIMD reduction for mean(z²)
// Pass 3: ln_std = 0.5 · ln(mean_sq + eps)
// Pass 4: out_mag[i] = ln|z[i]| + ln|γ[i]| - ln_std
//          out_sign[i] = sign(z[i]) × sign(γ[i])
```

Stays entirely on GPU. Output is in log domain for the next matmul.

### 9.3 Weight Upload

Weights are uploaded once in half-precision with packed signs:

```rust
pub struct GpuWeights {
    pub buf_mag: Buffer,    // half[cols × inner]: ln|weight|
    pub buf_sign: Buffer,   // u32[ceil(cols×inner/32)]: packed sign bits
    pub inner: usize,
    pub cols: usize,
}
```

### 9.4 Command Buffer Batching

Operations are batched to minimize CPU→GPU round-trips:
- **QKV**: ln_split + 3 matmul dispatches in one command buffer
- **FFN**: fused gate+up → silu_mul_ln → down in one command buffer

### 9.5 Performance

| Backend | tok/s | GPU Memory |
|---------|-------|------------|
| CPU (Rayon, 8 cores) | ~5.5 | — |
| GPU (Metal) | ~30 | ~1 GB |

The GPU kernel is **ALU-bound**: `exp()` costs ~4 cycles on Apple GPU vs ~1 cycle for a standard multiply. Theoretical bandwidth floor (546 GB/s) is 15× below actual latency—transcendental cost, not memory bandwidth, is the bottleneck.

---

## 10. Tokenizer (tokenizer.rs)

GPT-2 style byte-pair encoding, loaded from GGUF metadata.

### 10.1 Byte-to-Unicode Mapping

Standard GPT-2 mapping:
- Bytes 33–126, 161–172, 174–255 → themselves as Unicode chars
- Remaining bytes (0–32, 127–160, 173) → mapped to 256+ range

### 10.2 Encoding

```
Input text → UTF-8 bytes → GPT-2 unicode chars → BPE merging by rank → Token IDs
```

BPE merge priority loaded from `tokenizer.ggml.merges`. Each merge string is `"left right"`. Merges are applied greedily by lowest rank.

### 10.3 Special Tokens

| Token | ID | Usage |
|-------|----|-------|
| `<\|im_start\|>` | 151644 | ChatML turn start |
| `<\|im_end\|>` | 151645 | ChatML turn end / EOT |
| `<\|endoftext\|>` | 151643 | EOS |

### 10.4 ChatML Format

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

---

## 11. Purity Enforcement

### 11.1 The Invariant

> Every numerical result flows through `eml(x, y) = exp(x) - ln(y)`. The only non-EML operations are: token lookup (discrete indexing), argmax/sampling (comparison), and RoPE angle computation (constant, precomputed).

### 11.2 Allowed Raw Transcendentals

Two categories are correct by construction:

1. **The `eml_ops` module**: Functions like `eml_mul(a,b) = (a.ln() + b.ln()).exp()` *are* the EML primitive.
2. **The real-exp bypass**: The hot loop computes `f64::exp(re) * sign` instead of `Complex64::exp()`. Since both inputs are real, `Im[ln(a) + ln(b)] ∈ {0, π}`, so `cos = ±1` and `sin = 0`. This is a mathematical simplification, not a bypass.

### 11.3 Automated Audit (tests/purity_audit.rs)

```rust
// Scans every .rs source file for raw transcendental calls:
//   .exp(), .ln(), .cos(), .sin(), .powf()
// outside of exempt files (eml_ops.rs, autoeml_kernel.rs, autoeml_reference.rs).
//
// Any raw call in an audited file must carry an EML_AUDIT:OK marker
// with a written justification.
```

Two tests:
1. `no_raw_transcendentals_outside_allowed_zones` — No unaudited raw calls
2. `audit_ok_markers_have_justification` — Every marker includes rationale

```bash
cargo test --test purity_audit
```

### 11.4 EML_AUDIT:OK Markers

Used in the matmul inner loop for the real-exp bypass:

```rust
let e0 = (la_mags[a_off+k] + w.magnitudes[b_off+k]).exp();
// EML_AUDIT:OK — real-exp bypass: exp(ln|a|+ln|w|), sign handled separately
```

---

## 12. Command Reference

### Build

```bash
cd emilio

# CPU-only inference engine
cargo build --bin emilio --release

# GPU (Metal) inference engine
cargo build --bin emilio --release --features metal

# AutoEML optimization agent
cargo build --bin autoeml --release
```

### Compile Models

```bash
# GGUF → EML v1
cargo run --bin emilio --release -- \
    ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
    --compile ../models/qwen2.5-0.5b-instruct.eml

# GGUF → EML v2
cargo run --bin emilio --release -- \
    ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
    --compile-v2 ../models/qwen2.5-0.5b-instruct-v2.eml
```

### Run Inference

```bash
# From GGUF (computes ln(W) at load time)
cargo run --bin emilio --release -- \
    ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
    --chat "What is 2+2?"

# From compiled .eml (v1 or v2, auto-detected)
cargo run --bin emilio --release -- \
    ../models/qwen2.5-0.5b-instruct-v2.eml \
    --chat "What is 2+2?"

# GPU inference
cargo run --features metal --bin emilio --release -- \
    ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
    --generate --gpu "What is 2+2?"
```

### Run Tests

```bash
# Purity audit
cargo test --test purity_audit

# All tests
cargo test

# Python verification
cd ../python && python3 verify.py
```

### Kernel Benchmarks

```bash
cargo run --bin autoeml --release -- bench --transposed --iters 10
```

---

## 13. File Map

```
emilio/src/
├── main.rs                 # Tiny POC transformer demo
├── emilio_main.rs          # Main inference CLI (entry point)
├── lib.rs                  # Module exports
├── eml_ops.rs              # EML primitive + all derived ops (scalar + vector)
├── eml_optimizer.rs        # E-graph optimizer (egg): term language, rewrites, cost model
├── engine.rs               # Forward pass: RMSNorm, RoPE, GQA, SwiGLU, KV cache, generation
├── eml_format.rs           # .eml v1 serializer/deserializer
├── eml_v2.rs               # .eml v2: sign+magnitude, fusion, pruning, exec graph
├── gguf.rs                 # GGUF v3 parser: metadata, tensor info, Q8_0 dequantization
├── tokenizer.rs            # BPE tokenizer (GPT-2 style, from GGUF metadata)
├── metal_eml.rs            # Metal GPU backend: context, weight upload, dispatch
├── eml_matmul.metal        # 4 GPU compute shaders (pure EML)
├── model.rs                # Tiny 1-layer transformer POC
├── python.rs               # PyO3 bindings
├── autoeml_main.rs         # Autonomous optimization agent CLI
├── autoeml_kernel.rs       # Kernel under optimization (mutable target)
└── autoeml_reference.rs    # Correctness oracle (DO NOT MODIFY)
```

---

## 14. Dependencies

From `Cargo.toml`:

| Crate | Purpose |
|-------|---------|
| `num-complex` | Complex64 arithmetic for EML |
| `rayon` | Data-parallel iterators for matmul/softmax |
| `egg` | E-graph equality saturation optimizer |
| `ordered-float` | NaN-safe floats for e-graph cost model |
| `metal` (optional) | Apple Metal GPU compute |
| `pyo3` (optional) | Python FFI bindings |

---

## 15. Model Compatibility

Currently targets **Qwen2.5** architecture:

| Parameter | Qwen2.5-0.5B-Instruct |
|-----------|----------------------|
| Parameters | 494M |
| Layers | 24 |
| d_model | 896 |
| d_ff | 4864 |
| n_heads | 14 |
| n_kv_heads | 2 (GQA) |
| d_head | 64 |
| Vocab | 151,936 |
| RoPE base | 1,000,000 |
| Attention | Grouped-query (GQA) |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Position | RoPE |

Any Qwen2.5 variant with compatible architecture parameters should work. Other architectures (LLaMA, Mistral) would require adding their specific layer configurations.

---

## References

- Odrzywołek, A. (2026). "All elementary functions from a single binary operator." arXiv:2603.21852.
- Stepanov, A. & McJones, P. (2009). *Elements of Programming*. Addison-Wesley.
- Iverson, K.E. (1962). *A Programming Language*. John Wiley & Sons.
- Aho, A.V., Lam, M.S., Sethi, R. & Ullman, J.D. (2006). *Compilers: Principles, Techniques, and Tools*. 2nd ed.
- Dechter, R. (2003). *Constraint Processing*. Morgan Kaufmann.
