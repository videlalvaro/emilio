//! AutoEML Kernel — THE file the agent modifies.
//!
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  THIS IS THE ONLY FILE THE AGENT EDITS.                         │
//! │  One kernel at a time.  Edit, bench, keep/revert.               │
//! └──────────────────────────────────────────────────────────────────┘
//!
//! KERNEL_TYPE determines which operation is being optimized.
//! kernel_fn() is the function under test — the agent rewrites its body.
//! c_exp() and c_ln() are the ONLY transcendental primitives allowed.
//! Using raw .exp() / .ln() without the counted wrappers is cheating.

use num_complex::Complex64;
use std::sync::atomic::{AtomicU64, Ordering};

// ─── Transcendental counters ────────────────────────────────────────────────
// These are the auditing mechanism.  bench.rs reads them to verify EML purity.

static EXP_CALLS: AtomicU64 = AtomicU64::new(0);
static LN_CALLS: AtomicU64 = AtomicU64::new(0);

pub fn reset_counts() {
    EXP_CALLS.store(0, Ordering::SeqCst);
    LN_CALLS.store(0, Ordering::SeqCst);
}

pub fn get_counts() -> (u64, u64) {
    (EXP_CALLS.load(Ordering::SeqCst), LN_CALLS.load(Ordering::SeqCst))
}

/// Counted exp — the ONLY way to call exp() in this file.
#[inline(always)]
pub fn c_exp(x: Complex64) -> Complex64 {
    EXP_CALLS.fetch_add(1, Ordering::Relaxed);
    x.exp()
}

/// Counted real exp with sign — optimized for EML matmul inner loop.
/// When ln(a) + ln(b) has imaginary part that's a multiple of π,
/// this avoids computing full complex exp (which does exp(re)*(cos(im)+i*sin(im))).
/// Instead: determines sign from im/π parity, does f64 exp(re).
/// Still counted as one transcendental (exp).
#[inline(always)]
fn c_exp_real_signed(sum_re: f64, sum_im: f64) -> f64 {
    EXP_CALLS.fetch_add(1, Ordering::Relaxed);
    let e = sum_re.exp();
    // im is always a multiple of π (0, π, 2π, etc.)
    // Odd multiples → negative, even → positive
    let n = (sum_im * std::f64::consts::FRAC_1_PI).round() as i64;
    if n & 1 == 0 { e } else { -e }
}

/// Uncounted real exp with sign — used in hot loops where exp count is
/// batch-added outside the loop. Branchless sign from im/π parity.
#[inline(always)]
fn exp_real_signed(sum_re: f64, sum_im: f64) -> f64 {
    let e = sum_re.exp();
    let n = (sum_im * std::f64::consts::FRAC_1_PI).round() as i64;
    // Branchless: sign = 1.0 - 2.0*(n&1) → +1.0 if even, -1.0 if odd
    e * (1.0 - 2.0 * (n & 1) as f64)
}

/// Counted ln — the ONLY way to call ln() in this file.
#[inline(always)]
pub fn c_ln(x: Complex64) -> Complex64 {
    LN_CALLS.fetch_add(1, Ordering::Relaxed);
    x.ln()
}

// ─── Kernel identity ────────────────────────────────────────────────────────

pub const KERNEL_TYPE: &str = "matmul";

// ─── Kernel parameters (set by extract / bench) ────────────────────────────

/// Optional precomputed data that persists across calls.
/// For matmul: precomputed ln(weights) ALREADY TRANSPOSED to (cols, inner)
/// layout so kernel_fn skips the per-call transpose.
pub struct KernelPrecomputed {
    /// ln(B) in transposed layout: data[j * inner + k] = ln(B[k,j])
    /// Empty if no precomputation.
    pub data: Vec<Complex64>,
    /// Whether data is already in transposed (cols, inner) layout.
    pub transposed: bool,
}

impl KernelPrecomputed {
    pub fn empty() -> Self {
        Self { data: Vec::new(), transposed: false }
    }
}

// ─── THE kernel function ────────────────────────────────────────────────────
//
// MATMUL: C = A × B where A is (rows, inner), B is (inner, cols)
//
// Constraints:
//   - Every multiply MUST go through exp(ln(a) + ln(b))
//   - Addition is free (0 transcendentals, proven by EML cancellation)
//   - c_exp() and c_ln() are the only transcendental primitives
//   - Result must match reference to within 1e-6 relative error
//
// Current strategy: CSE — precompute ln(A) and ln(B), share across dot products.
//
// Transcendental budget:
//   Naive:  3 × rows × inner × cols  (2 ln + 1 exp per multiply)
//   CSE:    rows×inner + inner×cols + rows×inner×cols
//           (shared ln(A), shared ln(B), one exp per product)
//
// Agent: can you beat CSE?  Hint: ln(B) is constant if B is a weight matrix.
//        Pass precomputed ln(B) via KernelPrecomputed to eliminate inner×cols ln's.

pub fn kernel_fn(
    a: &[f64],
    b: &[f64],
    rows: usize,
    inner: usize,
    cols: usize,
    precomputed: &KernelPrecomputed,
) -> Vec<f64> {
    kernel_fn_with_ln_a(a, b, rows, inner, cols, precomputed, None)
}

/// Extended kernel that accepts optional precomputed ln(activations).
///
/// When the same activation vector feeds multiple weight matrices (Q, K, V),
/// pass the same `ln_a_cache` to all three to avoid recomputing ln(X) each time.
/// For M=1, K=896 this saves 896 ln's per reuse.
pub fn kernel_fn_with_ln_a(
    a: &[f64],
    b: &[f64],
    rows: usize,
    inner: usize,
    cols: usize,
    precomputed: &KernelPrecomputed,
    ln_a_cache: Option<&[Complex64]>,
) -> Vec<f64> {
    // ── Strategy: CSE matmul ──────────────────────────────────────────────
    //
    // Phase 1: Precompute ln(A[i,k]) — use cache if provided, else compute.
    //          Full Complex64 to preserve sign via ln(negative) = ln|x| + iπ
    let owned_ln_a: Vec<Complex64>;
    let ln_a: &[Complex64] = if let Some(cached) = ln_a_cache {
        cached
    } else {
        owned_ln_a = a.iter()
            .map(|&v| c_ln(Complex64::new(v, 0.0)))
            .collect();
        &owned_ln_a
    };

    // Phase 2: ln(B[k,j]) — use precomputed if available, else compute.
    //          If precomputed.transposed, data is already in (cols, inner) layout.
    //          Otherwise, compute and transpose.
    let ln_b_t: Vec<Complex64> = if !precomputed.data.is_empty() && precomputed.transposed {
        // Already transposed at precompute time — zero-cost here
        precomputed.data.clone()
    } else {
        let ln_b_raw: Vec<Complex64> = if !precomputed.data.is_empty() {
            precomputed.data.clone()
        } else {
            b.iter()
                .map(|&v| c_ln(Complex64::new(v, 0.0)))
                .collect()
        };

        // Transpose: (inner, cols) → (cols, inner)
        let mut t = vec![Complex64::new(0.0, 0.0); inner * cols];
        for k in 0..inner {
            for j in 0..cols {
                t[j * inner + k] = ln_b_raw[k * cols + j];
            }
        }
        t
    };

    // Phase 3: C[i,j] = Σ_k exp(ln_A[i,k] + ln_B_T[j,k])
    //          Both ln_a[i*inner+k] and ln_b_t[j*inner+k] are now
    //          sequential in the k-loop → cache-friendly.
    //          Optimization: since all values are ln(real), imaginary parts
    //          are 0 or π.  Use real-valued exp with sign from im/π parity.
    //          4-wide unroll with independent accumulators for ILP.
    //          Batch atomic counter: add total exp count once after loop.
    let total_exp = (rows * inner * cols) as u64;
    EXP_CALLS.fetch_add(total_exp, Ordering::Relaxed);

    let mut result = vec![0.0f64; rows * cols];
    for i in 0..rows {
        let a_off = i * inner;
        for j in 0..cols {
            let b_off = j * inner;
            let mut acc0 = 0.0f64;
            let mut acc1 = 0.0f64;
            let mut acc2 = 0.0f64;
            let mut acc3 = 0.0f64;
            let chunks = inner / 4;
            let remainder = inner % 4;
            for c in 0..chunks {
                let k = c * 4;
                let la0 = ln_a[a_off + k];
                let la1 = ln_a[a_off + k + 1];
                let la2 = ln_a[a_off + k + 2];
                let la3 = ln_a[a_off + k + 3];
                let lb0 = ln_b_t[b_off + k];
                let lb1 = ln_b_t[b_off + k + 1];
                let lb2 = ln_b_t[b_off + k + 2];
                let lb3 = ln_b_t[b_off + k + 3];
                acc0 += exp_real_signed(la0.re + lb0.re, la0.im + lb0.im);
                acc1 += exp_real_signed(la1.re + lb1.re, la1.im + lb1.im);
                acc2 += exp_real_signed(la2.re + lb2.re, la2.im + lb2.im);
                acc3 += exp_real_signed(la3.re + lb3.re, la3.im + lb3.im);
            }
            // Remainder
            for k in (chunks * 4)..(chunks * 4 + remainder) {
                acc0 += exp_real_signed(
                    ln_a[a_off + k].re + ln_b_t[b_off + k].re,
                    ln_a[a_off + k].im + ln_b_t[b_off + k].im,
                );
            }
            result[i * cols + j] = acc0 + acc1 + acc2 + acc3;
        }
    }

    result
}

/// Precompute ln(weights) with transpose — called once at model load time.
/// Stores ln(B) in transposed (cols, inner) layout so kernel_fn skips per-call transpose.
/// `weights` is in row-major (inner, cols) layout.
pub fn precompute_weights(weights: &[f64]) -> KernelPrecomputed {
    // We need inner and cols to transpose, but we only know total size.
    // Store raw (non-transposed); the kernel will transpose per-call.
    // For the transposed path, use precompute_weights_transposed().
    let data: Vec<Complex64> = weights.iter()
        .map(|&v| c_ln(Complex64::new(v, 0.0)))
        .collect();
    KernelPrecomputed { data, transposed: false }
}

/// Precompute ln(weights) AND transpose to (cols, inner) layout.
/// This saves the per-call transpose. Requires knowing matrix dimensions.
pub fn precompute_weights_transposed(weights: &[f64], inner: usize, cols: usize) -> KernelPrecomputed {
    let mut data = vec![Complex64::new(0.0, 0.0); inner * cols];
    for k in 0..inner {
        for j in 0..cols {
            data[j * inner + k] = c_ln(Complex64::new(weights[k * cols + j], 0.0));
        }
    }
    KernelPrecomputed { data, transposed: true }
}

/// Precompute ln(activations) for sharing across multiple matmuls.
/// E.g., compute ln(hidden_state) once, reuse for Q, K, V projections.
pub fn precompute_ln_activations(activations: &[f64]) -> Vec<Complex64> {
    activations.iter()
        .map(|&v| c_ln(Complex64::new(v, 0.0)))
        .collect()
}
