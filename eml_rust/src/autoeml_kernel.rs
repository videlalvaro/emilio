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
/// For matmul: precomputed ln(weights) if weights are known at load time.
pub struct KernelPrecomputed {
    pub data: Vec<Complex64>,
}

impl KernelPrecomputed {
    pub fn empty() -> Self {
        Self { data: Vec::new() }
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
    // ── Strategy: CSE matmul ──────────────────────────────────────────────
    //
    // Phase 1: Precompute ln(A[i,k]) — full Complex64 to preserve sign
    //          via ln(negative) = ln|x| + iπ
    let ln_a: Vec<Complex64> = a.iter()
        .map(|&v| c_ln(Complex64::new(v, 0.0)))
        .collect();

    // Phase 2: ln(B[k,j]) — use precomputed if available, else compute
    //          Store in TRANSPOSED layout: ln_b_t[j * inner + k] so the
    //          inner k-loop accesses sequential memory addresses.
    let ln_b_raw: Vec<Complex64> = if !precomputed.data.is_empty() {
        precomputed.data.clone()
    } else {
        b.iter()
            .map(|&v| c_ln(Complex64::new(v, 0.0)))
            .collect()
    };

    // Transpose: (inner, cols) → (cols, inner)
    let mut ln_b_t = vec![Complex64::new(0.0, 0.0); inner * cols];
    for k in 0..inner {
        for j in 0..cols {
            ln_b_t[j * inner + k] = ln_b_raw[k * cols + j];
        }
    }

    // Phase 3: C[i,j] = Σ_k exp(ln_A[i,k] + ln_B_T[j,k])
    //          Both ln_a[i*inner+k] and ln_b_t[j*inner+k] are now
    //          sequential in the k-loop → cache-friendly.
    let mut result = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..inner {
                let sum = ln_a[i * inner + k] + ln_b_t[j * inner + k];
                acc += c_exp(sum);
            }
            result[i * cols + j] = acc.re;
        }
    }

    result
}

/// Precompute ln(weights) — called once at model load time.
/// Returns a KernelPrecomputed that eliminates weight-side ln() calls.
pub fn precompute_weights(weights: &[f64]) -> KernelPrecomputed {
    let data: Vec<Complex64> = weights.iter()
        .map(|&v| c_ln(Complex64::new(v, 0.0)))
        .collect();
    KernelPrecomputed { data }
}
