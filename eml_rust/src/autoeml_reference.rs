//! AutoEML Reference Implementations — ground truth for correctness.
//!
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  DO NOT MODIFY THIS FILE.  These are correctness oracles.       │
//! │  If a kernel disagrees with reference, the kernel is wrong.     │
//! └──────────────────────────────────────────────────────────────────┘
//!
//! Each function is the naive EML implementation: correct but slow.
//! No CSE, no precomputation, no optimization.

use num_complex::Complex64;

// ─── Helpers (uncounted — reference doesn't need auditing) ──────────────────

#[inline(always)]
fn cx(v: f64) -> Complex64 {
    Complex64::new(v, 0.0)
}

// ─── MATMUL ─────────────────────────────────────────────────────────────────
//
// C[i,j] = Σ_k A[i,k] × B[k,j]
// Each multiply: exp(ln(a) + ln(b)) — 2 ln + 1 exp = 3 transcendentals
// Total: 3 × rows × inner × cols

pub fn reference_matmul(
    a: &[f64], b: &[f64],
    rows: usize, inner: usize, cols: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..inner {
                let av = cx(a[i * inner + k]);
                let bv = cx(b[k * cols + j]);
                // mul(a, b) = exp(ln(a) + ln(b))
                acc += (av.ln() + bv.ln()).exp();
            }
            result[i * cols + j] = acc.re;
        }
    }
    result
}

// ─── RMSNORM ────────────────────────────────────────────────────────────────
//
// rmsnorm(x, gamma, eps) = x / sqrt(mean(x²) + eps) * gamma
// All via EML: mul → exp(ln+ln), div → exp(ln-ln), sqrt → exp(0.5*ln)

pub fn reference_rmsnorm(x: &[f64], gamma: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let nc = cx(n as f64);

    // mean(x²) = Σ exp(2*ln(x)) / N  (via EML mul: x*x = exp(ln(x)+ln(x)) = exp(2*ln(x)))
    let sq_sum: Complex64 = x.iter()
        .map(|&v| {
            let lv = cx(v).ln();
            (cx(2.0) * lv).exp()       // x² = exp(2*ln(x))
        })
        .fold(Complex64::new(0.0, 0.0), |a, b| a + b);

    // mean = exp(ln(sq_sum) - ln(N))
    let mean_sq = (sq_sum.ln() - nc.ln()).exp();

    // std = sqrt(mean + eps) = exp(0.5 * ln(mean + eps))
    let std = (cx(0.5) * (mean_sq + cx(eps)).ln()).exp();
    let ln_std = std.ln();

    // x_i / std * gamma_i = exp(ln(x_i) - ln(std) + ln(gamma_i))
    (0..n).map(|i| {
        let ln_x = cx(x[i]).ln();
        let ln_g = cx(gamma[i]).ln();
        (ln_x - ln_std + ln_g).exp().re
    }).collect()
}

// ─── SOFTMAX ────────────────────────────────────────────────────────────────
//
// softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
// Sub is 0-cost, exp is counted, div via log domain.

pub fn reference_softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // exp(x_i - max) — sub is free, exp costs 1 each
    let exps: Vec<f64> = x.iter()
        .map(|&xi| cx(xi - max).exp().re)
        .collect();

    // Z = sum (free)
    let z: f64 = exps.iter().sum();
    let ln_z = cx(z).ln();

    // softmax_i = exp(ln(exp_i) - ln(Z))
    exps.iter()
        .map(|&e| {
            let ln_e = cx(e).ln();
            (ln_e - ln_z).exp().re
        })
        .collect()
}

// ─── SILU (SwiGLU activation) ───────────────────────────────────────────────
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Via EML: exp(ln(x) + ln(sigmoid(x)))

pub fn reference_silu(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| {
        let xc = cx(v);
        let one = cx(1.0);
        let sig = one / (one + (-xc).exp());    // sigmoid via exp
        (xc.ln() + sig.ln()).exp().re            // x * sig via log domain
    }).collect()
}

// ─── ROPE (Rotary Position Embeddings) ──────────────────────────────────────
//
// rope(x, cos_θ, sin_θ) for pairs (x0, x1):
//   out0 = x0 * cos_θ - x1 * sin_θ
//   out1 = x0 * sin_θ + x1 * cos_θ
// Each mul via EML, sub/add are free.

pub fn reference_rope(
    x: &[f64],       // (d_head,) — must be even length
    cos_t: &[f64],   // (d_head/2,)
    sin_t: &[f64],   // (d_head/2,)
) -> Vec<f64> {
    let half = x.len() / 2;
    let mut out = vec![0.0f64; x.len()];

    for i in 0..half {
        let x0 = cx(x[i]);
        let x1 = cx(x[i + half]);
        let ct = cx(cos_t[i]);
        let st = cx(sin_t[i]);

        // mul via log domain
        let x0_cos = (x0.ln() + ct.ln()).exp();
        let x1_sin = (x1.ln() + st.ln()).exp();
        let x0_sin = (x0.ln() + st.ln()).exp();
        let x1_cos = (x1.ln() + ct.ln()).exp();

        // add/sub are free
        out[i]        = (x0_cos - x1_sin).re;
        out[i + half] = (x0_sin + x1_cos).re;
    }
    out
}

// ─── Utility: numpy-style allclose ──────────────────────────────────────────

pub fn allclose(a: &[f64], b: &[f64], rtol: f64, atol: f64) -> (bool, f64) {
    assert_eq!(a.len(), b.len());
    let mut max_err = 0.0f64;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let err = (av - bv).abs();
        let tol = atol + rtol * bv.abs();
        if err > max_err {
            max_err = err;
        }
        if err > tol {
            return (false, max_err);
        }
    }
    (true, max_err)
}
