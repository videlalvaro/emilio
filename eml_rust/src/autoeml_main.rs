//! AutoEML — Autonomous EML Graph Optimization Agent.
//!
//! Inspired by AutoKernel (RightNow-AI): give it an EML operation,
//! go to sleep, wake up to a faster evaluation strategy.
//!
//! Commands:
//!   profile   — Analytical breakdown of transcendentals per transformer op
//!   extract   — Set up autoeml_kernel.rs for a specific operation type
//!   bench     — Run correctness + throughput benchmark on the current kernel
//!   verify    — Run all operation types, report aggregate results
//!
//! Usage:
//!   cargo run --bin autoeml --release -- profile --model qwen2-0.5b
//!   cargo run --bin autoeml --release -- bench
//!   cargo run --bin autoeml --release -- bench --size 896

use eml_rust_core::autoeml_kernel;
use eml_rust_core::autoeml_reference;
use std::time::Instant;

// ─── CLI ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "profile" => cmd_profile(&args[2..]),
        "bench"   => cmd_bench(&args[2..]),
        "verify"  => cmd_verify(&args[2..]),
        "help" | "--help" | "-h" => print_usage(),
        other => {
            eprintln!("Unknown command: {other}");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("AutoEML — Autonomous EML Graph Optimization");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  profile               Analytical transcendental breakdown");
    eprintln!("  bench [--size N]      Benchmark current kernel (default: model-sized)");
    eprintln!("  verify                Verify all operation types end-to-end");
    eprintln!();
    eprintln!("The agent edits autoeml_kernel.rs, then runs: cargo run --bin autoeml --release -- bench");
}

// ─── PROFILE ────────────────────────────────────────────────────────────────
//
// Analytical model of transcendental operations per transformer forward pass.
// Uses Qwen2.5-0.5B dimensions by default.

fn cmd_profile(_args: &[String]) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AutoEML Profiler — Transcendental Budget Analysis          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Qwen2.5-0.5B config
    let d = 896usize;        // d_model
    let nh = 14usize;        // n_heads
    let nkv = 2usize;        // n_kv_heads
    let dh = 64usize;        // d_head
    let dff = 4864usize;     // d_ff
    let nl = 24usize;        // n_layers
    let vocab = 151936usize; // vocab_size
    let seq = 1usize;        // single-token generation

    println!("Model: Qwen2.5-0.5B-Instruct (seq_len={seq})");
    println!("  d_model={d}, n_heads={nh}, n_kv_heads={nkv}, d_head={dh}, d_ff={dff}");
    println!();

    // Helper: CSE matmul transcendentals for (M, K, N)
    let cse_matmul = |m: usize, k: usize, n: usize| -> (u64, u64, u64) {
        let ln_a = (m * k) as u64;
        let ln_b = (k * n) as u64;
        let exp = (m * k * n) as u64;
        (ln_a, ln_b, exp)
    };

    // Helper: precomputed-weight matmul (0 ln for weights)
    let precomp_matmul = |m: usize, k: usize, n: usize| -> (u64, u64, u64) {
        let ln_a = (m * k) as u64;
        let ln_b = 0u64;
        let exp = (m * k * n) as u64;
        (ln_a, ln_b, exp)
    };

    println!("Per-layer breakdown (CSE baseline → with precomputed ln(W)):");
    println!("{:-<75}", "");

    // RMSNorm (attention)
    // approx: d ln(x) + d exp(2*ln) + div + sqrt + d ln + d exp = ~4d + const
    let rmsnorm_trans = 4 * d as u64 + 5;

    // Q projection: (seq, d) × (d, nh*dh) = (seq, d) × (d, d)
    let (qa, qb, qe) = cse_matmul(seq, d, nh * dh);
    let (pa, _pb, pe) = precomp_matmul(seq, d, nh * dh);

    // K projection: (seq, d) × (d, nkv*dh)
    let (ka, kb, ke) = cse_matmul(seq, d, nkv * dh);
    let (pka, _pkb, pke) = precomp_matmul(seq, d, nkv * dh);

    // V projection: same as K
    let (va, vb, ve) = (ka, kb, ke);
    let (pva, _pvb, pve) = (pka, _pkb, pke);

    let qkv_cse = qa + qb + qe + ka + kb + ke + va + vb + ve;
    let qkv_pre = pa + pe + pka + pke + pva + pve;

    println!("  RMSNorm (attn):     {:>12} transcendentals", rmsnorm_trans);
    println!("  Q projection:       {:>12} CSE → {:>12} precomputed", qa+qb+qe, pa+pe);
    println!("  K projection:       {:>12} CSE → {:>12} precomputed", ka+kb+ke, pka+pke);
    println!("  V projection:       {:>12} CSE → {:>12} precomputed", va+vb+ve, pva+pve);

    // Attention: Q@K^T — both sides are activations, no precomputation
    // (seq, dh) × (dh, seq) per head, nh heads
    let (aa, ab, ae) = cse_matmul(seq, dh, seq);
    let attn_qk = (aa + ab + ae) * nh as u64;

    // Softmax: 2*seq + seq + 1 ≈ 3*seq per head
    let softmax_trans = (3 * seq as u64 + 1) * nh as u64;

    // Attn@V: (seq, seq) × (seq, dh) per head — activations, no precomp
    let attn_v = attn_qk; // same shape

    // Output projection: (seq, d) × (d, d)
    let (oa, ob, oe) = cse_matmul(seq, d, d);
    let (poa, _pob, poe) = precomp_matmul(seq, d, d);

    println!("  Attention Q@K^T:    {:>12} (activations, no precomp)", attn_qk);
    println!("  Softmax:            {:>12}", softmax_trans);
    println!("  Attention Attn@V:   {:>12} (activations, no precomp)", attn_v);
    println!("  Output projection:  {:>12} CSE → {:>12} precomputed", oa+ob+oe, poa+poe);

    // RMSNorm (FFN)
    println!("  RMSNorm (FFN):      {:>12}", rmsnorm_trans);

    // SwiGLU FFN
    // Gate: (seq, d) × (d, dff)
    let (ga, gb, ge) = cse_matmul(seq, d, dff);
    let (pga, _pgb, pge) = precomp_matmul(seq, d, dff);
    // Up: same shape
    let (ua, ub, ue) = (ga, gb, ge);
    let (pua, _pub_, pue) = (pga, _pgb, pge);
    // SiLU(gate): ~4 per element (ln, exp, ln, exp)
    let silu_trans = 4 * dff as u64;
    // gate * silu: elementwise mul = 3 per element
    let gate_mul = 3 * dff as u64;
    // Down: (seq, dff) × (dff, d)
    let (da, db, de) = cse_matmul(seq, dff, d);
    let (pda, _pdb, pde) = precomp_matmul(seq, dff, d);

    let ffn_cse = ga+gb+ge + ua+ub+ue + silu_trans + gate_mul + da+db+de;
    let ffn_pre = pga+pge + pua+pue + silu_trans + gate_mul + pda+pde;

    println!("  Gate projection:    {:>12} CSE → {:>12} precomputed", ga+gb+ge, pga+pge);
    println!("  Up projection:      {:>12} CSE → {:>12} precomputed", ua+ub+ue, pua+pue);
    println!("  SiLU activation:    {:>12}", silu_trans);
    println!("  Gate × SiLU mul:    {:>12}", gate_mul);
    println!("  Down projection:    {:>12} CSE → {:>12} precomputed", da+db+de, pda+pde);

    let layer_cse = 2 * rmsnorm_trans + qkv_cse + attn_qk + softmax_trans
                    + attn_v + oa+ob+oe + ffn_cse;
    let layer_pre = 2 * rmsnorm_trans + qkv_pre + attn_qk + softmax_trans
                    + attn_v + poa+poe + ffn_pre;

    println!("{:-<75}", "");
    println!("  Per-layer total:    {:>12} CSE → {:>12} precomputed", layer_cse, layer_pre);

    let total_cse = layer_cse * nl as u64;
    let total_pre = layer_pre * nl as u64;

    // LM head: (seq, d) × (d, vocab)
    let (lha, lhb, lhe) = cse_matmul(seq, d, vocab);
    let (plha, _plhb, plhe) = precomp_matmul(seq, d, vocab);

    println!();
    println!("Full model ({nl} layers + LM head):");
    println!("  Layers:             {:>12} CSE → {:>12} precomputed",
             total_cse, total_pre);
    println!("  LM head:            {:>12} CSE → {:>12} precomputed",
             lha+lhb+lhe, plha+plhe);

    let grand_cse = total_cse + lha + lhb + lhe;
    let grand_pre = total_pre + plha + plhe;
    let reduction = 100.0 * (1.0 - grand_pre as f64 / grand_cse as f64);

    println!("{:-<75}", "");
    println!("  TOTAL:              {:>12} CSE → {:>12} precomputed  ({:.1}% reduction)",
             grand_cse, grand_pre, reduction);

    // Breakdown by operation type
    println!();
    println!("Optimization priority (Amdahl's law):");
    let ops = vec![
        ("FFN matmuls (gate+up+down)", (ffn_cse * nl as u64) as f64),
        ("QKV + output projections", ((qkv_cse + oa+ob+oe) * nl as u64) as f64),
        ("LM head", (lha+lhb+lhe) as f64),
        ("SiLU + gate mul", ((silu_trans + gate_mul) * nl as u64) as f64),
        ("Attention Q@K^T + Attn@V", ((attn_qk + attn_v) * nl as u64) as f64),
        ("RMSNorm", (2 * rmsnorm_trans * nl as u64) as f64),
        ("Softmax", (softmax_trans * nl as u64) as f64),
    ];

    let total = grand_cse as f64;
    for (name, count) in &ops {
        let pct = 100.0 * count / total;
        let bar_len = (pct * 0.4) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:<30} {:>5.1}% {:>12.0} {}", name, pct, count, bar);
    }
}

// ─── BENCH ──────────────────────────────────────────────────────────────────
//
// Fixed benchmark harness.  DO NOT MODIFY (the agent only edits kernel.rs).
//
// 5-stage correctness:
//   1. Smoke test (tiny 4×4)
//   2. Shape sweep (8×8 through model-size)
//   3. Numerical stability (near-zero, large, negative values)
//   4. Determinism (same input → same output)
//   5. EML purity (transcendental count matches expectation)
//
// Performance:
//   - Wall time (median of N iterations)
//   - Transcendentals per element
//   - Throughput (elements/sec)

fn cmd_bench(args: &[String]) {
    let size = parse_arg(args, "--size").unwrap_or(896);
    let iters = parse_arg(args, "--iters").unwrap_or(5);
    let precomp = args.iter().any(|a| a == "--precomputed");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AutoEML Benchmark — {}                             ║", autoeml_kernel::KERNEL_TYPE);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let kernel_type = autoeml_kernel::KERNEL_TYPE;
    match kernel_type {
        "matmul" => bench_matmul(size, iters, precomp),
        other => {
            eprintln!("Unknown kernel type: {other}");
            std::process::exit(1);
        }
    }
}

fn bench_matmul(size: usize, iters: usize, use_precomp: bool) {
    let mut all_pass = true;

    // ── Stage 1: Smoke test (4×4) ───────────────────────────────────────
    print!("Stage 1: Smoke test (4×4)... ");
    let (a, b) = gen_matmul_data(4, 4, 4, 42);
    let pc = if use_precomp {
        autoeml_kernel::precompute_weights(&b)
    } else {
        autoeml_kernel::KernelPrecomputed::empty()
    };
    autoeml_kernel::reset_counts();
    let result = autoeml_kernel::kernel_fn(&a, &b, 4, 4, 4, &pc);
    let reference = autoeml_reference::reference_matmul(&a, &b, 4, 4, 4);
    let (ok, max_err) = autoeml_reference::allclose(&result, &reference, 1e-6, 1e-8);
    if ok {
        println!("PASS (max_err={:.2e})", max_err);
    } else {
        println!("FAIL (max_err={:.2e})", max_err);
        all_pass = false;
    }

    // ── Stage 2: Shape sweep ────────────────────────────────────────────
    print!("Stage 2: Shape sweep... ");
    let shapes = [(8, 8, 8), (16, 32, 16), (64, 64, 64), (1, size, size)];
    let mut sweep_pass = true;
    for &(m, k, n) in &shapes {
        let (a, b) = gen_matmul_data(m, k, n, 123);
        let pc = if use_precomp {
            autoeml_kernel::precompute_weights(&b)
        } else {
            autoeml_kernel::KernelPrecomputed::empty()
        };
        autoeml_kernel::reset_counts();
        let result = autoeml_kernel::kernel_fn(&a, &b, m, k, n, &pc);
        let reference = autoeml_reference::reference_matmul(&a, &b, m, k, n);
        let (ok, _) = autoeml_reference::allclose(&result, &reference, 1e-6, 1e-8);
        if !ok {
            print!("FAIL({m}×{k}×{n}) ");
            sweep_pass = false;
        }
    }
    if sweep_pass { println!("PASS ({} shapes)", shapes.len()); }
    else { println!(); all_pass = false; }

    // ── Stage 3: Numerical stability ────────────────────────────────────
    print!("Stage 3: Numerical stability... ");
    let cases = [
        ("near-zero", gen_matmul_stability(8, 1e-10, 43)),
        ("large", gen_matmul_stability(8, 1e6, 44)),
        ("negative", gen_matmul_negative(8, 45)),
        ("mixed-sign", gen_matmul_mixed(8, 46)),
    ];
    let mut stab_pass = true;
    for (name, (a, b)) in &cases {
        let n = (a.len() as f64).sqrt() as usize;
        let pc = if use_precomp {
            autoeml_kernel::precompute_weights(b)
        } else {
            autoeml_kernel::KernelPrecomputed::empty()
        };
        autoeml_kernel::reset_counts();
        let result = autoeml_kernel::kernel_fn(a, b, n, n, n, &pc);
        let reference = autoeml_reference::reference_matmul(a, b, n, n, n);
        let (ok, max_err) = autoeml_reference::allclose(&result, &reference, 1e-4, 1e-6);
        if !ok {
            print!("FAIL({name}, err={max_err:.2e}) ");
            stab_pass = false;
        }
    }
    if stab_pass { println!("PASS (4 cases)"); }
    else { println!(); all_pass = false; }

    // ── Stage 4: Determinism ────────────────────────────────────────────
    print!("Stage 4: Determinism... ");
    let (a, b) = gen_matmul_data(16, 16, 16, 99);
    let pc = if use_precomp {
        autoeml_kernel::precompute_weights(&b)
    } else {
        autoeml_kernel::KernelPrecomputed::empty()
    };
    autoeml_kernel::reset_counts();
    let r1 = autoeml_kernel::kernel_fn(&a, &b, 16, 16, 16, &pc);
    autoeml_kernel::reset_counts();
    let r2 = autoeml_kernel::kernel_fn(&a, &b, 16, 16, 16, &pc);
    if r1 == r2 { println!("PASS"); }
    else { println!("FAIL (non-deterministic)"); all_pass = false; }

    // ── Stage 5: EML purity (transcendental count) ──────────────────────
    print!("Stage 5: EML purity... ");
    let (m, k, n) = (4, 8, 4);
    let (a, b) = gen_matmul_data(m, k, n, 55);
    let pc_purity = if use_precomp {
        // Precompute outside counting window
        let p = autoeml_kernel::precompute_weights(&b);
        autoeml_kernel::reset_counts();
        p
    } else {
        autoeml_kernel::reset_counts();
        autoeml_kernel::KernelPrecomputed::empty()
    };
    let _ = autoeml_kernel::kernel_fn(&a, &b, m, k, n, &pc_purity);
    let (exp_count, ln_count) = autoeml_kernel::get_counts();
    let total_trans = exp_count + ln_count;
    let elements = (m * n) as u64;
    let trans_per_elem = total_trans as f64 / elements as f64;

    // Minimum possible: at least K exp per element (one exp per accumulated product)
    let min_exp = (m * k * n) as u64;
    let purity_ok = exp_count >= min_exp && total_trans > 0;
    if purity_ok {
        println!("PASS (exp={exp_count}, ln={ln_count}, total={total_trans}, {trans_per_elem:.1}/elem)");
    } else {
        println!("FAIL (exp={exp_count}, ln={ln_count} — expected at least {min_exp} exp)");
        all_pass = false;
    }

    // ── Performance ─────────────────────────────────────────────────────
    println!();
    println!("Performance (size={size}, iters={iters}):");
    let (m, k, n) = (1, size, size);
    let (a, b) = gen_matmul_data(m, k, n, 77);
    let pc = if use_precomp {
        autoeml_kernel::precompute_weights(&b)
    } else {
        autoeml_kernel::KernelPrecomputed::empty()
    };

    // Warm up
    autoeml_kernel::reset_counts();
    let _ = autoeml_kernel::kernel_fn(&a, &b, m, k, n, &pc);
    let (warm_exp, warm_ln) = autoeml_kernel::get_counts();

    // Timed iterations
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        autoeml_kernel::reset_counts();
        let start = Instant::now();
        let _ = autoeml_kernel::kernel_fn(&a, &b, m, k, n, &pc);
        let elapsed = start.elapsed();
        times.push(elapsed.as_micros() as u64);
    }
    times.sort();
    let median_us = times[iters / 2];
    let elements_out = (m * n) as u64;
    let trans_total = warm_exp + warm_ln;
    let trans_per_out = trans_total as f64 / elements_out as f64;
    let throughput = elements_out as f64 / (median_us as f64 / 1_000_000.0);

    println!("  dimensions:           ({m}, {k}) × ({k}, {n})");
    println!("  transcendentals:      {trans_total} ({trans_per_out:.1}/output element)");
    println!("    exp calls:          {warm_exp}");
    println!("    ln calls:           {warm_ln}");
    println!("  median latency:       {} μs", median_us);
    println!("  throughput:           {:.0} elements/sec", throughput);
    if use_precomp {
        println!("  precomputed:          YES (ln(weights) excluded from count)");
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println!();
    let correctness = if all_pass { "PASS" } else { "FAIL" };
    println!("correctness: {correctness}");
    println!("kernel_type: {}", autoeml_kernel::KERNEL_TYPE);
    println!("transcendentals: {trans_total}");
    println!("trans_per_element: {trans_per_out:.1}");
    println!("throughput_elems_sec: {throughput:.0}");
    println!("latency_us: {median_us}");
    println!("exp_calls: {warm_exp}");
    println!("ln_calls: {warm_ln}");

    if !all_pass {
        std::process::exit(1);
    }
}

// ─── VERIFY ─────────────────────────────────────────────────────────────────

fn cmd_verify(_args: &[String]) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AutoEML Verification — All Operations                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut total_pass = 0;
    let mut total_fail = 0;

    // Matmul verification at model-relevant sizes
    println!("── Matmul ──────────────────────────────────────────────────");
    let shapes = [
        (1, 896, 896, "QKV projection"),
        (1, 896, 128, "K/V projection"),
        (1, 896, 4864, "FFN gate/up"),
        (1, 4864, 896, "FFN down"),
        (1, 64, 32, "Attention Q@K^T"),
    ];

    for &(m, k, n, label) in &shapes {
        let (a, b) = gen_matmul_data(m, k, n, 42);
        let pc = autoeml_kernel::KernelPrecomputed::empty();
        autoeml_kernel::reset_counts();
        let result = autoeml_kernel::kernel_fn(&a, &b, m, k, n, &pc);
        let reference = autoeml_reference::reference_matmul(&a, &b, m, k, n);
        let (ok, max_err) = autoeml_reference::allclose(&result, &reference, 1e-5, 1e-7);
        let (exp_c, ln_c) = autoeml_kernel::get_counts();
        let status = if ok { total_pass += 1; "PASS" } else { total_fail += 1; "FAIL" };
        println!("  {status} {label:<20} ({m}×{k})×({k}×{n})  err={max_err:.2e}  exp={exp_c} ln={ln_c}");
    }

    // Precomputed weight matmul
    println!();
    println!("── Matmul (precomputed weights) ───────────────────────────");
    for &(m, k, n, label) in &shapes {
        let (a, b) = gen_matmul_data(m, k, n, 42);
        let pc = autoeml_kernel::precompute_weights(&b);
        autoeml_kernel::reset_counts();
        let result = autoeml_kernel::kernel_fn(&a, &b, m, k, n, &pc);
        let reference = autoeml_reference::reference_matmul(&a, &b, m, k, n);
        let (ok, max_err) = autoeml_reference::allclose(&result, &reference, 1e-5, 1e-7);
        let (exp_c, ln_c) = autoeml_kernel::get_counts();
        let status = if ok { total_pass += 1; "PASS" } else { total_fail += 1; "FAIL" };
        println!("  {status} {label:<20} ({m}×{k})×({k}×{n})  err={max_err:.2e}  exp={exp_c} ln={ln_c}");
    }

    println!();
    println!("Results: {total_pass} passed, {total_fail} failed");
    if total_fail > 0 { std::process::exit(1); }
}

// ─── Data generation ────────────────────────────────────────────────────────

/// Simple xoshiro256** PRNG for reproducible test data.
struct Rng { s: [u64; 4] }

impl Rng {
    fn new(seed: u64) -> Self {
        let mut s = [seed, seed ^ 0x123456789, seed.wrapping_mul(6364136223846793005), seed ^ 0xdeadbeef];
        // Warm up
        for _ in 0..20 {
            let t = s[1] << 17;
            s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
            s[2] ^= t; s[3] = s[3].rotate_left(45);
        }
        Rng { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1]; self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t; self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [-scale, scale]
    fn uniform(&mut self, scale: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * scale
    }
}

fn gen_matmul_data(rows: usize, inner: usize, cols: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let a: Vec<f64> = (0..rows * inner).map(|_| rng.uniform(2.0)).collect();
    let b: Vec<f64> = (0..inner * cols).map(|_| rng.uniform(2.0)).collect();
    (a, b)
}

fn gen_matmul_stability(n: usize, scale: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let a: Vec<f64> = (0..n * n).map(|_| rng.uniform(1.0) * scale).collect();
    let b: Vec<f64> = (0..n * n).map(|_| rng.uniform(1.0) * scale).collect();
    (a, b)
}

fn gen_matmul_negative(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let a: Vec<f64> = (0..n * n).map(|_| -(rng.next_f64() + 0.01) * 3.0).collect();
    let b: Vec<f64> = (0..n * n).map(|_| -(rng.next_f64() + 0.01) * 3.0).collect();
    (a, b)
}

fn gen_matmul_mixed(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let a: Vec<f64> = (0..n * n).map(|_| rng.uniform(5.0)).collect();
    let b: Vec<f64> = (0..n * n).map(|_| rng.uniform(5.0)).collect();
    (a, b)
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2).find(|w| w[0] == flag).and_then(|w| w[1].parse().ok())
}
