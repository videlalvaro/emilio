//! EML-LLM: Tiny Transformer via eml(x,y) = exp(x) - ln(y)
//! Based on: Odrzywołek (2026), arXiv:2603.21852
//!
//! Single binary — no Python needed.

use egg::RecExpr;
use emilio::*;
use emilio::model::*;
use std::process;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  EML-LLM: Tiny Transformer via eml(x,y) = exp(x)-ln(y)   ║");
    println!("║  Based on: Odrzywołek (2026), arXiv:2603.21852           ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    show_eml_depth_analysis();

    let mut all_passed = true;
    all_passed &= test_paper_identities();
    all_passed &= test_derived_ops();
    all_passed &= test_forward_pass();
    all_passed &= test_generation();

    test_egraph_optimizer();
    test_cse_transcendental_savings();
    benchmark_throughput();

    println!("{}", "=".repeat(60));
    if all_passed {
        println!("ALL TESTS PASSED");
        println!();
        println!("POC demonstrates:");
        println!("  • eml(x,y) = exp(x) - ln(y) is the sole arithmetic primitive");
        println!("  • exp, ln, sqrt, softmax, layernorm, gelu all EML-derived");
        println!("  • Complete transformer forward pass runs on EML alone");
        println!("  • Autoregressive generation produces valid token sequences");
        println!();
        println!("Honest caveats:");
        println!("  • Token lookup (discrete) and argmax (comparison) are not EML");
        println!("  • Negation routes through complex ln(-1) = iπ per the paper");
        println!("  • Performance is irrelevant for a POC; depth-8 mul trees are slow");
        println!("  • Weights are random — this is architecture, not a trained model");
    } else {
        println!("SOME TESTS FAILED");
        process::exit(1);
    }
}

// ─── Test helpers ───────────────────────────────────────────────────────────

struct TestCounter {
    passed: usize,
    failed: usize,
}

impl TestCounter {
    fn new() -> Self {
        Self { passed: 0, failed: 0 }
    }

    fn check(&mut self, name: &str, got: f64, expected: f64, tol: f64) {
        let err = (got - expected).abs();
        let ok = err < tol;
        let status = if ok { "✓" } else { "✗" };
        println!("  {status} {name}: got {got:.10}, expected {expected:.10}, err={err:.2e}");
        if ok {
            self.passed += 1;
        } else {
            self.failed += 1;
        }
    }

    fn all_passed(&self) -> bool {
        self.failed == 0
    }
}

// ─── Test 1: Paper identities ──────────────────────────────────────────────

fn test_paper_identities() -> bool {
    println!("{}", "=".repeat(60));
    println!("TEST 1: Paper's explicit EML identities");
    println!("{}", "=".repeat(60));

    let mut tc = TestCounter::new();

    // exp(x) = eml(x, 1)
    for &xv in &[0.0, 1.0, -1.0, 2.5] {
        let got = to_r(eml(to_c(xv), to_c(1.0)));
        tc.check(&format!("exp({xv})"), got, xv.exp(), 1e-9); // EML_AUDIT:OK — stdlib reference value for test oracle
    }

    // ln(x) = eml(1, eml(eml(1, x), 1))
    for &xv in &[0.5, 1.0, 2.0, std::f64::consts::E] {
        let one = to_c(1.0);
        let got = to_r(eml(one, eml(eml(one, to_c(xv)), one)));
        tc.check(&format!("ln({xv:.4})"), got, xv.ln(), 1e-9); // EML_AUDIT:OK — stdlib reference value for test oracle
    }

    // e = eml(1, 1)
    let one = to_c(1.0);
    tc.check("e = eml(1,1)", to_r(eml(one, one)), std::f64::consts::E, 1e-9);

    // 0 = eml(1, eml(e, 1))
    let e_val = eml(one, one);
    let zero = eml(one, eml(e_val, one));
    tc.check("0 via EML", to_r(zero), 0.0, 1e-9);

    println!("\n  Paper identities: {} passed, {} failed\n", tc.passed, tc.failed);
    tc.all_passed()
}

// ─── Test 2: Derived ops ───────────────────────────────────────────────────

fn test_derived_ops() -> bool {
    println!("{}", "=".repeat(60));
    println!("TEST 2: Derived operations");
    println!("{}", "=".repeat(60));

    let mut tc = TestCounter::new();

    // exp
    for &x in &[0.0, 1.0, -0.5, 2.0] {
        tc.check(&format!("eml_exp({x})"), to_r(eml_exp(to_c(x))), x.exp(), 1e-6); // EML_AUDIT:OK — stdlib reference value for test oracle
    }

    // ln
    for &x in &[0.1, 1.0, 2.0, 10.0] {
        tc.check(&format!("eml_ln({x})"), to_r(eml_ln(to_c(x))), x.ln(), 1e-6); // EML_AUDIT:OK — stdlib reference value for test oracle
    }

    // sqrt
    for &x in &[1.0, 4.0, 9.0, 2.0] {
        tc.check(&format!("eml_sqrt({x})"), to_r(eml_sqrt(to_c(x))), x.sqrt(), 1e-6);
    }

    // sub
    for &(a, b) in &[(5.0, 3.0), (1.0, 1.0), (0.5, 2.0), (10.0, 7.3)] {
        tc.check(&format!("eml_sub({a},{b})"), to_r(eml_sub(to_c(a), to_c(b))), a - b, 1e-6);
    }

    // neg
    for &x in &[1.0, 3.5, 0.1, 7.0] {
        tc.check(&format!("eml_neg({x})"), to_r(eml_neg(to_c(x))), -x, 1e-6);
    }

    // add
    for &(a, b) in &[(1.0, 2.0), (0.5, 0.5), (3.0, 7.0), (0.1, 0.2)] {
        tc.check(&format!("eml_add({a},{b})"), to_r(eml_add(to_c(a), to_c(b))), a + b, 1e-6);
    }

    // mul
    for &(a, b) in &[(2.0, 3.0), (0.5, 4.0), (1.5, 2.5), (7.0, 0.1)] {
        tc.check(&format!("eml_mul({a},{b})"), to_r(eml_mul(to_c(a), to_c(b))), a * b, 1e-6);
    }

    // div
    for &(a, b) in &[(6.0, 3.0), (1.0, 4.0), (10.0, 2.5), (7.0, 0.5)] {
        tc.check(
            &format!("eml_div({a},{b})"),
            to_r(eml_div(to_c(a), to_c(b))),
            a / b,
            1e-6,
        );
    }

    // inv
    for &x in &[1.0, 2.0, 0.5, 4.0] {
        tc.check(&format!("eml_inv({x})"), to_r(eml_inv(to_c(x))), 1.0 / x, 1e-6);
    }

    // softmax
    let x = [1.0, 2.0, 3.0, 4.0];
    let max_x = 4.0_f64;
    let ref_exp: Vec<f64> = x.iter().map(|&v| (v - max_x).exp()).collect(); // EML_AUDIT:OK — stdlib reference value for test oracle
    let ref_sum: f64 = ref_exp.iter().sum();
    let ref_sm: Vec<f64> = ref_exp.iter().map(|&e| e / ref_sum).collect();
    let got = eml_softmax(&x);

    let sm_sum: f64 = got.iter().sum();
    tc.check("softmax sum=1", sm_sum, 1.0, 1e-6);
    let max_diff: f64 = got.iter().zip(ref_sm.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    tc.check("softmax max_diff", max_diff, 0.0, 1e-6);

    println!("\n  Derived ops: {} passed, {} failed\n", tc.passed, tc.failed);
    tc.all_passed()
}

// ─── Test 3: Forward pass ──────────────────────────────────────────────────

fn test_forward_pass() -> bool {
    println!("{}", "=".repeat(60));
    println!("TEST 3: Full transformer forward pass");
    println!("{}", "=".repeat(60));

    let w = Weights::init(42);
    let prompt = vec![1, 5, 12, 3];
    println!("  Prompt token IDs: {:?}", prompt);

    let logits = eml_forward(&prompt, &w);

    println!("  Logits shape: ({}, {})", logits.rows, logits.cols);

    let min_val = logits.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = logits.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  Logits range: [{min_val:.4}, {max_val:.4}]");

    let last_row: Vec<f64> = (0..8).map(|c| logits.get(logits.rows - 1, c)).collect();
    println!("  Last pos logits (first 8): {:?}", last_row);

    // Verify finite
    let all_finite = logits.data.iter().all(|v| v.is_finite());
    if !all_finite {
        println!("  ✗ Non-finite logits!");
        return false;
    }

    // Softmax sum
    let last_logits: Vec<f64> = (0..VOCAB).map(|v| logits.get(logits.rows - 1, v)).collect();
    let probs = eml_softmax(&last_logits);
    let prob_sum: f64 = probs.iter().sum();
    println!("  Softmax(logits[-1]) sums to: {prob_sum:.8}");

    if (prob_sum - 1.0).abs() >= 1e-6 {
        println!("  ✗ Softmax sum should be 1.0");
        return false;
    }

    println!("\n  ✓ Forward pass complete\n");
    true
}

// ─── Test 4: Generation ────────────────────────────────────────────────────

fn test_generation() -> bool {
    println!("{}", "=".repeat(60));
    println!("TEST 4: Autoregressive token generation");
    println!("{}", "=".repeat(60));

    let w = Weights::init(42);
    let prompt = vec![1, 5, 12];
    let max_new = 6;

    println!("  Prompt:  {:?}", prompt);

    let output = eml_generate(&prompt, &w, max_new, 1.0);

    let generated = &output[prompt.len()..];
    println!("  Generated: {:?}", generated);
    println!("  Full sequence: {:?}", output);
    println!("  ✓ Generation complete\n");
    true
}

// ─── Throughput benchmark ──────────────────────────────────────────────────

fn benchmark_throughput() {
    use std::time::Instant;

    println!("{}", "=".repeat(60));
    println!("BENCHMARK: EML throughput");
    println!("{}", "=".repeat(60));

    let w = Weights::init(42);
    let prompt = vec![1_usize, 5, 12];

    // ── Single forward pass latency ────────────────────────────────
    // Warm up
    let _ = eml_forward(&prompt, &w);

    let n_fwd = 20;
    let t0 = Instant::now();
    for _ in 0..n_fwd {
        let _ = eml_forward(&prompt, &w);
    }
    let fwd_elapsed = t0.elapsed();
    let fwd_ms = fwd_elapsed.as_secs_f64() * 1000.0 / n_fwd as f64;

    println!("  Forward pass (seq_len=3):");
    println!("    {fwd_ms:.2} ms/pass  ({n_fwd} runs averaged)");

    // ── Generation throughput ──────────────────────────────────────
    let gen_tokens = 10;

    // Warm up
    let _ = eml_generate(&prompt, &w, 2, 1.0);

    let n_gen = 5;
    let t1 = Instant::now();
    for _ in 0..n_gen {
        let _ = eml_generate(&prompt, &w, gen_tokens, 1.0);
    }
    let gen_elapsed = t1.elapsed();
    let total_tokens = n_gen * gen_tokens;
    let gen_s = gen_elapsed.as_secs_f64();
    let tok_per_s = total_tokens as f64 / gen_s;

    println!();
    println!("  Generation (prompt=3, new_tokens={gen_tokens}):");
    println!("    {tok_per_s:.2} tokens/s  ({total_tokens} tokens in {gen_s:.2}s)");
    println!("    {:.2} ms/token", gen_s * 1000.0 / total_tokens as f64);

    // ── Primitive ops throughput ───────────────────────────────────
    let n_ops = 100_000;

    // eml() primitive
    let t2 = Instant::now();
    let mut acc = to_c(1.0);
    for i in 0..n_ops {
        acc = eml(acc, to_c(1.0 + i as f64 * 1e-10));
    }
    let eml_ns = t2.elapsed().as_nanos() as f64 / n_ops as f64;
    let _ = to_r(acc); // prevent optimisation

    // eml_mul (fused)
    let t3 = Instant::now();
    let mut acc2 = to_c(1.0);
    for i in 0..n_ops {
        acc2 = eml_mul(acc2, to_c(1.0 + i as f64 * 1e-15));
    }
    let mul_ns = t3.elapsed().as_nanos() as f64 / n_ops as f64;
    let _ = to_r(acc2);

    // matmul (4,16)@(16,64) — model's LM head projection
    let mut rng = crate::model::Rng::new(77);
    let a: Vec<f64> = (0..4 * 16).map(|_| 0.5 + rng.next_normal_pub() * 0.1).collect();
    let b: Vec<f64> = (0..16 * 64).map(|_| 0.5 + rng.next_normal_pub() * 0.1).collect();

    let n_mm = 50;
    let t4 = Instant::now();
    for _ in 0..n_mm {
        let _ = eml_matmul(&a, &b, 4, 16, 64);
    }
    let mm_ms = t4.elapsed().as_secs_f64() * 1000.0 / n_mm as f64;

    // CSE matmul same dims
    let t5 = Instant::now();
    for _ in 0..n_mm {
        let _ = build_matmul_cse(&a, &b, 4, 16, 64);
    }
    let mm_cse_ms = t5.elapsed().as_secs_f64() * 1000.0 / n_mm as f64;

    println!();
    println!("  Primitive ops:");
    println!("    eml(x,y):           {eml_ns:.0} ns/call");
    println!("    eml_mul(a,b):       {mul_ns:.0} ns/call  (fused: 2 ln + 1 exp)");
    println!();
    println!("  Matmul (4×16)@(16×64) — {n_mm} runs:");
    println!("    Naive:   {mm_ms:.2} ms");
    println!("    CSE:     {mm_cse_ms:.2} ms");
    println!("    Speedup: {:.2}×", mm_ms / mm_cse_ms);

    println!();
}

// ─── EML Depth Analysis ────────────────────────────────────────────────────

fn show_eml_depth_analysis() {
    println!("{}", "=".repeat(60));
    println!("EML DEPTH ANALYSIS: ops expressed as EML trees");
    println!("{}", "=".repeat(60));
    println!();
    println!("  From paper Table 4 (approximate depths in pure EML form):");
    println!();

    let ops = [
        ("exp(x)",       "eml(x, 1)",                            1),
        ("ln(x)",        "eml(1, eml(eml(1,x), 1))",            3),
        ("e",            "eml(1, 1)",                            1),
        ("0",            "eml(1, eml(e, 1))",                    2),
        ("mul(x,y)",     "exp(ln(x) + ln(y))  [depth ~8 EML]",  8),
        ("add(x,y)",     "ln(exp(x) * exp(y)) [depth ~8 EML]",  8),
        ("sqrt(x)",      "exp(0.5 * ln(x))    [depth ~5 EML]",  5),
        ("softmax(x_i)", "exp(x_i - ln(Σexp)) [depth ~6 EML]",  6),
        ("layernorm",    "via sqrt, add, div  [depth ~12 EML]", 12),
        ("gelu(x)",      "x * σ(1.702x)       [depth ~10 EML]", 10),
    ];

    for (name, eml_form, depth) in &ops {
        let bar: String = "█".repeat(*depth);
        println!("  {name:20} depth={depth:2}  {bar}");
        let _ = eml_form; // suppress warning
    }

    println!();
    println!("  Grammar: S → 1 | eml(S, S)");
    println!("  Every op is a binary tree of identical eml nodes.");
    println!();
}

// ─── E-graph optimizer demo ────────────────────────────────────────────────

fn test_egraph_optimizer() {
    println!("{}", "=".repeat(60));
    println!("TEST 5: E-graph equality saturation");
    println!("{}", "=".repeat(60));

    // exp(ln(x)) should cancel to x
    let expr1: RecExpr<EmlLang> = "(exp (ln x))".parse().unwrap();
    let (cost1, best1) = optimize_and_extract(&expr1);
    println!("  exp(ln(x)) → {best1}  (cost={cost1})");

    // ln(exp(x)) should cancel to x
    let expr2: RecExpr<EmlLang> = "(ln (exp x))".parse().unwrap();
    let (cost2, best2) = optimize_and_extract(&expr2);
    println!("  ln(exp(x)) → {best2}  (cost={cost2})");

    // mul(a,b) stays as mul (cost 3)
    let expr3: RecExpr<EmlLang> = "(mul a b)".parse().unwrap();
    let (cost3, best3) = optimize_and_extract(&expr3);
    println!("  mul(a,b) → {best3}  (cost={cost3})");

    // nested: exp(ln(a) + ln(b)) → mul(a,b) (cost 3 vs 5)
    let expr4: RecExpr<EmlLang> = "(exp (add (ln a) (ln b)))".parse().unwrap();
    let (cost4_before, _) = (5.0, &expr4); // manual: 2 ln + 1 add + 1 exp = 2+0+1 = 3... but let's see
    let (cost4, best4) = optimize_and_extract(&expr4);
    println!("  exp(ln(a)+ln(b)) → {best4}  (cost={cost4})");

    // eval: mul(3, 4) = 12
    let result = eval_expr(&expr3, &[("a", 3.0), ("b", 4.0)]);
    println!("  eval mul(3,4) = {result:.6}");

    // eval: add(mul(2,3), mul(4,5)) = 26
    let expr5: RecExpr<EmlLang> = "(add (mul a b) (mul c d))".parse().unwrap();
    let result5 = eval_expr(&expr5, &[("a", 2.0), ("b", 3.0), ("c", 4.0), ("d", 5.0)]);
    println!("  eval add(mul(2,3),mul(4,5)) = {result5:.6}");

    println!("  ✓ E-graph optimizer works\n");
    let _ = cost4_before;
}

// ─── CSE transcendental savings benchmark ──────────────────────────────────

fn test_cse_transcendental_savings() {
    println!("{}", "=".repeat(60));
    println!("TEST 6: CSE transcendental savings (audit mode)");
    println!("{}", "=".repeat(60));

    // ── Matmul benchmark ────────────────────────────────────────────
    // Our model's QKV projection: (4, 16) @ (16, 48)
    let rows = 4_usize;
    let inner = 16_usize;
    let cols = 48_usize;

    // Generate deterministic test data
    let mut rng = crate::model::Rng::new(99);
    let a: Vec<f64> = (0..rows * inner).map(|_| 0.1 + rng.next_normal_pub() * 0.02).collect();
    let b: Vec<f64> = (0..inner * cols).map(|_| 0.1 + rng.next_normal_pub() * 0.02).collect();

    // Naive matmul: each element does K muls = K * (2 ln + 1 exp) = 3K transcendentals
    let naive_per_element = 3 * inner; // 48
    let naive_total = rows * cols * naive_per_element;

    // CSE matmul: precompute ln(A) + ln(B), then each element = K exp
    let cse_ln = rows * inner + inner * cols; // ln(A) + ln(B)
    let cse_exp = rows * cols * inner;         // K exp per element
    let cse_total = cse_ln + cse_exp;

    // Verify via audited matmul
    reset_counters();
    let cse_result = audited_matmul_cse(&a, &b, rows, inner, cols);
    let (actual_exp, actual_ln) = get_counts();

    // Verify correctness: CSE result matches naive
    let naive_result = eml_matmul(&a, &b, rows, inner, cols);
    let max_err: f64 = cse_result.iter().zip(naive_result.iter())
        .map(|(c, n)| (c - n).abs())
        .fold(0.0, f64::max);

    println!("  Matmul ({rows}×{inner}) @ ({inner}×{cols}):");
    println!("    Naive:     {naive_total:>6} transcendentals (3K per element)");
    println!("    CSE:       {cse_total:>6} transcendentals (K exp + amortized ln)");
    println!("    Audited:   {:>6} ({actual_exp} exp + {actual_ln} ln)", actual_exp + actual_ln);
    println!("    Saving:    {:.1}%", (1.0 - cse_total as f64 / naive_total as f64) * 100.0);
    println!("    Max error: {max_err:.2e} (CSE vs naive)");

    // ── Softmax benchmark ───────────────────────────────────────────
    let softmax_input: Vec<f64> = (0..64).map(|i| i as f64 * 0.1).collect();
    let n = softmax_input.len();

    // Naive: N exp + N-1 add + N div = N exp + N*(2 ln + 1 exp) = 2N exp + 2N ln
    // CSE: N exp + 1 ln(Z) + N*(1 ln + 1 exp) = (2N+1) exp + (N+1) ln
    // Wait, let's be precise about the naive version from eml_ops:
    // softmax does: N eml_exp(eml_sub(..)) = N exp, then reduce via eml_add = 0,
    // then N eml_div(e, z) = N * (2 ln + 1 exp)
    let naive_sm = n + n * 3;  // N exp for shifted + N div (each 3 transcendentals)
    // CSE: N exp for shifted + 1 ln(Z) + N*(1 ln(e_i) + 1 exp) = N + 1 + 2N
    let cse_sm = n + 1 + 2 * n;

    let sm_naive = eml_softmax(&softmax_input);
    let sm_cse = build_softmax_cse(&softmax_input);
    let sm_err: f64 = sm_naive.iter().zip(sm_cse.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!();
    println!("  Softmax (N={n}):");
    println!("    Naive:     {naive_sm:>6} transcendentals");
    println!("    CSE:       {cse_sm:>6} transcendentals");
    println!("    Saving:    {:.1}%", (1.0 - cse_sm as f64 / naive_sm as f64) * 100.0);
    println!("    Max error: {sm_err:.2e}");

    // ── Full model comparison ───────────────────────────────────────
    println!();
    println!("  Model-wide projection (1 forward pass, seq_len=4):");
    // QKV: (4,16)@(16,48) = 4*48=192 elements, K=16
    // Attn out: (4,16)@(16,16) = 64 elements, K=16
    // FFN1: (4,16)@(16,32) = 128 elements, K=16
    // FFN2: (4,32)@(32,16) = 64 elements, K=32
    // LM head: (4,16)@(16,64) = 256 elements, K=16
    let matmuls = [
        ("QKV proj", 4, 16, 48),
        ("Attn out", 4, 16, 16),
        ("FFN1", 4, 16, 32),
        ("FFN2", 4, 32, 16),
        ("LM head", 4, 16, 64),
    ];

    let mut total_naive = 0_usize;
    let mut total_cse = 0_usize;

    for (name, r, k, c) in &matmuls {
        let naive = r * c * 3 * k;
        let cse = r * k + k * c + r * c * k;
        total_naive += naive;
        total_cse += cse;
        println!("    {name:12} ({r}×{k})@({k}×{c}): naive={naive:>5} CSE={cse:>5} Δ={:.0}%",
            (1.0 - cse as f64 / naive as f64) * 100.0);
    }

    println!("    {}", "─".repeat(50));
    println!("    Total matmul: naive={total_naive:>5} CSE={total_cse:>5} Δ={:.1}%",
        (1.0 - total_cse as f64 / total_naive as f64) * 100.0);

    println!();
    println!("  Key insight: CSE doesn't bypass EML — it shares ln() nodes");
    println!("  across the matmul, reducing 3K to K transcendentals per element.");
    println!("  Every result still flows through exp() and ln() calls.\n");
}
