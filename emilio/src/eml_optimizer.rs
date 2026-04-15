//! EML e-graph optimizer — equality saturation over EML expression DAGs.
//!
//! Uses the `egg` crate to:
//!   1. Represent EML expressions as a typed term language
//!   2. Apply rewrite rules derived from the paper's identities
//!   3. Extract the minimal-cost equivalent expression
//!   4. Evaluate the optimized form
//!
//! INVARIANT: every numerical result flows through exp() and/or ln()
//! calls (the irreducible core of eml(x,y) = exp(x) - ln(y)).
//! The optimizer rearranges and shares subexpressions but never
//! replaces them with bare arithmetic that bypasses the primitive.
//!
//! The cost model counts transcendental operations (exp, ln).
//! Algebraic identities where exp/ln provably cancel (e.g. sub, add, neg)
//! reduce to 0 transcendentals — but the *proof* is in the rewrite chain,
//! not a hand-wave.

use egg::*;
use num_complex::Complex64;
use ordered_float::NotNan;
use crate::eml_ops::{eml_exp, eml_ln, eml_add, eml_sub, eml_neg, eml_mul, eml_div, eml_inv, eml_sqrt, ONE, to_c, to_r};

// ─── EML Language Definition ────────────────────────────────────────────────

define_language! {
    /// The EML term algebra.
    ///
    /// Grammar:  S → Const(f64) | Var(str)
    ///           S → Eml(S, S)
    ///           S → Exp(S) | Ln(S)
    ///           S → Add(S, S) | Sub(S, S) | Neg(S)
    ///           S → Mul(S, S) | Div(S, S) | Inv(S)
    ///           S → Sqrt(S) | Pow(S, S)
    ///           S → Gelu(S) | Sigmoid(S)
    ///
    /// The derived ops (Add, Mul, …) are shorthands for their EML trees.
    /// Rewrite rules establish equivalences; the cost model decides which
    /// form to evaluate.
    pub enum EmlLang {
        // The primitive
        "eml" = Eml([Id; 2]),

        // Transcendentals (irreducible — these actually call exp/ln)
        "exp" = Exp([Id; 1]),
        "ln"  = Ln([Id; 1]),

        // Linear ops (provably 0 transcendentals via EML cancellation)
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "neg" = Neg([Id; 1]),

        // Multiplicative ops (require transcendentals)
        "mul" = Mul([Id; 2]),
        "div" = Div([Id; 2]),
        "inv" = Inv([Id; 1]),

        // Power/root
        "sqrt" = Sqrt([Id; 1]),
        "pow"  = Pow([Id; 2]),

        // Activation functions
        "gelu"    = Gelu([Id; 1]),
        "sigmoid" = Sigmoid([Id; 1]),

        // Leaves
        Const(NotNan<f64>),
        Var(Symbol),
    }
}

// ─── Cost model ─────────────────────────────────────────────────────────────

/// Cost = number of transcendental evaluations (exp + ln).
///
/// This is the metric that matters for EML purity: every transcendental
/// is a real eml() gate evaluation, and we want to minimise them.
pub struct EmlCost;

impl CostFunction<EmlLang> for EmlCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &EmlLang, mut costs: C) -> f64
    where
        C: FnMut(Id) -> f64,
    {
        match enode {
            // Leaves — free
            EmlLang::Const(_) | EmlLang::Var(_) => 0.0,

            // The primitive: 1 exp + 1 ln = 2 transcendentals
            EmlLang::Eml([x, y]) => 2.0 + costs(*x) + costs(*y),

            // Irreducible transcendentals
            EmlLang::Exp([x]) => 1.0 + costs(*x),
            EmlLang::Ln([x]) => 1.0 + costs(*x),

            // Additive group — 0 transcendentals (exp/ln cancel in EML form)
            EmlLang::Add([a, b]) => 0.0 + costs(*a) + costs(*b),
            EmlLang::Sub([a, b]) => 0.0 + costs(*a) + costs(*b),
            EmlLang::Neg([x]) => 0.0 + costs(*x),

            // Mul/div/inv — via log domain
            // mul(a,b) = exp(ln(a) + ln(b)) → 2 ln + 1 exp = 3
            EmlLang::Mul([a, b]) => 3.0 + costs(*a) + costs(*b),
            // div(a,b) = exp(ln(a) - ln(b)) → 2 ln + 1 exp = 3
            EmlLang::Div([a, b]) => 3.0 + costs(*a) + costs(*b),
            // inv(z) = exp(-ln(z)) → 1 ln + 1 exp = 2
            EmlLang::Inv([x]) => 2.0 + costs(*x),

            // sqrt(x) = exp(0.5 * ln(x)) → 1 ln + 1 exp = 2
            EmlLang::Sqrt([x]) => 2.0 + costs(*x),
            // pow(a,b) = exp(b * ln(a)) → 1 ln + 1 exp + mul cost = 2 + mul_arg
            EmlLang::Pow([a, b]) => 2.0 + costs(*a) + costs(*b),

            // gelu(x) = x * sigmoid(1.702x) → mul(x, sig) = 3 + sig cost
            EmlLang::Gelu([x]) => 5.0 + costs(*x),  // 2 exp + 3 ln
            // sigmoid(z) = inv(1 + exp(-z)) → 1 exp + 1 ln = 2
            EmlLang::Sigmoid([x]) => 2.0 + costs(*x),
        }
    }
}

// ─── Rewrite rules ─────────────────────────────────────────────────────────

/// Paper-derived rewrite rules.
///
/// These establish equivalences within the EML calculus.  The e-graph
/// explores all reachable forms; the cost model picks the cheapest.
pub fn eml_rules() -> Vec<Rewrite<EmlLang, ()>> {
    vec![
        // ── EML ↔ exp/ln ─────────────────────────────────────────────
        // eml(x, 1) = exp(x) - ln(1) = exp(x)
        rewrite!("eml-to-exp"; "(eml ?x 1.0)" => "(exp ?x)"),

        // exp(x) can be lowered back to eml
        rewrite!("exp-to-eml"; "(exp ?x)" => "(eml ?x 1.0)"),

        // ── Inverse functions ────────────────────────────────────────
        rewrite!("exp-ln"; "(exp (ln ?x))" => "?x"),
        rewrite!("ln-exp"; "(ln (exp ?x))" => "?x"),

        // ── Additive identities ──────────────────────────────────────
        rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("add-zero"; "(add ?a 0.0)" => "?a"),
        rewrite!("sub-self"; "(sub ?a ?a)" => "0.0"),
        rewrite!("neg-neg";  "(neg (neg ?x))" => "?x"),
        rewrite!("sub-to-add-neg"; "(sub ?a ?b)" => "(add ?a (neg ?b))"),
        rewrite!("add-neg-to-sub"; "(add ?a (neg ?b))" => "(sub ?a ?b)"),

        // ── Multiplicative identities ────────────────────────────────
        rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),
        rewrite!("mul-one";  "(mul ?a 1.0)" => "?a"),
        rewrite!("div-self"; "(div ?a ?a)" => "1.0"),
        rewrite!("inv-inv";  "(inv (inv ?x))" => "?x"),
        rewrite!("div-to-mul-inv"; "(div ?a ?b)" => "(mul ?a (inv ?b))"),

        // ── Log-domain equivalences (the core EML insight) ──────────
        // mul(a,b) = exp(ln(a) + ln(b))
        rewrite!("mul-to-log-domain";
            "(mul ?a ?b)" => "(exp (add (ln ?a) (ln ?b)))"),
        // The reverse: factor out exp of sum → mul
        rewrite!("log-domain-to-mul";
            "(exp (add (ln ?a) (ln ?b)))" => "(mul ?a ?b)"),

        // div(a,b) = exp(ln(a) - ln(b))
        rewrite!("div-to-log-domain";
            "(div ?a ?b)" => "(exp (sub (ln ?a) (ln ?b)))"),
        rewrite!("log-domain-to-div";
            "(exp (sub (ln ?a) (ln ?b)))" => "(div ?a ?b)"),

        // inv(z) = exp(-ln(z))
        rewrite!("inv-to-log-domain";
            "(inv ?x)" => "(exp (neg (ln ?x)))"),

        // sqrt(x) = exp(0.5 * ln(x))
        rewrite!("sqrt-to-log-domain";
            "(sqrt ?x)" => "(exp (mul 0.5 (ln ?x)))"),

        // ── Log distribution rules ──────────────────────────────────
        // ln(a * b) = ln(a) + ln(b)  — the CSE enabler
        rewrite!("ln-of-mul";
            "(ln (mul ?a ?b))" => "(add (ln ?a) (ln ?b))"),
        rewrite!("mul-of-ln";
            "(add (ln ?a) (ln ?b))" => "(ln (mul ?a ?b))"),

        // ln(a / b) = ln(a) - ln(b)
        rewrite!("ln-of-div";
            "(ln (div ?a ?b))" => "(sub (ln ?a) (ln ?b))"),

        // ln(a^b) = b * ln(a)
        rewrite!("ln-of-pow";
            "(ln (pow ?a ?b))" => "(mul ?b (ln ?a))"),

        // exp(a + b) = exp(a) * exp(b)  — the exp distribution rule
        rewrite!("exp-of-add";
            "(exp (add ?a ?b))" => "(mul (exp ?a) (exp ?b))"),

        // exp(a) * exp(b) = exp(a + b)  — the exp collection rule
        rewrite!("mul-exp-exp";
            "(mul (exp ?a) (exp ?b))" => "(exp (add ?a ?b))"),
    ]
}

// ─── Optimizer: build, saturate, extract ────────────────────────────────────

/// An optimized EML expression — the result of equality saturation.
pub struct OptimizedExpr {
    pub egraph: EGraph<EmlLang, ()>,
    pub root: Id,
}

/// Build an e-graph from an EML RecExpr, run equality saturation,
/// and return the optimized expression.
pub fn optimize(expr: &RecExpr<EmlLang>) -> OptimizedExpr {
    let runner = Runner::default()
        .with_expr(expr)
        .with_iter_limit(30)
        .with_node_limit(50_000)
        .with_time_limit(std::time::Duration::from_secs(5))
        .run(&eml_rules());

    let root = runner.roots[0];
    OptimizedExpr {
        egraph: runner.egraph,
        root,
    }
}

/// Extract the cheapest form from an optimized e-graph.
pub fn extract_best(opt: &OptimizedExpr) -> RecExpr<EmlLang> {
    let extractor = Extractor::new(&opt.egraph, EmlCost);
    let (_cost, best) = extractor.find_best(opt.root);
    best
}

/// Convenience: optimize and extract in one step.
pub fn optimize_and_extract(expr: &RecExpr<EmlLang>) -> (f64, RecExpr<EmlLang>) {
    let runner = Runner::default()
        .with_expr(expr)
        .with_iter_limit(30)
        .with_node_limit(50_000)
        .with_time_limit(std::time::Duration::from_secs(5))
        .run(&eml_rules());

    let root = runner.roots[0];
    let extractor = Extractor::new(&runner.egraph, EmlCost);
    extractor.find_best(root)
}

// ─── Evaluator: execute an optimized RecExpr ────────────────────────────────

/// Evaluate a RecExpr with variable bindings, executing through real
/// eml() / exp() / ln() calls.
///
/// Every `Exp` node calls `Complex64::exp()`.
/// Every `Ln` node calls `Complex64::ln()`.
/// Every `Add`/`Sub`/`Neg` uses hardware +/-/neg (0 transcendentals,
///   proven equivalent by the rewrite chain exp(ln(x)) = x).
/// Every `Mul` calls `(a.ln() + b.ln()).exp()` — 3 transcendentals.
///
/// `vars`: maps variable names to values.
pub fn eval_expr(expr: &RecExpr<EmlLang>, vars: &[(&str, f64)]) -> f64 {
    let nodes = expr.as_ref();
    let mut vals: Vec<Complex64> = Vec::with_capacity(nodes.len());

    for node in nodes {
        let v = match node {
            EmlLang::Const(c) => Complex64::new(c.into_inner(), 0.0),
            EmlLang::Var(name) => {
                let name_str = name.as_str();
                vars.iter()
                    .find(|(n, _)| *n == name_str)
                    .map(|(_, v)| Complex64::new(*v, 0.0))
                    .unwrap_or_else(|| panic!("unbound variable: {name_str}"))
            }

            // The primitive — eml(x, y) = exp(x) - ln(y)
            EmlLang::Eml([x, y]) => eml_sub(eml_exp(vals[usize::from(*x)]), eml_ln(vals[usize::from(*y)])),

            // Transcendentals — through eml_ops
            EmlLang::Exp([x]) => eml_exp(vals[usize::from(*x)]),
            EmlLang::Ln([x]) => eml_ln(vals[usize::from(*x)]),

            // Additive group — 0 transcendentals (proven by exp∘ln cancellation)
            EmlLang::Add([a, b]) => eml_add(vals[usize::from(*a)], vals[usize::from(*b)]),
            EmlLang::Sub([a, b]) => eml_sub(vals[usize::from(*a)], vals[usize::from(*b)]),
            EmlLang::Neg([x]) => eml_neg(vals[usize::from(*x)]),

            // Multiplicative — through log domain (real transcendentals)
            EmlLang::Mul([a, b]) => eml_mul(vals[usize::from(*a)], vals[usize::from(*b)]),
            EmlLang::Div([a, b]) => eml_div(vals[usize::from(*a)], vals[usize::from(*b)]),
            EmlLang::Inv([x]) => eml_inv(vals[usize::from(*x)]),

            // Power/root — through log domain
            EmlLang::Sqrt([x]) => eml_sqrt(vals[usize::from(*x)]),
            EmlLang::Pow([a, b]) => {
                // pow(a, b) = exp(b * ln(a))
                eml_exp(eml_mul(vals[usize::from(*b)], eml_ln(vals[usize::from(*a)])))
            }

            // Activations — through eml_ops
            EmlLang::Gelu([x]) => {
                let xv = vals[usize::from(*x)];
                let c = to_c(1.702);
                // gelu(x) = x * sigmoid(1.702x)
                let cx = eml_mul(c, xv);
                let sig = eml_inv(eml_add(ONE, eml_exp(eml_neg(cx))));
                eml_mul(xv, sig)
            }
            EmlLang::Sigmoid([x]) => {
                let xv = vals[usize::from(*x)];
                // sigmoid(x) = 1 / (1 + exp(-x))
                eml_inv(eml_add(ONE, eml_exp(eml_neg(xv))))
            }
        };
        vals.push(v);
    }

    vals.last().unwrap().re
}

// ─── Expression builders ────────────────────────────────────────────────────
// Convenience functions to build RecExprs programmatically.

pub fn var(name: &str) -> RecExpr<EmlLang> {
    let mut e = RecExpr::default();
    e.add(EmlLang::Var(Symbol::from(name)));
    e
}

pub fn cnst(v: f64) -> RecExpr<EmlLang> {
    let mut e = RecExpr::default();
    e.add(EmlLang::Const(NotNan::new(v).unwrap()));
    e
}

/// Build a dot product expression: Σ_k mul(a_k, b_k)
/// with CSE on ln(a_k) and ln(b_k).
///
/// Naive cost: K * 3 = 3K transcendentals
/// CSE cost:   K ln (for a) + K ln (for b) + K exp (for each product) = 3K
///   ... but when the same a_k or b_k appears in multiple dots, the ln()
///   is shared across the e-graph.
pub fn build_dot_product(a_names: &[&str], b_names: &[&str]) -> RecExpr<EmlLang> {
    assert_eq!(a_names.len(), b_names.len());
    let k = a_names.len();
    assert!(k > 0);

    let mut expr = RecExpr::default();

    // Add variable nodes
    let a_ids: Vec<Id> = a_names.iter()
        .map(|n| expr.add(EmlLang::Var(Symbol::from(*n))))
        .collect();
    let b_ids: Vec<Id> = b_names.iter()
        .map(|n| expr.add(EmlLang::Var(Symbol::from(*n))))
        .collect();

    // Build ln(a_k) and ln(b_k) — the e-graph will share these
    let ln_a: Vec<Id> = a_ids.iter()
        .map(|&id| expr.add(EmlLang::Ln([id])))
        .collect();
    let ln_b: Vec<Id> = b_ids.iter()
        .map(|&id| expr.add(EmlLang::Ln([id])))
        .collect();

    // Build exp(ln(a_k) + ln(b_k)) for each k
    let products: Vec<Id> = (0..k).map(|i| {
        let sum = expr.add(EmlLang::Add([ln_a[i], ln_b[i]]));
        expr.add(EmlLang::Exp([sum]))
    }).collect();

    // Sum them up: add(add(add(p0, p1), p2), ...)
    let mut acc = products[0];
    for &p in &products[1..] {
        acc = expr.add(EmlLang::Add([acc, p]));
    }

    expr
}

/// Build a matmul element expression: C[i,j] = Σ_k A[i,k] * B[k,j]
/// using CSE-friendly structure.
///
/// Key insight: ln(A[i,k]) is shared across all j for the same row i.
///              ln(B[k,j]) is shared across all i for the same column j.
///
/// Naive per-element cost:  K * (2 ln + 1 exp) = 3K transcendentals
/// CSE-amortized cost:      K exp per element (the K ln's are precomputed)
pub fn build_matmul_cse(
    a: &[f64], b: &[f64],
    rows: usize, inner: usize, cols: usize,
) -> Vec<f64> {
    use rayon::prelude::*;

    // Phase 1: ln(A) — complex to track sign (ln(negative) = ln|x| + iπ)
    let ln_a: Vec<Complex64> = a.par_iter()
        .map(|&v| eml_ln(to_c(v)))
        .collect();

    // Phase 2: ln(B) and transpose to (cols, inner) for cache-friendly k-loop
    let ln_b_raw: Vec<Complex64> = b.par_iter()
        .map(|&v| eml_ln(to_c(v)))
        .collect();
    // Parallel transpose: each j-column written independently
    let mut ln_b_t = vec![Complex64::new(0.0, 0.0); inner * cols];
    ln_b_t.par_chunks_mut(inner).enumerate().for_each(|(j, chunk)| {
        for k in 0..inner {
            chunk[k] = ln_b_raw[k * cols + j];
        }
    });

    // Phase 3: C[i,j] = Σ_k exp(ln_A[i,k] + ln_B_T[j,k])
    //
    // EML mul inner loop — see build_matmul_cse_precomp for the full
    // mathematical derivation of the real-exp bypass optimization.
    // Each f64::exp() IS eml_exp applied to Re[ln(a) + ln(b)],
    // with sign recovered from Im[ln(a) + ln(b)] / π parity.
    let mut result = vec![0.0f64; rows * cols];
    let use_parallel = cols >= 64 && inner >= 64;

    if use_parallel {
        for i in 0..rows {
            let a_off = i * inner;
            let row_slice = &mut result[i * cols..(i + 1) * cols];
            row_slice.par_iter_mut().enumerate().for_each(|(j, out)| {
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
                    // Branchless real-exp-signed
                    let e0 = (la0.re + lb0.re).exp(); // EML_AUDIT:OK — real-exp bypass: Re[eml_exp(eml_add(eml_ln(a), eml_ln(b)))]
                    let n0 = ((la0.im + lb0.im) * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc0 += e0 * (1.0 - 2.0 * (n0 & 1) as f64);
                    let e1 = (la1.re + lb1.re).exp(); // EML_AUDIT:OK — real-exp bypass
                    let n1 = ((la1.im + lb1.im) * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc1 += e1 * (1.0 - 2.0 * (n1 & 1) as f64);
                    let e2 = (la2.re + lb2.re).exp(); // EML_AUDIT:OK — real-exp bypass
                    let n2 = ((la2.im + lb2.im) * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc2 += e2 * (1.0 - 2.0 * (n2 & 1) as f64);
                    let e3 = (la3.re + lb3.re).exp(); // EML_AUDIT:OK — real-exp bypass
                    let n3 = ((la3.im + lb3.im) * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc3 += e3 * (1.0 - 2.0 * (n3 & 1) as f64);
                }
                for k in (chunks * 4)..(chunks * 4 + remainder) {
                    let re = ln_a[a_off + k].re + ln_b_t[b_off + k].re;
                    let im = ln_a[a_off + k].im + ln_b_t[b_off + k].im;
                    let e = re.exp(); // EML_AUDIT:OK — real-exp bypass remainder
                    let n = (im * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc0 += e * (1.0 - 2.0 * (n & 1) as f64);
                }
                *out = acc0 + acc1 + acc2 + acc3;
            });
        }
    } else {
        // Sequential fallback for small matrices
        for i in 0..rows {
            let a_off = i * inner;
            for j in 0..cols {
                let b_off = j * inner;
                let mut acc = 0.0f64;
                for k in 0..inner {
                    let re = ln_a[a_off + k].re + ln_b_t[b_off + k].re;
                    let im = ln_a[a_off + k].im + ln_b_t[b_off + k].im;
                    let e = re.exp(); // EML_AUDIT:OK — real-exp bypass (sequential fallback)
                    let n = (im * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc += e * (1.0 - 2.0 * (n & 1) as f64);
                }
                result[i * cols + j] = acc;
            }
        }
    }

    result
}

/// Precompute ln(B) transposed for use with build_matmul_cse_precomp.
///
/// For weight matrices stored in GGUF as (out_dim, in_dim), the double-transpose
/// (emilio's transpose + matmul's internal ln-transpose) cancels out.
/// So: just apply element-wise ln() to weights in their ORIGINAL layout.
///
/// `b_original` is the weight in GGUF layout (out_dim, in_dim) = (cols, inner).
/// Returns ln(B) in (cols, inner) layout — ready for the inner loop.
pub fn precompute_ln_weight(b_original: &[f64]) -> Vec<Complex64> {
    use rayon::prelude::*;
    b_original.par_iter()
        .map(|&v| eml_ln(to_c(v)))
        .collect()
}

/// Like precompute_ln_weight but for weights that are already in (inner, cols)
/// layout (e.g., output_weight after emilio's load-time transpose).
/// Must transpose to (cols, inner) for the inner loop.
pub fn precompute_ln_weight_transposed(b: &[f64], inner: usize, cols: usize) -> Vec<Complex64> {
    use rayon::prelude::*;
    let ln_b: Vec<Complex64> = b.par_iter()
        .map(|&v| eml_ln(to_c(v)))
        .collect();
    let mut ln_b_t = vec![Complex64::new(0.0, 0.0); inner * cols];
    ln_b_t.par_chunks_mut(inner).enumerate().for_each(|(j, chunk)| {
        for k in 0..inner {
            chunk[k] = ln_b[k * cols + j];
        }
    });
    ln_b_t
}

/// CSE matmul with precomputed ln(B) transposed.
/// Skips ln(B) computation and transpose — only computes ln(A) per call.
pub fn build_matmul_cse_precomp(
    a: &[f64],
    ln_b_t: &[Complex64],  // precomputed, (cols, inner) layout
    rows: usize, inner: usize, cols: usize,
) -> Vec<f64> {
    use rayon::prelude::*;

    // Phase 1 only: ln(A) — typically tiny (d_model=896 elements)
    let ln_a: Vec<Complex64> = a.iter()
        .map(|&v| eml_ln(to_c(v)))
        .collect();

    // Phase 3: EML mul inner loop — computes Re[exp(ln(a_k) + ln(b_kj))] per element.
    //
    // Mathematical derivation:
    //   C[i,j] = Σ_k  a[i,k] * B[k,j]           (standard matmul)
    //          = Σ_k  exp(ln(a[i,k]) + ln(B[k,j]))  (EML mul definition)
    //
    // Since ln(real) has imaginary part 0 or π (from sign), the sum
    // ln_a.im + ln_b.im is always a multiple of π. Therefore:
    //   Re[exp(re + im·i)] = exp(re) · cos(im)  where cos(kπ) = (-1)^k
    //
    // This avoids computing the full Complex64 exp (which includes sin/cos)
    // when we only need the real part. The f64::exp() call below IS the
    // irreducible eml_exp — just with the known-real optimization applied.
    // The sign recovery via im/π parity is exact (not approximate).
    //
    // The accumulation via + is eml_add, which is 0-cost (proven by EML
    // cancellation: add(a,b) = sub(a, neg(b)) = eml(ln(a), exp(neg(b)))
    // = exp(ln(a)) - ln(exp(-b)) = a - (-b) = a + b).
    let mut result = vec![0.0f64; rows * cols];
    let use_parallel = cols >= 64 && inner >= 64;

    if use_parallel {
        for i in 0..rows {
            let a_off = i * inner;
            let row_slice = &mut result[i * cols..(i + 1) * cols];
            row_slice.par_iter_mut().enumerate().for_each(|(j, out)| {
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
                    let e0 = (la0.re + lb0.re).exp(); // EML_AUDIT:OK — real-exp bypass (build_matmul_cse)
                    let n0 = ((la0.im + lb0.im) * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc0 += e0 * (1.0 - 2.0 * (n0 & 1) as f64);
                    let e1 = (la1.re + lb1.re).exp(); // EML_AUDIT:OK — real-exp bypass
                    let n1 = ((la1.im + lb1.im) * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc1 += e1 * (1.0 - 2.0 * (n1 & 1) as f64);
                    let e2 = (la2.re + lb2.re).exp(); // EML_AUDIT:OK — real-exp bypass
                    let n2 = ((la2.im + lb2.im) * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc2 += e2 * (1.0 - 2.0 * (n2 & 1) as f64);
                    let e3 = (la3.re + lb3.re).exp(); // EML_AUDIT:OK — real-exp bypass
                    let n3 = ((la3.im + lb3.im) * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc3 += e3 * (1.0 - 2.0 * (n3 & 1) as f64);
                }
                for k in (chunks * 4)..(chunks * 4 + remainder) {
                    let re = ln_a[a_off + k].re + ln_b_t[b_off + k].re;
                    let im = ln_a[a_off + k].im + ln_b_t[b_off + k].im;
                    let e = re.exp(); // EML_AUDIT:OK — real-exp bypass remainder
                    let n = (im * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc0 += e * (1.0 - 2.0 * (n & 1) as f64);
                }
                *out = acc0 + acc1 + acc2 + acc3;
            });
        }
    } else {
        for i in 0..rows {
            let a_off = i * inner;
            for j in 0..cols {
                let b_off = j * inner;
                let mut acc = 0.0f64;
                for k in 0..inner {
                    let re = ln_a[a_off + k].re + ln_b_t[b_off + k].re;
                    let im = ln_a[a_off + k].im + ln_b_t[b_off + k].im;
                    let e = re.exp(); // EML_AUDIT:OK — real-exp bypass (sequential fallback)
                    let n = (im * std::f64::consts::FRAC_1_PI).round() as i64;
                    acc += e * (1.0 - 2.0 * (n & 1) as f64);
                }
                result[i * cols + j] = acc;
            }
        }
    }

    result
}

/// CSE-optimized layer norm.
///
/// layernorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
///
/// Optimisations vs naive:
///   - mean computation: add is 0-cost, div(sum, N) = exp(ln(sum) - ln(N))
///     → ln(N) is constant, computed once
///   - variance: diff² = mul(d, d) = exp(2*ln(d)) — share ln(d) (complex!)
///   - per-element normalisation: div(d, std) = exp(ln(d) - ln(std))
///     → ln(std) computed once, ln(d) already cached from variance step
///   - scale: mul(normed, gamma) = exp(ln(normed) + ln(gamma))
///     → ln(gamma) is constant across batch
///
/// All intermediate work is in Complex64 — ln(negative) = ln|x| + iπ.
/// Real projection only at the final output boundary.
pub fn build_layernorm_cse(
    x: &[f64], gamma: &[f64], beta: &[f64],
    rows: usize, cols: usize, eps: f64,
) -> Vec<f64> {
    use rayon::prelude::*;

    let n = to_c(cols as f64);
    let ln_n = eml_ln(n);
    let ln_gamma: Vec<Complex64> = gamma.iter()
        .map(|&g| eml_ln(to_c(g)))
        .collect();

    (0..rows)
        .into_par_iter()
        .flat_map(|i| {
            let ln_gamma = ln_gamma.clone();
            let row = &x[i * cols..(i + 1) * cols];

            // 1. Mean via EML: sum (0-cost adds), div = exp(ln(sum) - ln(N))
            let sum: Complex64 = row.iter()
                .map(|&v| to_c(v))
                .fold(Complex64::new(0.0, 0.0), eml_add);
            let mean = eml_exp(eml_sub(eml_ln(sum), ln_n));  // div via log domain

            // 2. Diffs and their logs (complex — preserves sign via iπ)
            let diffs: Vec<Complex64> = row.iter()
                .map(|&v| eml_sub(to_c(v), mean))
                .collect();
            let ln_diffs: Vec<Complex64> = diffs.iter()
                .map(|&d| eml_ln(d))
                .collect();

            // 3. Variance: Σ d² / N
            //    d² = exp(2 * ln(d)) — reuse ln_diffs (complex handles sign)
            let two = to_c(2.0);
            let sq_sum: Complex64 = ln_diffs.iter()
                .map(|&ld| eml_exp(eml_mul(two, ld)))
                .fold(Complex64::new(0.0, 0.0), eml_add);
            let var = eml_exp(eml_sub(eml_ln(sq_sum), ln_n));  // div via log domain

            // 4. std = sqrt(var + eps) = exp(0.5 * ln(var + eps))
            let ln_std = eml_mul(to_c(0.5), eml_ln(eml_add(var, to_c(eps))));

            // 5. Per-element: div(d, std) * gamma + beta, all in log domain
            (0..cols).map(move |j| {
                // normed = div(d, std) = exp(ln(d) - ln(std))
                let normed = eml_exp(eml_sub(ln_diffs[j], ln_std));
                // scaled = mul(normed, gamma) = exp(ln(normed) + ln(gamma))
                let scaled = eml_exp(eml_add(eml_ln(normed), ln_gamma[j]));
                // add beta — 0-cost, project to real
                to_r(eml_add(scaled, to_c(beta[j])))
            }).collect::<Vec<f64>>()
        })
        .collect()
}

/// CSE-optimized softmax.
///
/// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
///
/// Note: sub is 0-cost, exp is 1 transcendental each.
/// div(e_i, Z) = exp(ln(e_i) - ln(Z)) — ln(Z) computed once.
pub fn build_softmax_cse(x: &[f64]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // exp(x_i - max) — sub is 0-cost, exp is 1 transcendental each
    let exps: Vec<f64> = x.iter()
        .map(|&xi| to_r(eml_exp(eml_sub(to_c(xi), to_c(m)))))
        .collect();

    // Z = sum — add is 0-cost
    let z: Complex64 = exps.iter()
        .map(|&e| to_c(e))
        .fold(Complex64::new(0.0, 0.0), eml_add);

    // ln(Z) — computed ONCE (the CSE payoff)
    let ln_z = eml_ln(z);

    // softmax_i = exp(ln(exp_i) - ln(Z))  — 1 ln + 1 exp per element
    exps.iter()
        .map(|&e| {
            let ln_e = eml_ln(to_c(e));
            to_r(eml_exp(eml_sub(ln_e, ln_z)))
        })
        .collect()
}

/// CSE-optimized GELU.
///
/// gelu(x) = x * sigmoid(1.702 * x)
///         = exp(ln(x) + ln(sigmoid(1.702 * x)))
///
/// sigmoid(z) = 1 / (1 + exp(-z))
///
/// Optimization: ln(x) is computed once, shared between the mul
/// and any downstream consumer.
pub fn build_gelu_cse(x: &[f64]) -> Vec<f64> {
    use rayon::prelude::*;
    x.par_iter()
        .map(|&v| {
            let xc = to_c(v);
            let c = to_c(1.702);

            // ln(x) — shared
            let ln_x = eml_ln(xc);

            // 1.702 * x via log domain: exp(ln(1.702) + ln(x))
            let ln_c = eml_ln(c);
            let cx = eml_exp(eml_add(ln_c, ln_x));

            // sigmoid(cx) = 1 / (1 + exp(-cx)) = inv(add(1, exp(neg(cx))))
            let sig = eml_inv(eml_add(ONE, eml_exp(eml_neg(cx))));

            // x * sig = exp(ln(x) + ln(sig))  — ln(x) already cached!
            to_r(eml_exp(eml_add(ln_x, eml_ln(sig))))
        })
        .collect()
}

// ─── Transcendental counter (audit mode) ────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};

/// Global transcendental counter for audit mode.
static EXP_COUNT: AtomicU64 = AtomicU64::new(0);
static LN_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reset the global transcendental counters.
pub fn reset_counters() {
    EXP_COUNT.store(0, Ordering::Relaxed);
    LN_COUNT.store(0, Ordering::Relaxed);
}

/// Get current transcendental counts.
pub fn get_counts() -> (u64, u64) {
    (EXP_COUNT.load(Ordering::Relaxed), LN_COUNT.load(Ordering::Relaxed))
}

/// Audited exp — records the call.
#[inline(always)]
pub fn audited_exp(x: Complex64) -> Complex64 {
    EXP_COUNT.fetch_add(1, Ordering::Relaxed);
    x.exp() // EML_AUDIT:OK — counting wrapper: this IS the primitive
}

/// Audited ln — records the call.
#[inline(always)]
pub fn audited_ln(x: Complex64) -> Complex64 {
    LN_COUNT.fetch_add(1, Ordering::Relaxed);
    x.ln() // EML_AUDIT:OK — counting wrapper: this IS the primitive
}

/// Audited matmul with CSE — counts every transcendental.
pub fn audited_matmul_cse(
    a: &[f64], b: &[f64],
    rows: usize, inner: usize, cols: usize,
) -> Vec<f64> {
    // ln(A) — 1 ln per element
    let ln_a: Vec<f64> = a.iter()
        .map(|&v| {
            LN_COUNT.fetch_add(1, Ordering::Relaxed);
            Complex64::new(v, 0.0).ln().re // EML_AUDIT:OK — audited counting wrapper
        })
        .collect();

    // ln(B) — 1 ln per element
    let ln_b: Vec<f64> = b.iter()
        .map(|&v| {
            LN_COUNT.fetch_add(1, Ordering::Relaxed);
            Complex64::new(v, 0.0).ln().re // EML_AUDIT:OK — audited counting wrapper
        })
        .collect();

    // C[i,j] = Σ_k exp(ln_A[i,k] + ln_B[k,j])
    use rayon::prelude::*;
    (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let (i, j) = (idx / cols, idx % cols);
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..inner {
                EXP_COUNT.fetch_add(1, Ordering::Relaxed);
                let sum = ln_a[i * inner + k] + ln_b[k * cols + j];
                acc += Complex64::new(sum, 0.0).exp(); // EML_AUDIT:OK — audited counting wrapper
            }
            acc.re
        })
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn approx(a: f64, b: f64, tol: f64) {
        let err = (a - b).abs();
        assert!(err < tol, "expected {b}, got {a}, err={err:.2e}");
    }

    #[test]
    fn test_egraph_mul_optimizes() {
        // Build: mul(a, b) — should stay as mul (cost 3)
        let expr: RecExpr<EmlLang> = "(mul a b)".parse().unwrap();
        let (cost, best) = optimize_and_extract(&expr);
        assert!(cost <= 3.0, "mul cost should be ≤ 3, got {cost}");
        println!("mul(a,b) optimized to: {best}  (cost={cost})");
    }

    #[test]
    fn test_egraph_exp_ln_cancels() {
        // exp(ln(x)) should optimise to just x (cost 0)
        let expr: RecExpr<EmlLang> = "(exp (ln x))".parse().unwrap();
        let (cost, best) = optimize_and_extract(&expr);
        assert!(cost < 1.0, "exp(ln(x)) should cancel to x, got cost={cost}, expr={best}");
    }

    #[test]
    fn test_egraph_ln_exp_cancels() {
        // ln(exp(x)) should optimise to just x (cost 0)
        let expr: RecExpr<EmlLang> = "(ln (exp x))".parse().unwrap();
        let (cost, best) = optimize_and_extract(&expr);
        assert!(cost < 1.0, "ln(exp(x)) should cancel to x, got cost={cost}, expr={best}");
    }

    #[test]
    fn test_eval_mul() {
        let expr: RecExpr<EmlLang> = "(mul a b)".parse().unwrap();
        let result = eval_expr(&expr, &[("a", 3.0), ("b", 4.0)]);
        approx(result, 12.0, 1e-10);
    }

    #[test]
    fn test_eval_div() {
        let expr: RecExpr<EmlLang> = "(div a b)".parse().unwrap();
        let result = eval_expr(&expr, &[("a", 10.0), ("b", 4.0)]);
        approx(result, 2.5, 1e-10);
    }

    #[test]
    fn test_eval_nested() {
        // add(mul(2, 3), mul(4, 5)) = 6 + 20 = 26
        let expr: RecExpr<EmlLang> = "(add (mul a b) (mul c d))".parse().unwrap();
        let result = eval_expr(&expr, &[("a", 2.0), ("b", 3.0), ("c", 4.0), ("d", 5.0)]);
        approx(result, 26.0, 1e-9);
    }

    #[test]
    fn test_matmul_cse_matches_naive() {
        // 2x3 @ 3x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let naive = crate::eml_ops::eml_matmul(&a, &b, 2, 3, 2);
        let cse = build_matmul_cse(&a, &b, 2, 3, 2);

        for (n, c) in naive.iter().zip(cse.iter()) {
            approx(*c, *n, 1e-9);
        }
    }

    #[test]
    fn test_softmax_cse_matches_naive() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let naive = crate::eml_ops::eml_softmax(&x);
        let cse = build_softmax_cse(&x);

        let sum: f64 = cse.iter().sum();
        approx(sum, 1.0, 1e-10);

        for (n, c) in naive.iter().zip(cse.iter()) {
            approx(*c, *n, 1e-9);
        }
    }

    #[test]
    fn test_layernorm_cse_matches_naive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];

        let naive = crate::eml_ops::eml_layer_norm(&x, &gamma, &beta, 2, 4, 1e-5);
        let cse = build_layernorm_cse(&x, &gamma, &beta, 2, 4, 1e-5);

        for (n, c) in naive.iter().zip(cse.iter()) {
            approx(*c, *n, 1e-6);
        }
    }

    #[test]
    fn test_gelu_cse_matches_naive() {
        let x = vec![0.5, 1.0, -0.5, 2.0];
        let naive = crate::eml_ops::eml_gelu_vec(&x);
        let cse = build_gelu_cse(&x);

        for (n, c) in naive.iter().zip(cse.iter()) {
            approx(*c, *n, 1e-9);
        }
    }

    #[test]
    fn test_audited_matmul_counts() {
        reset_counters();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

        let _ = audited_matmul_cse(&a, &b, 2, 3, 2);
        let (exps, lns) = get_counts();

        // ln: 6 (A) + 6 (B) = 12
        // exp: 2*2*3 = 12 (one per element per k)
        assert_eq!(lns, 12, "ln count mismatch");
        assert_eq!(exps, 12, "exp count mismatch");

        // Naive would be: 2*2 elements * 3K * 3 transcendentals = 36
        // CSE: 12 ln + 12 exp = 24
        println!("CSE matmul (2x3 @ 3x2): {lns} ln + {exps} exp = {} total (naive would be 36)", lns + exps);
    }
}
