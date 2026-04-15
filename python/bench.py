"""
EML Python Benchmark — measures cost of the primitive chain.

Benchmarks:
  1. Scalar primitives: eml, exp, ln, sub, add, mul, div, sqrt
  2. Composite ops: softmax, layer_norm, matmul, gelu
  3. Full forward pass (1-layer toy transformer)
  4. EML call counts per operation
  5. Stepanov power algorithm & monoid optimization comparison

Reports wall-clock time and eml() call counts.
"""

import numpy as np
import time
import sys

from eml_core import (
    eml, eml_exp, eml_ln, eml_sub, eml_add, eml_mul, eml_div,
    eml_neg, eml_inv, eml_sqrt, eml_pow, eml_softmax,
    eml_layer_norm, eml_matmul, eml_gelu, ONE,
    # Stepanov optimized versions
    eml_power_semigroup, eml_power_monoid, _cache,
    eml_neg_r, eml_add_r, eml_inv_r, eml_mul_r, eml_mul_precomp,
    eml_div_r, eml_sqrt_r, eml_gelu_r,
    eml_matmul_precomp, eml_softmax_r, eml_layer_norm_r,
)
from eml_model import init_weights, eml_forward, D_MODEL, D_FF, VOCAB, N_HEADS, D_HEAD

# ─── EML call counter ────────────────────────────────────────────────────────

eml_call_count = 0
_raw_eml = eml

def counting_eml(x, y):
    global eml_call_count
    eml_call_count += 1
    return _raw_eml(x, y)

def reset_count():
    global eml_call_count
    eml_call_count = 0

def get_count():
    return eml_call_count

# Patch eml into both modules
import eml_core, eml_model
eml_core.eml = counting_eml
eml_model.eml = counting_eml


# ─── Timing helper ───────────────────────────────────────────────────────────

def bench(name, fn, iters=100):
    """Run fn() iters times, report avg time and eml() calls."""
    # Warmup
    fn()
    reset_count()

    t0 = time.perf_counter()
    for _ in range(iters):
        reset_count()
        fn()
    t1 = time.perf_counter()

    calls = get_count()  # from last iteration
    avg_us = (t1 - t0) / iters * 1e6
    return avg_us, calls


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)
    w = init_weights(rng)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EML Python Benchmark                                      ║")
    print("║  eml(x,y) = exp(x) - ln(y)                                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Model: 1 layer, {N_HEADS} heads, d_model={D_MODEL}, d_ff={D_FF}, vocab={VOCAB}")
    print()

    # ── 1. Scalar primitives ──────────────────────────────────────────────
    print("─── Scalar Primitives (100 iters each) ─────────────────────────")
    print(f"  {'Op':<12} {'Time (µs)':>12} {'eml() calls':>14}")
    print(f"  {'─'*12} {'─'*12} {'─'*14}")

    scalars = [
        ("eml",    lambda: eml(2.5, 1.3)),
        ("exp",    lambda: eml_exp(2.5)),
        ("ln",     lambda: eml_ln(2.5)),
        ("sub",    lambda: eml_sub(5.0, 3.0)),
        ("neg",    lambda: eml_neg(2.5)),
        ("add",    lambda: eml_add(2.5, 1.3)),
        ("inv",    lambda: eml_inv(2.5)),
        ("mul",    lambda: eml_mul(2.5, 1.3)),
        ("div",    lambda: eml_div(2.5, 1.3)),
        ("pow",    lambda: eml_pow(2.0, 3.0)),
        ("sqrt",   lambda: eml_sqrt(2.5)),
        ("gelu",   lambda: eml_gelu(0.5)),
    ]
    for name, fn in scalars:
        us, calls = bench(name, fn, iters=1000)
        print(f"  {name:<12} {us:>12.1f} {calls:>14}")

    # ── 2. Vector/matrix ops ──────────────────────────────────────────────
    print()
    print("─── Vector/Matrix Ops ─────────────────────────────────────────")
    print(f"  {'Op':<24} {'Time (ms)':>12} {'eml() calls':>14}")
    print(f"  {'─'*24} {'─'*12} {'─'*14}")

    vec8 = rng.standard_normal(8).astype(np.float64) + 2.0  # keep positive for ln
    vec16 = rng.standard_normal(16).astype(np.float64) + 2.0
    mat_4x4 = rng.standard_normal((4, 4)).astype(np.float64) + 2.0
    gamma16 = np.ones(16, dtype=np.float64)
    beta16 = np.zeros(16, dtype=np.float64)

    composites = [
        ("softmax(8)",      lambda: eml_softmax(vec8),                          10),
        ("softmax(16)",     lambda: eml_softmax(vec16),                         10),
        ("layer_norm(1,16)",lambda: eml_layer_norm(vec16.reshape(1,16), gamma16, beta16), 5),
        ("matmul(4,4)@(4,4)", lambda: eml_matmul(mat_4x4, mat_4x4),            5),
        ("gelu(16)",        lambda: np.vectorize(lambda v: np.real(eml_gelu(v)))(vec16), 10),
    ]
    for name, fn, iters in composites:
        us, calls = bench(name, fn, iters=iters)
        print(f"  {name:<24} {us/1000:>12.2f} {calls:>14}")

    # ── 3. Forward pass ──────────────────────────────────────────────────
    print()
    print("─── Forward Pass (T tokens → logits) ──────────────────────────")
    print(f"  {'Seq len':<12} {'Time (s)':>12} {'eml() calls':>14} {'calls/token':>14}")
    print(f"  {'─'*12} {'─'*12} {'─'*14} {'─'*14}")

    for T in [1, 2, 4]:
        token_ids = list(range(T))
        reset_count()
        t0 = time.perf_counter()
        eml_forward(token_ids, w)
        t1 = time.perf_counter()
        calls = get_count()
        elapsed = t1 - t0
        per_tok = calls // T
        print(f"  T={T:<9} {elapsed:>12.3f} {calls:>14,} {per_tok:>14,}")

    # ── 4. EML call depth analysis ────────────────────────────────────────
    print()
    print("─── EML Primitive Depth (calls to build each op) ──────────────")
    print("  (Shows exponential cost of compositional construction)")
    print()

    depth_ops = [
        ("eml(x,y)", lambda: eml(2.5, 1.3)),
        ("exp(x)",   lambda: eml_exp(2.5)),
        ("ln(x)",    lambda: eml_ln(2.5)),
        ("sub(a,b)", lambda: eml_sub(5.0, 3.0)),
        ("add(a,b)", lambda: eml_add(2.5, 1.3)),
        ("mul(a,b)", lambda: eml_mul(2.5, 1.3)),
        ("div(a,b)", lambda: eml_div(2.5, 1.3)),
        ("sqrt(x)",  lambda: eml_sqrt(2.5)),
    ]
    for name, fn in depth_ops:
        reset_count()
        fn()
        calls = get_count()
        bar = "█" * min(calls, 80)
        print(f"  {name:<12} {calls:>6} eml()  {bar}")

    # ── 5. Stepanov Power Algorithm ──────────────────────────────────────
    print()
    print("═══════════════════════════════════════════════════════════════")
    print("  STEPANOV POWER ALGORITHM & MONOID OPTIMIZATIONS")
    print("  From 'Elements of Programming' (Stepanov & McJones, 2009)")
    print("═══════════════════════════════════════════════════════════════")

    # 5a. Power algorithm vs algebraic identity
    print()
    print("─── Power Algorithm vs EML Algebraic Path ─────────────────────")
    print("  power(x, n, ⊕) computes x⊕x⊕...⊕x in O(log n) applications.")
    print("  EML's exp/ln gives O(1). Which wins?")
    print()
    print(f"  {'Operation':<28} {'Method':<16} {'eml() calls':>12}")
    print(f"  {'─'*28} {'─'*16} {'─'*12}")

    # n*x via power algorithm (additive monoid)
    for n in [5, 10, 100]:
        # Power algorithm: O(log n) additions
        _cache.reset()
        reset_count()
        eml_power_semigroup(3.0, n, eml_add)
        power_calls = get_count()

        # Power with reduced add
        _cache.reset()
        reset_count()
        eml_power_semigroup(3.0, n, eml_add_r)
        power_r_calls = get_count()

        # Algebraic: eml_mul(n, x)
        reset_count()
        eml_mul(float(n), 3.0)
        alg_calls = get_count()

        # Algebraic reduced
        _cache.reset()
        reset_count()
        eml_mul_r(float(n), 3.0)
        alg_r_calls = get_count()

        print(f"  {n}*x = x+x+...+x            {'power(add)':16} {power_calls:>12}")
        print(f"  {'':<28} {'power(add_r)':16} {power_r_calls:>12}")
        print(f"  {'':<28} {'eml_mul':16} {alg_calls:>12}")
        print(f"  {'':<28} {'eml_mul_r':16} {alg_r_calls:>12}")
        print()

    # x^n via power algorithm (multiplicative monoid)
    for n in [8, 16]:
        _cache.reset()
        reset_count()
        eml_power_semigroup(2.0, n, eml_mul)
        power_calls = get_count()

        _cache.reset()
        reset_count()
        eml_power_semigroup(2.0, n, eml_mul_r)
        power_r_calls = get_count()

        reset_count()
        eml_pow(2.0, float(n))
        alg_calls = get_count()

        print(f"  x^{n} = x*x*...*x            {'power(mul)':16} {power_calls:>12}")
        print(f"  {'':<28} {'power(mul_r)':16} {power_r_calls:>12}")
        print(f"  {'':<28} {'eml_pow':16} {alg_calls:>12}")
        print()

    print("  → EML's exp/ln IS the ultimate 'power algorithm': O(1) via")
    print("    the logarithmic morphism. Stepanov's O(log n) can't beat it")
    print("    for scalar arithmetic.")
    print()
    print("  → But the STRUCTURAL insight transfers: factor morphisms,")
    print("    cache identities, precompute shared subexpressions.")

    # 5b. Reduced ops comparison
    print()
    print("─── Monoid Identity Caching: Naive vs Reduced ─────────────────")
    print("  Cache const_zero() and half to eliminate redundant eml() calls.")
    print()
    print(f"  {'Op':<12} {'Naive':>8} {'Reduced':>8} {'Saved':>8} {'%':>6}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

    comparisons = [
        ("neg(x)",  lambda: eml_neg(2.5),     lambda: eml_neg_r(2.5)),
        ("add(a,b)", lambda: eml_add(2.5,1.3), lambda: eml_add_r(2.5,1.3)),
        ("inv(z)",  lambda: eml_inv(2.5),      lambda: eml_inv_r(2.5)),
        ("mul(a,b)", lambda: eml_mul(2.5,1.3), lambda: eml_mul_r(2.5,1.3)),
        ("div(a,b)", lambda: eml_div(2.5,1.3), lambda: eml_div_r(2.5,1.3)),
        ("sqrt(x)", lambda: eml_sqrt(2.5),     lambda: eml_sqrt_r(2.5)),
        ("gelu(x)", lambda: eml_gelu(0.5),     lambda: eml_gelu_r(0.5)),
    ]
    for name, naive_fn, reduced_fn in comparisons:
        reset_count()
        naive_fn()
        naive_calls = get_count()

        _cache.reset()
        reset_count()
        reduced_fn()
        reduced_calls = get_count()

        saved = naive_calls - reduced_calls
        pct = (saved / naive_calls * 100) if naive_calls else 0
        print(f"  {name:<12} {naive_calls:>8} {reduced_calls:>8} {saved:>8} {pct:>5.0f}%")

    # 5c. Morphism-factored matmul
    print()
    print("─── Morphism Factoring: Matmul with Precomputed ln() ──────────")
    print("  ln: (R>0, ×) → (R, +) is a monoid homomorphism.")
    print("  Precompute ln(A), ln(B) once, reuse across all output cells.")
    print()

    for sz_name, mat_a, mat_b in [
        ("4×4 @ 4×4",   mat_4x4, mat_4x4),
    ]:
        # Verify correctness first
        _cache.reset()
        r_naive = eml_matmul(mat_a, mat_b)
        _cache.reset()
        r_precomp = eml_matmul_precomp(mat_a, mat_b)
        err = np.max(np.abs(r_naive - r_precomp))

        # Count calls
        reset_count()
        eml_matmul(mat_a, mat_b)
        naive_calls = get_count()

        _cache.reset()
        reset_count()
        eml_matmul_precomp(mat_a, mat_b)
        precomp_calls = get_count()

        saved = naive_calls - precomp_calls
        pct = (saved / naive_calls * 100) if naive_calls else 0

        # Time
        t0 = time.perf_counter()
        for _ in range(5):
            eml_matmul(mat_a, mat_b)
        t_naive = (time.perf_counter() - t0) / 5

        t0 = time.perf_counter()
        for _ in range(5):
            _cache.reset()
            eml_matmul_precomp(mat_a, mat_b)
        t_precomp = (time.perf_counter() - t0) / 5

        print(f"  {sz_name}:")
        print(f"    Naive:    {naive_calls:>8} calls   {t_naive*1000:>8.2f} ms")
        print(f"    Precomp:  {precomp_calls:>8} calls   {t_precomp*1000:>8.2f} ms")
        print(f"    Saved:    {saved:>8} calls   ({pct:.0f}% reduction)")
        print(f"    Max err:  {err:.2e} (correctness check)")

    # 5d. Composite ops comparison
    print()
    print("─── Full Composite Ops: Naive vs Stepanov-Optimized ───────────")
    print(f"  {'Op':<24} {'Naive calls':>14} {'Optimized':>14} {'Saved %':>8}")
    print(f"  {'─'*24} {'─'*14} {'─'*14} {'─'*8}")

    composite_cmp = [
        ("softmax(8)",
         lambda: eml_softmax(vec8),
         lambda: eml_softmax_r(vec8)),
        ("softmax(16)",
         lambda: eml_softmax(vec16),
         lambda: eml_softmax_r(vec16)),
        ("layer_norm(1,16)",
         lambda: eml_layer_norm(vec16.reshape(1,16), gamma16, beta16),
         lambda: eml_layer_norm_r(vec16.reshape(1,16), gamma16, beta16)),
        ("gelu(16)",
         lambda: np.vectorize(lambda v: np.real(eml_gelu(v)))(vec16),
         lambda: np.vectorize(lambda v: np.real(eml_gelu_r(v)))(vec16)),
    ]
    for name, naive_fn, opt_fn in composite_cmp:
        reset_count()
        naive_fn()
        nc = get_count()

        _cache.reset()
        reset_count()
        opt_fn()
        oc = get_count()

        saved = nc - oc
        pct = (saved / nc * 100) if nc else 0
        print(f"  {name:<24} {nc:>14,} {oc:>14,} {pct:>7.0f}%")

    # 5e. Visual comparison
    print()
    print("─── Call Depth: Naive vs Reduced (visual) ─────────────────────")
    visual_ops = [
        ("add",      lambda: eml_add(2.5, 1.3),      lambda: eml_add_r(2.5, 1.3)),
        ("mul",      lambda: eml_mul(2.5, 1.3),      lambda: eml_mul_r(2.5, 1.3)),
        ("div",      lambda: eml_div(2.5, 1.3),      lambda: eml_div_r(2.5, 1.3)),
        ("sqrt",     lambda: eml_sqrt(2.5),           lambda: eml_sqrt_r(2.5)),
        ("gelu",     lambda: eml_gelu(0.5),           lambda: eml_gelu_r(0.5)),
    ]
    for name, naive_fn, reduced_fn in visual_ops:
        reset_count()
        naive_fn()
        nc = get_count()
        _cache.reset()
        reset_count()
        reduced_fn()
        rc = get_count()
        bar_n = "█" * nc
        bar_r = "▓" * rc
        print(f"  {name:<6} naive:   {nc:>3}  {bar_n}")
        print(f"  {'':<6} reduced: {rc:>3}  {bar_r}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
