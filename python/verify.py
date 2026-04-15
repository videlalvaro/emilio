"""
EML LLM POC - Verification & Demo

1. Verifies EML primitive correctness against math identities from the paper
2. Verifies derived ops against numpy reference
3. Runs full forward pass and generation demo
4. Shows EML call counts to make the primitive-reduction concrete
"""

import numpy as np
import sys
from eml_core import (eml, eml_exp, eml_ln, eml_softmax, eml_sqrt,
                     eml_sub, eml_add, eml_mul, eml_div, eml_neg, eml_inv)
from eml_model import init_weights, eml_forward, eml_generate, VOCAB

# ─── EML call counter ────────────────────────────────────────────────────────

eml_call_count = 0
_raw_eml = eml

def counting_eml(x, y):
    global eml_call_count
    eml_call_count += 1
    return _raw_eml(x, y)

# Patch into modules
import eml_core
import eml_model
eml_core.eml = counting_eml
eml_model.eml = counting_eml

# ─── Test 1: Paper's explicit identities ────────────────────────────────────

def test_paper_identities():
    print("=" * 60)
    print("TEST 1: Paper's explicit EML identities")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    def check(name, got, expected, tol=1e-9):
        nonlocal passed, failed
        got = np.real(got)
        err = abs(got - expected)
        ok = err < tol
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: got {got:.10f}, expected {expected:.10f}, err={err:.2e}")
        if ok: passed += 1
        else:  failed += 1
    
    # Direct from paper abstract:
    # exp(x) = eml(x, 1)
    for xv in [0.0, 1.0, -1.0, 2.5]:
        got = eml(xv, 1.0)
        check(f"exp({xv})", got, np.exp(xv))
    
    # ln(x) = eml(1, eml(eml(1, x), 1))
    for xv in [0.5, 1.0, 2.0, np.e]:
        got = eml(1.0, eml(eml(1.0, xv), 1.0))
        check(f"ln({xv:.4f})", got, np.log(xv))
    
    # e = eml(1, 1)
    check("e = eml(1,1)", eml(1.0, 1.0), np.e)
    
    # 0 = eml(1, eml(e, 1)) = e - ln(e^e) ... let's verify
    e_val = eml(1.0, 1.0)
    zero = eml(1.0, eml(e_val, 1.0))
    check("0 via EML", zero, 0.0)
    
    print(f"\n  Paper identities: {passed} passed, {failed} failed\n")
    return failed == 0


# ─── Test 2: Derived ops ────────────────────────────────────────────────────

def test_derived_ops():
    print("=" * 60)
    print("TEST 2: Derived operations")
    print("=" * 60)
    
    passed = 0; failed = 0
    
    def check(name, got, expected, tol=1e-6):
        nonlocal passed, failed
        got = np.real(got)
        err = abs(got - expected)
        ok = err < tol
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {got:.8f} vs {expected:.8f} (err={err:.2e})")
        if ok: passed += 1
        else:  failed += 1
    
    # eml_exp
    for x in [0.0, 1.0, -0.5, 2.0]:
        check(f"eml_exp({x})", eml_exp(x), np.exp(x))
    
    # eml_ln
    for x in [0.1, 1.0, 2.0, 10.0]:
        check(f"eml_ln({x})", eml_ln(x), np.log(x))
    
    # eml_sqrt
    for x in [1.0, 4.0, 9.0, 2.0]:
        check(f"eml_sqrt({x})", eml_sqrt(x), np.sqrt(x))
    
    # eml_sub
    for a, b in [(5.0, 3.0), (1.0, 1.0), (0.5, 2.0), (10.0, 7.3)]:
        check(f"eml_sub({a},{b})", eml_sub(a, b), a - b)
    
    # eml_neg
    for x in [1.0, 3.5, 0.1, 7.0]:
        check(f"eml_neg({x})", eml_neg(x), -x)
    
    # eml_add
    for a, b in [(1.0, 2.0), (0.5, 0.5), (3.0, 7.0), (0.1, 0.2)]:
        check(f"eml_add({a},{b})", eml_add(a, b), a + b)
    
    # eml_mul
    for a, b in [(2.0, 3.0), (0.5, 4.0), (1.5, 2.5), (7.0, 0.1)]:
        check(f"eml_mul({a},{b})", eml_mul(a, b), a * b)
    
    # eml_div
    for a, b in [(6.0, 3.0), (1.0, 4.0), (10.0, 2.5), (7.0, 0.5)]:
        check(f"eml_div({a},{b})", eml_div(a, b), a / b)
    
    # eml_inv
    for x in [1.0, 2.0, 0.5, 4.0]:
        check(f"eml_inv({x})", eml_inv(x), 1.0 / x)
    
    # eml_softmax
    x = np.array([1.0, 2.0, 3.0, 4.0])
    ref = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    got = eml_softmax(x)
    check("softmax sum=1", np.sum(got), 1.0)
    check("softmax max_diff", np.max(np.abs(got - ref)), 0.0, tol=1e-6)
    
    print(f"\n  Derived ops: {passed} passed, {failed} failed\n")
    return failed == 0


# ─── Test 3: Full forward pass ───────────────────────────────────────────────

def test_forward_pass():
    print("=" * 60)
    print("TEST 3: Full transformer forward pass")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    w = init_weights(rng)
    
    prompt = [1, 5, 12, 3]
    print(f"  Prompt token IDs: {prompt}")
    
    global eml_call_count
    eml_call_count = 0
    
    logits = eml_forward(prompt, w)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Last pos logits (first 8): {logits[-1, :8]}")
    
    # Verify output is finite
    assert np.all(np.isfinite(logits)), "Non-finite logits!"
    
    # Verify softmax over last position sums to 1
    probs = eml_softmax(logits[-1])
    prob_sum = np.sum(probs)
    print(f"  Softmax(logits[-1]) sums to: {prob_sum:.8f}")
    assert abs(prob_sum - 1.0) < 1e-6
    
    print(f"\n  EML primitive calls in forward pass: {eml_call_count:,}")
    print(f"  ✓ Forward pass complete\n")
    return True


# ─── Test 4: Generation ──────────────────────────────────────────────────────

def test_generation():
    print("=" * 60)
    print("TEST 4: Autoregressive token generation")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    w = init_weights(rng)
    
    prompt = [1, 5, 12]
    max_new = 6
    
    print(f"  Prompt:  {prompt}")
    
    global eml_call_count
    eml_call_count = 0
    
    output = eml_generate(prompt, w, max_new=max_new, temperature=1.0)
    
    generated = output[len(prompt):]
    print(f"  Generated: {generated}")
    print(f"  Full sequence: {output}")
    print(f"  EML primitive calls for {max_new} tokens: {eml_call_count:,}")
    print(f"  Avg EML calls per token: {eml_call_count / max_new:,.0f}")
    print(f"  ✓ Generation complete\n")
    return True


# ─── EML depth analysis ──────────────────────────────────────────────────────

def show_eml_depth_analysis():
    print("=" * 60)
    print("EML DEPTH ANALYSIS: ops expressed as EML trees")
    print("=" * 60)
    print()
    print("  From paper Table 4 (approximate depths in pure EML form):")
    print()
    
    ops = [
        ("exp(x)",        "eml(x, 1)",                              1),
        ("ln(x)",         "eml(1, eml(eml(1,x), 1))",              3),
        ("e",             "eml(1, 1)",                              1),
        ("0",             "eml(1, eml(e, 1))",                      2),
        ("mul(x,y)",      "exp(ln(x) + ln(y))  [depth ~8 EML]",    8),
        ("add(x,y)",      "ln(exp(x) * exp(y)) [depth ~8 EML]",    8),
        ("sqrt(x)",       "exp(0.5 * ln(x))    [depth ~5 EML]",    5),
        ("softmax(x_i)",  "exp(x_i - ln(Σexp)) [depth ~6 EML]",   6),
        ("layernorm",     "via sqrt, add, div  [depth ~12 EML]",   12),
        ("gelu(x)",       "x * σ(1.702x)       [depth ~10 EML]",  10),
    ]
    
    for name, eml_form, depth in ops:
        bar = "█" * depth
        print(f"  {name:20s} depth={depth:2d}  {bar}")
    
    print()
    print("  Grammar: S → 1 | eml(S, S)")
    print("  Every op is a binary tree of identical eml nodes.")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  EML-LLM: Tiny Transformer via eml(x,y) = exp(x)-ln(y) ║")
    print("║  Based on: Odrzywołek (2026), arXiv:2603.21852           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    show_eml_depth_analysis()
    
    all_passed = True
    all_passed &= test_paper_identities()
    all_passed &= test_derived_ops()
    all_passed &= test_forward_pass()
    all_passed &= test_generation()
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        print()
        print("POC demonstrates:")
        print("  • eml(x,y) = exp(x) - ln(y) is the sole arithmetic primitive")
        print("  • exp, ln, sqrt, softmax, layernorm, gelu all EML-derived")
        print("  • Complete transformer forward pass runs on EML alone")
        print("  • Autoregressive generation produces valid token sequences")
        print()
        print("Honest caveats:")
        print("  • Token lookup (discrete) and argmax (comparison) are not EML")
        print("  • Negation routes through complex ln(-1) = iπ per the paper")
        print("  • Performance is irrelevant for a POC; depth-8 mul trees are slow")
        print("  • Weights are random — this is architecture, not a trained model")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
