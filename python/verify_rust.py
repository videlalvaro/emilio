"""
Verify that emilio (Rust+Rayon) matches eml_core (Python) exactly.
Also benchmarks both backends.
"""

import time
import numpy as np

# Python reference
import eml_core as py

# Rust fused+parallel
import emilio as rs

def check(name, py_val, rs_val, tol=1e-9):
    py_v = np.real(py_val)
    rs_v = np.real(rs_val)
    err = abs(py_v - rs_v)
    ok = err < tol
    status = "✓" if ok else "✗"
    print(f"  {status} {name}: py={py_v:.10f} rs={rs_v:.10f} err={err:.2e}")
    return ok

def check_arr(name, py_arr, rs_arr, tol=1e-6):
    py_a = np.real(np.asarray(py_arr, dtype=np.float64))
    rs_a = np.real(np.asarray(rs_arr, dtype=np.float64))
    max_err = np.max(np.abs(py_a - rs_a))
    ok = max_err < tol
    status = "✓" if ok else "✗"
    print(f"  {status} {name}: max_err={max_err:.2e}")
    return ok


print("=" * 60)
print("SCALAR OPS: Python vs Rust")
print("=" * 60)

all_ok = True

# eml primitive
all_ok &= check("eml(1, 1)", py.eml(1.0, 1.0), rs.eml(1.0, 1.0))
all_ok &= check("eml(2, 3)", py.eml(2.0, 3.0), rs.eml(2.0, 3.0))

# exp
for x in [0.0, 1.0, -1.0, 2.5]:
    all_ok &= check(f"exp({x})", py.eml_exp(x), rs.eml_exp(x))

# ln
for x in [0.5, 1.0, 2.0, 10.0]:
    all_ok &= check(f"ln({x})", py.eml_ln(x), rs.eml_ln(x))

# sub
for a, b in [(5.0, 3.0), (0.5, 2.0)]:
    all_ok &= check(f"sub({a},{b})", py.eml_sub(a, b), rs.eml_sub(a, b))

# neg
for x in [1.0, 3.5, 7.0]:
    all_ok &= check(f"neg({x})", py.eml_neg(x), rs.eml_neg(x))

# add
for a, b in [(1.0, 2.0), (0.5, 0.5), (0.1, 0.2)]:
    all_ok &= check(f"add({a},{b})", py.eml_add(a, b), rs.eml_add(a, b))

# mul
for a, b in [(2.0, 3.0), (0.5, 4.0), (-1.0, 5.0), (7.0, 0.1)]:
    all_ok &= check(f"mul({a},{b})", py.eml_mul(a, b), rs.eml_mul(a, b))

# div
for a, b in [(6.0, 3.0), (1.0, 4.0)]:
    all_ok &= check(f"div({a},{b})", py.eml_div(a, b), rs.eml_div(a, b))

# inv
for x in [1.0, 2.0, 0.5]:
    all_ok &= check(f"inv({x})", py.eml_inv(x), rs.eml_inv(x))

# sqrt
for x in [1.0, 4.0, 9.0, 2.0]:
    all_ok &= check(f"sqrt({x})", py.eml_sqrt(x), rs.eml_sqrt(x))

# gelu
for x in [0.0, 1.0, -1.0, 2.0]:
    all_ok &= check(f"gelu({x})", py.eml_gelu(x), rs.eml_gelu(x), tol=1e-6)


print()
print("=" * 60)
print("ARRAY OPS: Python vs Rust")
print("=" * 60)

# softmax
x = np.array([1.0, 2.0, 3.0, 4.0])
py_sm = py.eml_softmax(x)
rs_sm = np.asarray(rs.eml_softmax(x))
all_ok &= check_arr("softmax([1,2,3,4])", py_sm, rs_sm)

# matmul
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
py_mm = py.eml_matmul(A, B)
rs_mm = np.asarray(rs.eml_matmul(A, B))
all_ok &= check_arr("matmul([[1,2],[3,4]] @ [[5,6],[7,8]])", py_mm, rs_mm)

# layer norm
x_ln = np.array([[1.0, 2.0, 3.0, 4.0]])
gamma = np.ones(4)
beta = np.zeros(4)
py_ln = py.eml_layer_norm(x_ln, gamma, beta)
rs_ln = np.asarray(rs.eml_layer_norm(x_ln, gamma, beta, 1e-5))
all_ok &= check_arr("layer_norm([1,2,3,4])", py_ln, rs_ln)


print()
print("=" * 60)
print("BENCHMARK: matmul (16x16 @ 16x32)")
print("=" * 60)

rng = np.random.default_rng(42)
A_bench = rng.standard_normal((16, 16))
B_bench = rng.standard_normal((16, 32))

# Rust
t0 = time.perf_counter()
rs_result = np.asarray(rs.eml_matmul(A_bench, B_bench))
t_rs = time.perf_counter() - t0
print(f"  Rust:   {t_rs:.4f}s")

# Python
t0 = time.perf_counter()
py_result = py.eml_matmul(A_bench, B_bench)
t_py = time.perf_counter() - t0
print(f"  Python: {t_py:.4f}s")

speedup = t_py / t_rs if t_rs > 0 else float('inf')
print(f"  Speedup: {speedup:.1f}x")

all_ok &= check_arr("matmul(16x16 @ 16x32)", py_result, rs_result, tol=1e-4)


print()
print("=" * 60)
if all_ok:
    print("ALL CHECKS PASSED — Rust mirrors Python correctly")
else:
    print("SOME CHECKS FAILED")
print("=" * 60)
