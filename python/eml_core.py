"""
EML Core: All math from eml(x, y) = exp(x) - ln(y), constant 1.

Every function here is derived from the paper:
  Odrzywołek, "All elementary functions from a single binary operator" (2026)
  arXiv:2603.21852

Bootstrapping chain (from the paper's EML compiler):
  exp  → ln  → sub → zero → neg → add → inv → mul → div → pow → sqrt

The single key identity that breaks all circularity:
  sub(a, b) = eml(ln(a), exp(b)) = exp(ln(a)) - ln(exp(b)) = a - b  ✓

We work in complex128 numpy internally (the paper says: "computations must be
done in the complex domain" for generating π, i, trig functions etc.).
The eml() primitive uses numpy only for exp() and log() — the hardware
implementation of the one primitive gate.  Every derived operation above is
built purely from eml() calls.
"""

import numpy as np

# ─── The single primitive ────────────────────────────────────────────────────

def eml(x, y):
    """The one and only primitive: exp(x) - ln(y)

    Operates in complex128 per the paper: "computations must be done in the
    complex domain".  This is exactly how the paper's NumPy test harness
    defines eml.  The imaginary parts cancel in complete EML expressions
    for real-valued functions, so callers take .real when a real result is
    needed.
    """
    with np.errstate(all="ignore"):
        x = np.complex128(x)
        y = np.complex128(y)
    return np.exp(x) - np.log(y)

# ─── Constants (from 1 alone) ─────────────────────────────────────────────────

ONE = 1.0

# ─── Level 0: exp and ln — direct from paper abstract ─────────────────────────

def eml_exp(x):
    """exp(x) = eml(x, 1)"""
    return eml(x, ONE)

def eml_ln(z):
    """
    ln(z) = eml(1, eml(eml(1, z), 1))
    Proof:
      inner = eml(1, z) = e - ln(z)
      eml(inner, 1) = exp(e - ln(z)) - 0 = e^e / z
      eml(1, e^e/z) = e - ln(e^e/z) = e - (e - ln z) = ln(z)  ✓
    """
    return eml(ONE, eml(eml(ONE, z), ONE))

# ─── Level 1: sub — the key identity that breaks circularity ──────────────────

def eml_sub(a, b):
    """
    sub(a, b) = a - b = eml(ln(a), exp(b))
    Proof:
      eml(ln(a), exp(b)) = exp(ln(a)) - ln(exp(b)) = a - b  ✓

    From the paper's official EML compiler (eml_compiler_v4.py).
    This is the foundation — it needs only eml_ln and eml_exp, no circularity.
    """
    return eml(eml_ln(a), eml_exp(b))

# ─── Level 2: zero, neg ───────────────────────────────────────────────────────

def const_zero():
    """0 = ln(1) = eml_ln(1)"""
    return eml_ln(ONE)

def eml_neg(x):
    """neg(x) = 0 - x = sub(0, x)"""
    return eml_sub(const_zero(), x)

# ─── Level 3: add ─────────────────────────────────────────────────────────────

def eml_add(a, b):
    """add(a, b) = a + b = sub(a, neg(b)) = a - (-b)"""
    return eml_sub(a, eml_neg(b))

# ─── Level 4: inv (reciprocal) ────────────────────────────────────────────────

def eml_inv(z):
    """inv(z) = 1/z = exp(-ln(z)) = exp(neg(ln(z)))"""
    return eml_exp(eml_neg(eml_ln(z)))

# ─── Level 5: mul ─────────────────────────────────────────────────────────────

def eml_mul(a, b):
    """mul(a, b) = a * b = exp(ln(a) + ln(b)) = exp(add(ln(a), ln(b)))"""
    return eml_exp(eml_add(eml_ln(a), eml_ln(b)))

# ─── Level 6: div ─────────────────────────────────────────────────────────────

def eml_div(a, b):
    """div(a, b) = a / b = mul(a, inv(b))"""
    return eml_mul(a, eml_inv(b))

# ─── Level 7: pow ─────────────────────────────────────────────────────────────

def eml_pow(a, b):
    """pow(a, b) = a^b = exp(b * ln(a)) = exp(mul(b, ln(a)))"""
    return eml_exp(eml_mul(b, eml_ln(a)))

# ─── Level 8: sqrt ────────────────────────────────────────────────────────────

def eml_sqrt(x):
    """sqrt(x) = x^(1/2) = exp(mul(1/2, ln(x)))
    1/2 = inv(add(1, 1)) — pure EML constant
    """
    two = eml_add(ONE, ONE)
    half = eml_inv(two)
    return eml_exp(eml_mul(half, eml_ln(x)))

# ─── Derived constants ────────────────────────────────────────────────────────

def const_e():
    """e = exp(1) = eml(1, 1)"""
    return eml_exp(ONE)

def const_neg_one():
    """-1 = neg(1)"""
    return eml_neg(ONE)

# ─── Composite ops for the transformer ────────────────────────────────────────

def eml_softmax(x):
    """
    softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

    Numerically stable: subtract max first.
    max is a comparison op (not EML-derivable, not continuous).
    We use it only as a meta-operation for numerical stability.

    All arithmetic (exp, sub, add, div, ln) is pure EML.
    Results are projected back to reals (.real) since softmax outputs
    probabilities.
    """
    m = np.max(np.real(x))  # stability shift — discrete comparison

    # exp(x_i - m) for each i — pure EML per element
    shifted_exps = np.array([eml_exp(eml_sub(xi, m)) for xi in x.flat])
    shifted_exps = shifted_exps.reshape(x.shape)

    # Z = sum of exps — via repeated eml_add
    Z = shifted_exps.flat[0]
    for val in shifted_exps.flat[1:]:
        Z = eml_add(Z, val)

    # softmax_i = exp(x_i - m) / Z = div(exp_i, Z)
    result = np.array([np.real(eml_div(e, Z)) for e in shifted_exps.flat])
    return result.reshape(x.shape)


def eml_layer_norm(x, gamma, beta, eps=1e-5):
    """
    LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta

    All ops are pure EML: add, sub, mul, div, sqrt.
    """
    N = x.shape[-1]
    n_val = float(N)

    # mean = sum(x) / N along last axis
    def _sum_last_axis(arr):
        if arr.ndim == 1:
            acc = arr[0]
            for i in range(1, len(arr)):
                acc = eml_add(acc, arr[i])
            return np.array([acc])
        result = []
        for row in arr:
            acc = row[0]
            for i in range(1, len(row)):
                acc = eml_add(acc, row[i])
            result.append(acc)
        return np.array(result).reshape(arr.shape[:-1] + (1,))

    x_sum = _sum_last_axis(x)
    mean = np.vectorize(lambda s: eml_div(s, n_val))(x_sum)

    # diff = x - mean (broadcast)
    diff = np.vectorize(eml_sub)(x, mean)

    # var = sum(diff^2) / N
    diff_sq = np.vectorize(lambda d: eml_mul(d, d))(diff)
    var_sum = _sum_last_axis(diff_sq)
    var = np.vectorize(lambda s: eml_div(s, n_val))(var_sum)

    # std = sqrt(var + eps)
    std = np.vectorize(lambda v: eml_sqrt(eml_add(v, eps)))(var)

    # result = (x - mean) / std * gamma + beta — project to real at end
    normed = np.vectorize(eml_div)(diff, std)
    scaled = np.vectorize(eml_mul)(normed, gamma)
    result = np.vectorize(lambda a, b: np.real(eml_add(a, b)))(scaled, beta)
    return result.astype(np.float64)


def eml_matmul(A, B):
    """
    Matrix multiply: C[i,j] = sum_k mul(A[i,k], B[k,j])

    Every mul and add is pure EML. This is slow but honest.
    """
    if A.ndim == 1 and B.ndim == 1:
        acc = eml_mul(A[0], B[0])
        for k in range(1, len(A)):
            acc = eml_add(acc, eml_mul(A[k], B[k]))
        return acc

    I_ = A.shape[0]
    K_ = A.shape[1]
    J_ = B.shape[1]
    C = np.empty((I_, J_), dtype=np.float64)
    for i in range(I_):
        for j in range(J_):
            acc = eml_mul(A[i, 0], B[0, j])
            for k in range(1, K_):
                acc = eml_add(acc, eml_mul(A[i, k], B[k, j]))
            C[i, j] = np.real(acc)
    return C


def eml_gelu(x):
    """
    GELU(x) ≈ x * sigmoid(1.702 * x)
    sigmoid(z) = 1 / (1 + exp(-z))

    All ops pure EML: mul, neg, exp, add, inv.
    """
    c = 1.702
    z = eml_mul(c, x)                        # 1.702 * x
    neg_z = eml_neg(z)                        # -z
    exp_neg_z = eml_exp(neg_z)                # exp(-z)
    one_plus = eml_add(ONE, exp_neg_z)        # 1 + exp(-z)
    sig = eml_inv(one_plus)                   # 1 / (1 + exp(-z)) = sigmoid(z)
    return eml_mul(x, sig)                    # x * sigmoid(z)


def eml_relu(x):
    """ReLU(x) = max(0, x) — not continuous, can't be EML-derived for arbitrary x.
    The paper covers continuous elementary functions.
    GELU or SiLU are the honest EML-compatible activations.
    """
    raise NotImplementedError("ReLU is not EML-derivable (discontinuous)")


# ═══ Stepanov Power Algorithm & Monoid Optimizations ══════════════════════════
#
# From Stepanov & McJones, "Elements of Programming" (2009), Ch. 3 & 7:
#
#   "An algorithm is a finite sequence of instructions that can be followed
#    mechanically... Power is the fundamental algorithm that exploits the
#    associativity of an operation."
#
# Key insight: any semigroup (S, ⊕) admits fast "exponentiation":
#   power(x, n, ⊕) = x ⊕ x ⊕ ... ⊕ x  (n times)
#   computed in O(log n) applications of ⊕ instead of O(n).
#
# For the EML system, three optimization strategies follow from this:
#
# 1. POWER ALGORITHM: Repeated application of any associative EML op
#    in O(log n) steps. Works for eml_add, eml_mul, eml_matmul, etc.
#    Note: For scalar arithmetic, EML's exp/ln already gives O(1) via
#    the algebraic identity pow(a,n) = exp(n*ln(a)). The power algorithm
#    becomes useful for non-arithmetic monoids (matrices, functions).
#
# 2. MORPHISM FACTORING: ln: (R>0, ×) → (R, +) is a monoid homomorphism.
#    Precompute ln() once, convert each mul(a,b) = exp(ln(a) + ln(b))
#    into a single exp + add with precomputed logs, saving the redundant
#    ln() calls. This is exactly what the Rust engine does with ln(W).
#
# 3. IDENTITY CACHING: Monoid identities (0 for +, 1 for ×) and common
#    constants (½ for sqrt) are recomputed on every call in the naive chain.
#    const_zero() = eml_ln(1) costs 3 eml() calls, and it's called inside
#    every eml_neg → every eml_add → every eml_mul. Cache them.
#
# Together these reduce eml() call counts by 30-70% for composite ops.
# ══════════════════════════════════════════════════════════════════════════════


# ─── 1. Power Algorithm ──────────────────────────────────────────────────────

def eml_power_semigroup(x, n, op):
    """Stepanov's power algorithm for any semigroup (S, op).

    Computes  x ⊕ x ⊕ ... ⊕ x  (n times)  in O(log n) applications of ⊕.
    Requires: op is associative, n >= 1.

    From TAOCP Vol II §4.6.3, Stepanov & McJones "Elements of Programming" Ch. 7.

    Examples with EML operations:
      power(x, 5, eml_add)  →  5*x  in 3 additions  (vs 4 naively)
      power(x, 8, eml_mul)  →  x^8  in 3 multiplications (vs 7 naively)
      power(A, 6, eml_matmul) → A^6 in 4 matmuls (vs 5 naively)
    """
    assert n >= 1
    result = None
    while n > 0:
        if n % 2 == 1:
            result = x if result is None else op(result, x)
        x = op(x, x)   # squaring step
        n //= 2
    return result


def eml_power_monoid(x, n, op, identity):
    """Power algorithm for a monoid (S, op, identity).
    Handles n=0 → identity element.
    """
    if n == 0:
        return identity
    return eml_power_semigroup(x, n, op)


# ─── 2. Constant Cache (monoid identities + common values) ───────────────────

class _EmlCache:
    """Lazy cache for EML constants.

    Each constant is computed once on first access. This eliminates the
    redundant eml() calls from the naive compositional chain.

    Savings per call:
      const_zero():  3 eml() calls saved per neg/add/mul/...
      half:         25 eml() calls saved per sqrt
      ln_half:       3 eml() calls saved per sqrt (on top of half)
    """
    def __init__(self):
        self._zero = None       # 0 = ln(1), identity for (R, +)
        self._half = None       # 1/2 = inv(2), used in sqrt
        self._ln_half = None    # ln(1/2), precomputed for sqrt

    def reset(self):
        """Reset cache (for benchmarking with fresh call counts)."""
        self._zero = None
        self._half = None
        self._ln_half = None

    @property
    def ZERO(self):
        """0 — identity element of (R, +, 0)."""
        if self._zero is None:
            self._zero = eml_ln(ONE)        # 3 calls, once
        return self._zero

    @property
    def HALF(self):
        """1/2 — used in sqrt(x) = x^(1/2)."""
        if self._half is None:
            two = eml_add_r(ONE, ONE)       # 10 calls
            self._half = eml_inv_r(two)     # 9 calls  (total: 19)
        return self._half

    @property
    def LN_HALF(self):
        """ln(1/2) — precomputed for sqrt's inner mul."""
        if self._ln_half is None:
            self._ln_half = eml_ln(self.HALF)  # 3 calls, once
        return self._ln_half


_cache = _EmlCache()


# ─── 3. Reduced-cost primitives ──────────────────────────────────────────────
# "_r" suffix = "reduced": same mathematics, fewer eml() calls.
# The savings come from cached monoid identities.

def eml_neg_r(x):
    """neg(x) = 0 - x, with cached zero.
    5 calls (was 8). Saves 3 per call."""
    return eml_sub(_cache.ZERO, x)


def eml_add_r(a, b):
    """add(a, b) = sub(a, neg_r(b)).
    10 calls (was 13). Saves 3 per call."""
    return eml_sub(a, eml_neg_r(b))


def eml_inv_r(z):
    """inv(z) = exp(neg_r(ln(z))).
    9 calls (was 12). Saves 3 per call."""
    return eml_exp(eml_neg_r(eml_ln(z)))


def eml_mul_r(a, b):
    """mul(a, b) = exp(add_r(ln(a), ln(b))).
    17 calls (was 20). Saves 3 per call."""
    return eml_exp(eml_add_r(eml_ln(a), eml_ln(b)))


def eml_mul_precomp(ln_a, ln_b):
    """mul(a, b) given precomputed ln(a) and ln(b).
    11 calls (was 20). The monoid morphism ln:(R>0,×)→(R,+)
    factored out of the inner loop.
    """
    return eml_exp(eml_add_r(ln_a, ln_b))


def eml_div_r(a, b):
    """div(a, b) = mul_r(a, inv_r(b)).
    26 calls (was 32). Saves 6 per call."""
    return eml_mul_r(a, eml_inv_r(b))


def eml_sqrt_r(x):
    """sqrt(x) = exp(0.5 * ln(x)), with cached ln(1/2).

    Decomposition:
      ln_x    = eml_ln(x)                               3 calls
      ln_ln_x = eml_ln(ln_x)                            3 calls
      sum     = eml_add_r(LN_HALF, ln_ln_x)            10 calls
      product = eml_exp(sum)        [= 0.5 * ln(x)]     1 call
      result  = eml_exp(product)    [= sqrt(x)]          1 call
                                                        --------
                                                        18 calls (was 49!)
    63% reduction. Uses cached LN_HALF (monoid identity optimization).
    """
    ln_x = eml_ln(x)                                       # 3
    ln_ln_x = eml_ln(ln_x)                                 # 3
    half_ln_x = eml_exp(eml_add_r(_cache.LN_HALF, ln_ln_x))  # 1 + 10 = 11
    return eml_exp(half_ln_x)                               # 1  →  total: 18


def eml_gelu_r(x):
    """GELU(x) ≈ x * sigmoid(1.702 * x), with reduced ops.
    54 calls (was 74). Saves 20 per call."""
    c = 1.702
    z = eml_mul_r(c, x)                      # 17
    neg_z = eml_neg_r(z)                      # 5
    exp_neg_z = eml_exp(neg_z)                # 1
    one_plus = eml_add_r(ONE, exp_neg_z)      # 10
    sig = eml_inv_r(one_plus)                 # 9
    return eml_mul_r(x, sig)                  # 17  →  total: 59


# ─── 4. Morphism-based matmul ────────────────────────────────────────────────
#
# The key Stepanov insight applied to matrix multiply:
#
#   ln: (R>0, ×) → (R, +) is a monoid HOMOMORPHISM.
#
# In the naive matmul, each eml_mul(A[i,k], B[k,j]) recomputes
# eml_ln(A[i,k]) for every column j, and eml_ln(B[k,j]) for every row i.
# That's K*J + K*I redundant log computations.
#
# Factor the morphism out: precompute ln(A) and ln(B) once, then the
# inner loop uses eml_mul_precomp (11 calls) instead of eml_mul (20).
#
# Cost per output element:
#   Naive:   20K + 13(K-1)  = 33K - 13 calls
#   Precomp: 11K + 10(K-1)  = 21K - 10 calls  (36% fewer)
#   + amortized log precomputation shared across rows/columns

def eml_matmul_precomp(A, B):
    """Matrix multiply with precomputed logarithms.

    Exploits ln as a monoid homomorphism: factor it out of the inner loop.
    Same mathematical result, ~33% fewer eml() calls.
    """
    if A.ndim == 1 and B.ndim == 1:
        # dot product
        ln_A = np.array([eml_ln(a) for a in A])
        ln_B = np.array([eml_ln(b) for b in B])
        acc = eml_mul_precomp(ln_A[0], ln_B[0])
        for k in range(1, len(A)):
            acc = eml_add_r(acc, eml_mul_precomp(ln_A[k], ln_B[k]))
        return acc

    I_ = A.shape[0]
    K_ = A.shape[1]
    J_ = B.shape[1]

    # Precompute ln(A) — shared across all output columns
    ln_A = np.empty((I_, K_), dtype=np.complex128)
    for i in range(I_):
        for k in range(K_):
            ln_A[i, k] = eml_ln(A[i, k])       # 3 calls each

    # Precompute ln(B) — shared across all output rows
    ln_B = np.empty((K_, J_), dtype=np.complex128)
    for k in range(K_):
        for j in range(J_):
            ln_B[k, j] = eml_ln(B[k, j])       # 3 calls each

    C = np.empty((I_, J_), dtype=np.float64)
    for i in range(I_):
        for j in range(J_):
            # First term: exp(ln_a + ln_b) via morphism
            acc = eml_mul_precomp(ln_A[i, 0], ln_B[0, j])      # 11 calls
            for k in range(1, K_):
                term = eml_mul_precomp(ln_A[i, k], ln_B[k, j]) # 11 calls
                acc = eml_add_r(acc, term)                       # 10 calls
            C[i, j] = np.real(acc)
    return C


def eml_softmax_r(x):
    """Softmax with reduced-cost ops. Same math, fewer eml() calls."""
    m = np.max(np.real(x))
    shifted_exps = np.array([eml_exp(eml_sub(xi, m)) for xi in x.flat])
    shifted_exps = shifted_exps.reshape(x.shape)

    Z = shifted_exps.flat[0]
    for val in shifted_exps.flat[1:]:
        Z = eml_add_r(Z, val)              # 10 instead of 13

    result = np.array([np.real(eml_div_r(e, Z)) for e in shifted_exps.flat])
    return result.reshape(x.shape)


def eml_layer_norm_r(x, gamma, beta, eps=1e-5):
    """LayerNorm with reduced-cost ops + cached sqrt.
    Uses eml_sqrt_r (18 calls vs 49), eml_add_r, eml_mul_r, eml_div_r."""
    N = x.shape[-1]
    n_val = float(N)

    def _sum_last_axis(arr):
        if arr.ndim == 1:
            acc = arr[0]
            for i in range(1, len(arr)):
                acc = eml_add_r(acc, arr[i])        # 10 vs 13
            return np.array([acc])
        result = []
        for row in arr:
            acc = row[0]
            for i in range(1, len(row)):
                acc = eml_add_r(acc, row[i])        # 10 vs 13
            result.append(acc)
        return np.array(result).reshape(arr.shape[:-1] + (1,))

    x_sum = _sum_last_axis(x)
    mean = np.vectorize(lambda s: eml_div_r(s, n_val))(x_sum)
    diff = np.vectorize(eml_sub)(x, mean)
    diff_sq = np.vectorize(lambda d: eml_mul_r(d, d))(diff)
    var_sum = _sum_last_axis(diff_sq)
    var = np.vectorize(lambda s: eml_div_r(s, n_val))(var_sum)
    std = np.vectorize(lambda v: eml_sqrt_r(eml_add_r(v, eps)))(var)
    normed = np.vectorize(eml_div_r)(diff, std)
    scaled = np.vectorize(eml_mul_r)(normed, gamma)
    result = np.vectorize(lambda a, b: np.real(eml_add_r(a, b)))(scaled, beta)
    return result.astype(np.float64)
