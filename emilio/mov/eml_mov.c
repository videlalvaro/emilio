/*
 * eml_mov.c -- Emilio EML v2 inference engine (C89, MOV-only target)
 *
 * Complete single-file port of emilio's v2 inference path:
 *   - Software math (exp, log, sin, cos, sqrt) -- no libm dependency
 *   - EML fused algebraic ops (all arithmetic via exp/ln)
 *   - Sign+magnitude matmul kernel
 *   - RMSNorm, SiLU, Softmax
 *   - RoPE positional encoding
 *   - GQA attention with KV cache
 *   - SwiGLU FFN
 *   - .eml v2 model loader
 *   - BPE tokenizer (in eml_tokenizer.c)
 *   - Generation loop with greedy argmax
 *
 * Compiles with: movcc eml_mov.c eml_tokenizer.c -o emilio_mov
 * Or for testing: gcc -std=c89 -Wall -m32 eml_mov.c eml_tokenizer.c -o emilio_test -lm
 *
 * Reference: Odrzywołek (2026) arXiv:2603.21852
 */

#include "eml_mov.h"

/* ========================================================================
 * SECTION 1: Software Math (100% MOV-only -- no libm)
 *
 * These replace exp(), log(), sin(), cos(), sqrt() from <math.h>.
 * When compiled by the movfuscator, these become pure MOV sequences.
 * ======================================================================== */

double sw_fabs(double x)
{
    if (x < 0.0) return -x;
    return x;
}

double sw_floor(double x)
{
    /* C89-safe floor: truncate toward negative infinity */
    long i = (long)x;
    if (x < 0.0 && (double)i != x) return (double)(i - 1);
    return (double)i;
}

double sw_fmod(double x, double y)
{
    return x - sw_floor(x / y) * y;
}

/*
 * sw_exp(x): e^x via range reduction + polynomial approximation.
 *
 * Method: x = n*ln(2) + r where |r| <= ln(2)/2
 *         exp(x) = 2^n * exp(r)
 *         exp(r) approximated by minimax polynomial (degree 13).
 *
 * Handles overflow/underflow and special values.
 */
double sw_exp(double x)
{
    double r, result, term;
    long n;
    int i;
    /* overflow / underflow guards */
    if (x > 709.0) return EML_POS_INF;
    if (x < -745.0) return 0.0;

    /* range reduction: x = n*ln2 + r, |r| <= ln2/2 */
    n = (long)(x * EML_INV_LN2 + (x >= 0.0 ? 0.5 : -0.5));
    r = x - (double)n * EML_LN2;

    /* exp(r) via Taylor series -- 13 terms for ~1e-15 precision on |r|<=0.347 */
    result = 1.0;
    term = 1.0;
    for (i = 1; i <= 13; i++) {
        term *= r / (double)i;
        result += term;
    }

    /* multiply by 2^n via repeated squaring */
    if (n >= 0) {
        for (i = 0; i < (int)n; i++) result *= 2.0;
    } else {
        for (i = 0; i < (int)(-n); i++) result *= 0.5;
    }

    return result;
}

/*
 * sw_log(x): natural logarithm via range reduction + series.
 *
 * Method: x = m * 2^e  (extract mantissa/exponent conceptually)
 *         ln(x) = e*ln(2) + ln(m)  where 1 <= m < 2
 *         ln(m) via series: let t = (m-1)/(m+1)
 *         ln(m) = 2 * (t + t^3/3 + t^5/5 + ... )
 */
double sw_log(double x)
{
    double m, t, t2, sum, term;
    int e, i;

    if (x <= 0.0) {
        if (x == 0.0) return EML_NEG_INF;
        return 0.0 / 0.0;  /* NaN for negative */
    }
    if (x == EML_POS_INF) return EML_POS_INF;

    /* range reduction: extract exponent so that 1 <= m < 2 */
    e = 0;
    m = x;
    while (m >= 2.0) { m *= 0.5; e++; }
    while (m < 1.0)  { m *= 2.0; e--; }

    /* ln(m) via the series 2*sum(t^(2k+1)/(2k+1)) where t=(m-1)/(m+1) */
    t = (m - 1.0) / (m + 1.0);
    t2 = t * t;
    sum = t;
    term = t;
    for (i = 1; i <= 20; i++) {
        term *= t2;
        sum += term / (double)(2 * i + 1);
    }
    return (double)e * EML_LN2 + 2.0 * sum;
}

/*
 * sw_sqrt(x): Newton-Raphson iteration.
 * x_{n+1} = 0.5 * (x_n + S/x_n)
 */
double sw_sqrt(double x)
{
    double guess, prev;
    int i;
    if (x <= 0.0) return 0.0;

    /* initial guess */
    guess = x;
    if (guess > 1.0) guess = guess * 0.5;
    if (guess < 0.001) guess = 0.1;

    for (i = 0; i < 60; i++) {
        prev = guess;
        guess = 0.5 * (guess + x / guess);
        /* convergence check */
        if (guess == prev) break;
    }
    return guess;
}

/*
 * sw_sin(x) and sw_cos(x): range reduction + Taylor series.
 * Reduce x to [-pi, pi], then Taylor to degree 19.
 */
double sw_sin(double x)
{
    double x2, term, sum;
    int i;

    /* range reduce to [-pi, pi] */
    x = sw_fmod(x, 2.0 * EML_PI);
    if (x > EML_PI) x -= 2.0 * EML_PI;
    if (x < -EML_PI) x += 2.0 * EML_PI;

    /* Taylor series: sin(x) = x - x^3/3! + x^5/5! - ... */
    x2 = x * x;
    term = x;
    sum = x;
    for (i = 1; i <= 10; i++) {
        term *= -x2 / (double)(2 * i * (2 * i + 1));
        sum += term;
    }
    return sum;
}

double sw_cos(double x)
{
    double x2, term, sum;
    int i;

    x = sw_fmod(x, 2.0 * EML_PI);
    if (x > EML_PI) x -= 2.0 * EML_PI;
    if (x < -EML_PI) x += 2.0 * EML_PI;

    x2 = x * x;
    term = 1.0;
    sum = 1.0;
    for (i = 1; i <= 10; i++) {
        term *= -x2 / (double)(2 * i * (2 * i - 1));
        sum += term;
    }
    return sum;
}

/*
 * sw_atan2(y, x): four-quadrant arctangent.
 * Uses the identity: atan(t) = t - t^3/3 + t^5/5 - ... for |t| <= 1
 * For |t| > 1: atan(t) = pi/2 - atan(1/t)
 */
double sw_atan2(double y, double x)
{
    double a, t, t2, sum, term;
    int i, negate;

    if (x == 0.0 && y == 0.0) return 0.0;
    if (x == 0.0) return (y > 0.0) ? (EML_PI * 0.5) : -(EML_PI * 0.5);

    a = y / x;
    negate = 0;
    if (a < 0.0) { a = -a; negate = 1; }

    if (a > 1.0) {
        a = 1.0 / a;
        /* atan(original) = pi/2 - atan(1/original) */
        t = a;
        t2 = t * t;
        sum = t;
        term = t;
        for (i = 1; i <= 25; i++) {
            term *= -t2;
            sum += term / (double)(2 * i + 1);
        }
        sum = EML_PI * 0.5 - sum;
    } else {
        t = a;
        t2 = t * t;
        sum = t;
        term = t;
        for (i = 1; i <= 25; i++) {
            term *= -t2;
            sum += term / (double)(2 * i + 1);
        }
    }

    if (negate) sum = -sum;
    if (x < 0.0) {
        if (y >= 0.0) sum += EML_PI;
        else sum -= EML_PI;
    }
    return sum;
}

double sw_pow(double base, double e)
{
    if (base <= 0.0) return 0.0;
    return sw_exp(e * sw_log(base));
}


/* ========================================================================
 * SECTION 2: EML Core Operations (pure f64, fused algebraic forms)
 *
 * These are the fused scalar ops from eml_ops.rs.
 * The v2 sign+magnitude path means the hot inner loop only needs
 * real-valued exp() and log() -- no complex arithmetic.
 * ======================================================================== */

/* eml(x,y) = exp(x) - ln(y) -- the one and only gate */

/* exp(x): cost 1 exp */
double eml_exp_f(double x) { return sw_exp(x); }

/* ln(z): cost 1 log */
double eml_ln_f(double x)  { return sw_log(x); }

/* add(a,b) = a + b: cost 0 transcendentals */
double eml_add_f(double a, double b) { return a + b; }

/* sub(a,b) = a - b: cost 0 */
double eml_sub_f(double a, double b) { return a - b; }

/* neg(x) = -x: cost 0 */
double eml_neg_f(double x) { return -x; }

/* mul(a,b) = exp(ln(a) + ln(b)): cost 2 log + 1 exp */
double eml_mul_f(double a, double b)
{
    /* handle signs explicitly (log-domain multiply) */
    double sa, sb, mag;
    sa = (a >= 0.0) ? 1.0 : -1.0;
    sb = (b >= 0.0) ? 1.0 : -1.0;
    if (a == 0.0 || b == 0.0) return 0.0;
    mag = sw_exp(sw_log(sw_fabs(a)) + sw_log(sw_fabs(b)));
    return sa * sb * mag;
}

/* div(a,b) = exp(ln(a) - ln(b)): cost 2 log + 1 exp */
double eml_div_f(double a, double b)
{
    double sa, sb, mag;
    if (a == 0.0) return 0.0;
    sa = (a >= 0.0) ? 1.0 : -1.0;
    sb = (b >= 0.0) ? 1.0 : -1.0;
    mag = sw_exp(sw_log(sw_fabs(a)) - sw_log(sw_fabs(b)));
    return sa * sb * mag;
}

/* inv(z) = 1/z: cost 0 (direct division) */
double eml_inv_f(double x) { return 1.0 / x; }

/* sqrt(x) = exp(0.5 * ln(x)): cost 1 log + 1 exp */
double eml_sqrt_f(double x)
{
    if (x <= 0.0) return 0.0;
    return sw_exp(0.5 * sw_log(x));
}


/* ========================================================================
 * SECTION 3: Vector / Composite Operations
 * ======================================================================== */

void eml_add_vec(double *out, const double *a, const double *b, int n)
{
    int i;
    for (i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void eml_mul_vec(double *out, const double *a, const double *b, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        double sa, sb, mag;
        if (a[i] == 0.0 || b[i] == 0.0) { out[i] = 0.0; continue; }
        sa = (a[i] >= 0.0) ? 1.0 : -1.0;
        sb = (b[i] >= 0.0) ? 1.0 : -1.0;
        mag = sw_exp(sw_log(sw_fabs(a[i])) + sw_log(sw_fabs(b[i])));
        out[i] = sa * sb * mag;
    }
}


/* ========================================================================
 * SECTION 4: Sign+Magnitude Matmul Kernel
 *
 * Translates build_matmul_sm_precomp from eml_v2.rs.
 * Sequential only (no Rayon). The inner loop is pure f64:
 *   acc += exp(la_mag[k] + w_mag[k]) * la_sign[k] * w_sign[k]
 * ======================================================================== */

void build_matmul_sm(
    double *result,
    const double *a,
    const SmTensor *w,
    int rows, int inner, int cols)
{
    int i, j, k;
    double *la_mags;
    double *la_signs;
    int a_len;

    a_len = rows * inner;
    la_mags  = (double *)malloc((eml_size_t)a_len * sizeof(double));
    la_signs = (double *)malloc((eml_size_t)a_len * sizeof(double));

    /* Phase 1: precompute ln|a| and sign(a) */
    for (i = 0; i < a_len; i++) {
        if (a[i] > 0.0) {
            la_mags[i] = sw_log(a[i]);
            la_signs[i] = 1.0;
        } else if (a[i] < 0.0) {
            la_mags[i] = sw_log(-a[i]);
            la_signs[i] = -1.0;
        } else {
            la_mags[i] = EML_NEG_INF;
            la_signs[i] = 1.0;
        }
    }

    /* Phase 2: matmul with 4-wide unroll */
    for (i = 0; i < rows; i++) {
        int a_off = i * inner;
        for (j = 0; j < cols; j++) {
            int b_off = j * inner;
            double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
            int chunks = inner / 4;
            int remainder = inner % 4;
            int c;
            double e;

            for (c = 0; c < chunks; c++) {
                k = c * 4;
                e = sw_exp(la_mags[a_off+k] + w->magnitudes[b_off+k]);
                acc0 += e * la_signs[a_off+k] * w->signs[b_off+k];

                e = sw_exp(la_mags[a_off+k+1] + w->magnitudes[b_off+k+1]);
                acc1 += e * la_signs[a_off+k+1] * w->signs[b_off+k+1];

                e = sw_exp(la_mags[a_off+k+2] + w->magnitudes[b_off+k+2]);
                acc2 += e * la_signs[a_off+k+2] * w->signs[b_off+k+2];

                e = sw_exp(la_mags[a_off+k+3] + w->magnitudes[b_off+k+3]);
                acc3 += e * la_signs[a_off+k+3] * w->signs[b_off+k+3];
            }
            for (k = chunks * 4; k < chunks * 4 + remainder; k++) {
                e = sw_exp(la_mags[a_off+k] + w->magnitudes[b_off+k]);
                acc0 += e * la_signs[a_off+k] * w->signs[b_off+k];
            }
            result[i * cols + j] = acc0 + acc1 + acc2 + acc3;
        }
    }

    free(la_mags);
    free(la_signs);
}


/* ========================================================================
 * SECTION 5: RMSNorm, SiLU, Softmax
 * ======================================================================== */

/*
 * RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
 *
 * In EML: x^2 = exp(2*ln(x)), mean via sum/N, sqrt = exp(0.5*ln(...))
 * All ln(x) cached and reused in final: exp(ln(x) + ln(gamma) - ln(std))
 */
void eml_rms_norm(double *out, const double *x, const double *gamma,
                  int n, double eps)
{
    int i;
    double sq_sum, mean_sq, std_val, ln_std;
    double *ln_x;
    double nc;

    ln_x = (double *)malloc((eml_size_t)n * sizeof(double));
    nc = (double)n;

    /* cache ln(x_i) -- handles sign via abs+sign tracking */
    for (i = 0; i < n; i++) {
        if (x[i] > 0.0)      ln_x[i] = sw_log(x[i]);
        else if (x[i] < 0.0) ln_x[i] = sw_log(-x[i]);
        else                  ln_x[i] = EML_NEG_INF;
    }

    /* sum of x^2 = sum of exp(2*ln|x|) */
    sq_sum = 0.0;
    for (i = 0; i < n; i++) {
        sq_sum += sw_exp(2.0 * ln_x[i]);
    }

    /* mean(x^2) */
    mean_sq = sq_sum / nc;

    /* std = sqrt(mean_sq + eps) */
    std_val = sw_sqrt(mean_sq + eps);
    ln_std = sw_log(std_val);

    /* result_i = x_i * gamma_i / std = sign(x_i) * exp(ln|x_i| + ln|gamma_i| - ln_std) */
    for (i = 0; i < n; i++) {
        double sign_x = (x[i] >= 0.0) ? 1.0 : -1.0;
        double sign_g = (gamma[i] >= 0.0) ? 1.0 : -1.0;
        double ln_g = sw_log(sw_fabs(gamma[i]));
        if (x[i] == 0.0) { out[i] = 0.0; continue; }
        out[i] = sign_x * sign_g * sw_exp(ln_x[i] + ln_g - ln_std);
    }

    free(ln_x);
}

/*
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
void eml_silu_vec(double *out, const double *x, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        double sig = eml_inv_f(eml_add_f(1.0, sw_exp(eml_neg_f(x[i]))));
        out[i] = eml_mul_f(x[i], sig);
    }
}

/*
 * CSE softmax: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 */
void build_softmax_cse(double *out, const double *x, int n)
{
    int i;
    double m, z;

    /* find max */
    m = EML_NEG_INF;
    for (i = 0; i < n; i++) {
        if (x[i] > m) m = x[i];
    }

    /* exp(x_i - max) */
    for (i = 0; i < n; i++) {
        out[i] = sw_exp(x[i] - m);
    }

    /* sum */
    z = 0.0;
    for (i = 0; i < n; i++) z += out[i];

    /* normalize */
    if (z > 0.0) {
        double inv_z = 1.0 / z;
        for (i = 0; i < n; i++) out[i] *= inv_z;
    }
}


/* ========================================================================
 * SECTION 6: RoPE Cache
 * ======================================================================== */

RopeCache *rope_cache_new(int d_head, int max_len, double base)
{
    RopeCache *r;
    int d_half, pos, i;
    eml_size_t sz;

    r = (RopeCache *)malloc(sizeof(RopeCache));
    d_half = d_head / 2;
    r->d_half = d_half;
    r->max_len = max_len;

    sz = (eml_size_t)max_len * (eml_size_t)d_half * sizeof(double);
    r->cos_cache = (double *)malloc(sz);
    r->sin_cache = (double *)malloc(sz);

    for (pos = 0; pos < max_len; pos++) {
        for (i = 0; i < d_half; i++) {
            /* freq = 1 / base^(2i/d_head) = exp(-2i/d_head * ln(base)) */
            double exponent = 2.0 * (double)i / (double)d_head;
            double ln_base = sw_log(base);
            double freq = sw_exp(-(exponent * ln_base));
            double angle = (double)pos * freq;
            int idx = pos * d_half + i;
            r->cos_cache[idx] = sw_cos(angle);
            r->sin_cache[idx] = sw_sin(angle);
        }
    }
    return r;
}

void rope_cache_free(RopeCache *r)
{
    if (r) {
        free(r->cos_cache);
        free(r->sin_cache);
        free(r);
    }
}

void rope_apply(const RopeCache *r, double *x, int pos)
{
    int i;
    int d_half = r->d_half;
    for (i = 0; i < d_half; i++) {
        double cos_v = r->cos_cache[pos * d_half + i];
        double sin_v = r->sin_cache[pos * d_half + i];
        double x0 = x[i];
        double x1 = x[i + d_half];
        /* x'[i]        = x[i]*cos - x[i+d_half]*sin */
        /* x'[i+d_half] = x[i]*sin + x[i+d_half]*cos */
        x[i]          = eml_sub_f(eml_mul_f(x0, cos_v), eml_mul_f(x1, sin_v));
        x[i + d_half] = eml_add_f(eml_mul_f(x0, sin_v), eml_mul_f(x1, cos_v));
    }
}


/* ========================================================================
 * SECTION 7: KV Cache
 * ======================================================================== */

KVCache *kv_cache_new(const EmlConfig *cfg, int max_len)
{
    KVCache *kv;
    int kv_dim, i;

    kv = (KVCache *)malloc(sizeof(KVCache));
    kv->n_layers = cfg->n_layers;
    kv->layers = (LayerKVCache *)malloc((eml_size_t)cfg->n_layers * sizeof(LayerKVCache));

    kv_dim = cfg->n_kv_heads * cfg->d_head;

    for (i = 0; i < cfg->n_layers; i++) {
        kv->layers[i].k = (double *)calloc((eml_size_t)(max_len * kv_dim), sizeof(double));
        kv->layers[i].v = (double *)calloc((eml_size_t)(max_len * kv_dim), sizeof(double));
        kv->layers[i].len = 0;
        kv->layers[i].max_len = max_len;
        kv->layers[i].kv_dim = kv_dim;
    }
    return kv;
}

void kv_cache_free(KVCache *kv)
{
    int i;
    if (!kv) return;
    for (i = 0; i < kv->n_layers; i++) {
        free(kv->layers[i].k);
        free(kv->layers[i].v);
    }
    free(kv->layers);
    free(kv);
}

static void kv_append(LayerKVCache *lkv, const double *k, const double *v)
{
    int off;
    if (lkv->len >= lkv->max_len) {
        fprintf(stderr, "Error: KV cache overflow (len=%d, max=%d)\n",
                lkv->len, lkv->max_len);
        return;
    }
    off = lkv->len * lkv->kv_dim;
    memcpy(lkv->k + off, k, (eml_size_t)lkv->kv_dim * sizeof(double));
    memcpy(lkv->v + off, v, (eml_size_t)lkv->kv_dim * sizeof(double));
    lkv->len++;
}


/* ========================================================================
 * SECTION 8: GQA Attention
 * ======================================================================== */

static void v2_gqa_attention_one(
    double *attn_out,
    const double *x,
    const V2LayerWeights *layer,
    const EmlConfig *cfg,
    const RopeCache *rope,
    int pos,
    LayerKVCache *kv)
{
    int d = cfg->d_model;
    int d_head = cfg->d_head;
    int n_heads = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int heads_per_kv = n_heads / n_kv_heads;
    int q_dim = n_heads * d_head;
    int kv_dim = n_kv_heads * d_head;
    int qkv_dim = q_dim + kv_dim + kv_dim;
    int h, j, dd, t;
    double scale;
    double *qkv, *q, *k_new, *v_new;
    double *scores, *attn_w;

    qkv = (double *)malloc((eml_size_t)qkv_dim * sizeof(double));

    /* Fused QKV matmul */
    build_matmul_sm(qkv, x, &layer->sm_qkv, 1, d, qkv_dim);

    q     = (double *)malloc((eml_size_t)q_dim * sizeof(double));
    k_new = (double *)malloc((eml_size_t)kv_dim * sizeof(double));
    v_new = (double *)malloc((eml_size_t)kv_dim * sizeof(double));

    memcpy(q,     qkv,                   (eml_size_t)q_dim * sizeof(double));
    memcpy(k_new, qkv + q_dim,           (eml_size_t)kv_dim * sizeof(double));
    memcpy(v_new, qkv + q_dim + kv_dim,  (eml_size_t)kv_dim * sizeof(double));
    free(qkv);

    /* Add bias */
    for (j = 0; j < q_dim; j++)  q[j] += layer->q_bias[j];
    for (j = 0; j < kv_dim; j++) k_new[j] += layer->k_bias[j];
    for (j = 0; j < kv_dim; j++) v_new[j] += layer->v_bias[j];

    /* Apply RoPE */
    for (h = 0; h < n_heads; h++)
        rope_apply(rope, q + h * d_head, pos);
    for (h = 0; h < n_kv_heads; h++)
        rope_apply(rope, k_new + h * d_head, pos);

    /* Store in KV cache */
    kv_append(kv, k_new, v_new);
    t = kv->len;

    free(k_new);
    free(v_new);

    /* Attention: single query against all cached K,V */
    scale = eml_inv_f(eml_sqrt_f((double)d_head));
    memset(attn_out, 0, (eml_size_t)d * sizeof(double));

    scores = (double *)malloc((eml_size_t)t * sizeof(double));
    attn_w = (double *)malloc((eml_size_t)t * sizeof(double));

    for (h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;

        /* Compute attention scores */
        for (j = 0; j < t; j++) {
            double dot = 0.0;
            for (dd = 0; dd < d_head; dd++) {
                double qv = q[h * d_head + dd];
                double kval = kv->k[j * kv_dim + kv_h * d_head + dd];
                dot += eml_mul_f(qv, kval);
            }
            scores[j] = eml_mul_f(dot, scale);
        }

        /* Softmax */
        build_softmax_cse(attn_w, scores, t);

        /* Weighted sum over cached V */
        for (dd = 0; dd < d_head; dd++) {
            double acc = 0.0;
            for (j = 0; j < t; j++) {
                double vv = kv->v[j * kv_dim + kv_h * d_head + dd];
                acc += eml_mul_f(attn_w[j], vv);
            }
            attn_out[h * d_head + dd] = acc;
        }
    }

    free(scores);
    free(attn_w);
    free(q);

    /* Output projection */
    {
        double *o_out = (double *)malloc((eml_size_t)d * sizeof(double));
        build_matmul_sm(o_out, attn_out, &layer->sm_o, 1, q_dim, d);
        memcpy(attn_out, o_out, (eml_size_t)d * sizeof(double));
        free(o_out);
    }
}


/* ========================================================================
 * SECTION 9: SwiGLU FFN
 * ======================================================================== */

static void v2_swiglu_ffn(
    double *ffn_out,
    const double *x,
    const V2LayerWeights *layer,
    const EmlConfig *cfg)
{
    int d = cfg->d_model;
    int d_ff = cfg->d_ff;
    double *gate_up, *gate_act, *hidden;

    gate_up = (double *)malloc((eml_size_t)(2 * d_ff) * sizeof(double));

    /* Fused gate+up matmul */
    build_matmul_sm(gate_up, x, &layer->sm_gate_up, 1, d, 2 * d_ff);

    gate_act = (double *)malloc((eml_size_t)d_ff * sizeof(double));
    hidden   = (double *)malloc((eml_size_t)d_ff * sizeof(double));

    /* SiLU(gate) */
    eml_silu_vec(gate_act, gate_up, d_ff);

    /* gate_act * up */
    eml_mul_vec(hidden, gate_act, gate_up + d_ff, d_ff);

    free(gate_up);
    free(gate_act);

    /* Down projection */
    build_matmul_sm(ffn_out, hidden, &layer->sm_down, 1, d_ff, d);
    free(hidden);
}


/* ========================================================================
 * SECTION 10: Transformer Layer + Forward Pass + Generation
 * ======================================================================== */

static void v2_transformer_layer_one(
    double *x_out,     /* output: d_model */
    const double *x,   /* input: d_model */
    const V2LayerWeights *layer,
    const EmlConfig *cfg,
    const RopeCache *rope,
    int pos,
    LayerKVCache *kv)
{
    int d = cfg->d_model;
    double *normed, *attn_out, *x2, *normed2, *ffn_out;

    normed   = (double *)malloc((eml_size_t)d * sizeof(double));
    attn_out = (double *)malloc((eml_size_t)d * sizeof(double));
    x2       = (double *)malloc((eml_size_t)d * sizeof(double));
    normed2  = (double *)malloc((eml_size_t)d * sizeof(double));
    ffn_out  = (double *)malloc((eml_size_t)d * sizeof(double));

    /* Pre-attention RMSNorm */
    eml_rms_norm(normed, x, layer->attn_norm, d, cfg->rms_norm_eps);

    /* GQA Attention */
    v2_gqa_attention_one(attn_out, normed, layer, cfg, rope, pos, kv);

    /* Residual */
    eml_add_vec(x2, x, attn_out, d);

    /* Pre-FFN RMSNorm */
    eml_rms_norm(normed2, x2, layer->ffn_norm, d, cfg->rms_norm_eps);

    /* SwiGLU FFN */
    v2_swiglu_ffn(ffn_out, normed2, layer, cfg);

    /* Residual */
    eml_add_vec(x_out, x2, ffn_out, d);

    free(normed);
    free(attn_out);
    free(x2);
    free(normed2);
    free(ffn_out);
}

void v2_forward_one(
    double *logits,
    int token_id, int pos,
    const V2ModelWeights *w,
    const RopeCache *rope,
    KVCache *kv)
{
    const EmlConfig *cfg = &w->config;
    int d = cfg->d_model;
    int layer_idx;
    double *x, *x_new, *normed, *tmp;

    x     = (double *)malloc((eml_size_t)d * sizeof(double));
    x_new = (double *)malloc((eml_size_t)d * sizeof(double));
    normed = (double *)malloc((eml_size_t)d * sizeof(double));

    /* 1. Token embedding */
    memcpy(x, w->token_embd + token_id * d, (eml_size_t)d * sizeof(double));

    /* 2. Transformer layers */
    for (layer_idx = 0; layer_idx < cfg->n_layers; layer_idx++) {
        v2_transformer_layer_one(x_new, x, &w->layers[layer_idx],
                                 cfg, rope, pos, &kv->layers[layer_idx]);
        /* swap x and x_new */
        tmp = x; x = x_new; x_new = tmp;
    }

    /* 3. Final RMSNorm */
    eml_rms_norm(normed, x, w->output_norm, d, cfg->rms_norm_eps);

    /* 4. LM head: (1, d_model) @ (d_model, vocab) */
    build_matmul_sm(logits, normed, &w->sm_output, 1, d, cfg->vocab_size);

    free(x);
    free(x_new);
    free(normed);
}

static int argmax(const double *arr, int n)
{
    int i, best = 0;
    double best_val = arr[0];
    for (i = 1; i < n; i++) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best = i;
        }
    }
    return best;
}

int *v2_generate(
    const int *prompt, int prompt_len,
    const V2ModelWeights *w,
    const RopeCache *rope,
    int max_new,
    int *out_len)
{
    const EmlConfig *cfg = &w->config;
    int max_len, total_max, step, next_id;
    int *ids;
    int ids_len;
    double *logits;
    KVCache *kv;

    max_len = cfg->max_seq_len;
    if (prompt_len + max_new + 16 < max_len)
        max_len = prompt_len + max_new + 16;

    kv = kv_cache_new(cfg, max_len);
    logits = (double *)malloc((eml_size_t)cfg->vocab_size * sizeof(double));

    total_max = prompt_len + max_new;
    ids = (int *)malloc((eml_size_t)total_max * sizeof(int));
    memcpy(ids, prompt, (eml_size_t)prompt_len * sizeof(int));
    ids_len = prompt_len;

    /* Prefill */
    fprintf(stderr, "  Prefilling %d prompt tokens...\n", prompt_len);
    {
        int i;
        for (i = 0; i < prompt_len; i++) {
            v2_forward_one(logits, prompt[i], i, w, rope, kv);
            if ((i + 1) % 10 == 0 || i == prompt_len - 1) {
                fprintf(stderr, "\r  Prefilled %d/%d", i + 1, prompt_len);
            }
        }
        fprintf(stderr, "\n");
    }

    /* Decode */
    for (step = 0; step < max_new; step++) {
        if (step > 0) {
            int last_tok = ids[ids_len - 1];
            int pos = ids_len - 1;
            v2_forward_one(logits, last_tok, pos, w, rope, kv);
        }

        next_id = argmax(logits, cfg->vocab_size);
        ids[ids_len++] = next_id;

        fprintf(stderr, "\r  Generated %d/%d tokens", step + 1, max_new);

        if (next_id == EML_EOT_ID || next_id == EML_EOS_ID) break;
    }
    fprintf(stderr, "\n");

    free(logits);
    kv_cache_free(kv);

    *out_len = ids_len;
    return ids;
}


/* ========================================================================
 * SECTION 11: .eml v2 File Loader
 * ======================================================================== */

static int read_u32(EML_FILE *f, unsigned long *val)
{
    unsigned char buf[4];
    if (fread(buf, 1, 4, f) != 4) return -1;
    *val = (unsigned long)buf[0]
         | ((unsigned long)buf[1] << 8)
         | ((unsigned long)buf[2] << 16)
         | ((unsigned long)buf[3] << 24);
    return 0;
}

static int read_u64(EML_FILE *f, unsigned long *val)
{
    unsigned char buf[8];
    if (fread(buf, 1, 8, f) != 8) return -1;
    /* On 32-bit, unsigned long is 32 bits. We truncate high bits.
       For element counts this is safe (< 4 billion elements). */
    *val = (unsigned long)buf[0]
         | ((unsigned long)buf[1] << 8)
         | ((unsigned long)buf[2] << 16)
         | ((unsigned long)buf[3] << 24);
    return 0;
}

static int read_f64(EML_FILE *f, double *val)
{
    unsigned char buf[8];
    if (fread(buf, 1, 8, f) != 8) return -1;
    memcpy(val, buf, 8);
    return 0;
}

static double *read_f64_array(EML_FILE *f, int *out_len)
{
    unsigned long len;
    double *data;
    int i;
    if (read_u64(f, &len) != 0) return EML_NULL;
    *out_len = (int)len;
    data = (double *)malloc((eml_size_t)len * sizeof(double));
    for (i = 0; i < (int)len; i++) {
        if (read_f64(f, &data[i]) != 0) { free(data); return EML_NULL; }
    }
    return data;
}

static char *read_string(EML_FILE *f)
{
    unsigned long slen;
    char *s;
    if (read_u32(f, &slen) != 0) return EML_NULL;
    s = (char *)malloc((eml_size_t)(slen + 1));
    if (slen > 0) {
        if (fread(s, 1, (eml_size_t)slen, f) != (eml_size_t)slen) {
            free(s); return EML_NULL;
        }
    }
    s[slen] = '\0';
    return s;
}

static int read_sm_tensor(EML_FILE *f, SmTensor *t)
{
    unsigned long len, packed_len;
    unsigned char *packed;
    int i;

    if (read_u64(f, &len) != 0) return -1;
    t->len = (int)len;
    t->magnitudes = (double *)malloc((eml_size_t)len * sizeof(double));
    for (i = 0; i < (int)len; i++) {
        if (read_f64(f, &t->magnitudes[i]) != 0) return -1;
    }

    if (read_u32(f, &packed_len) != 0) return -1;
    packed = (unsigned char *)malloc((eml_size_t)packed_len);
    if (fread(packed, 1, (eml_size_t)packed_len, f) != (eml_size_t)packed_len) {
        free(packed); return -1;
    }

    /* Unpack signs from bitmap */
    t->signs = (double *)malloc((eml_size_t)len * sizeof(double));
    for (i = 0; i < (int)len; i++) {
        int bit = (packed[i / 8] >> (i % 8)) & 1;
        t->signs[i] = (bit == 1) ? -1.0 : 1.0;
    }
    free(packed);
    return 0;
}

static void skip_exec_graph(EML_FILE *f)
{
    /* Read and discard execution graph (not needed at runtime) */
    unsigned long count;
    int i;
    unsigned char tag;

    if (read_u32(f, &count) != 0) return;
    for (i = 0; i < (int)count; i++) {
        if (fread(&tag, 1, 1, f) != 1) return;
        switch (tag) {
            case 6: { unsigned long dummy; read_u32(f, &dummy); read_u32(f, &dummy); break; }
            case 11: break; /* LmHead: no payload */
            default: { unsigned long dummy; read_u32(f, &dummy); break; }
        }
    }
}

V2ModelWeights *load_eml_v2(const char *path, Tokenizer *tok)
{
    EML_FILE *f;
    unsigned char magic[4];
    unsigned long version, val;
    V2ModelWeights *w;
    int i, dummy_len;
    unsigned long vocab_size_file, merges_count, bos_id, eos_id;

    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", path);
        exit(1);
    }

    /* Magic: "EML2" */
    if (fread(magic, 1, 4, f) != 4 ||
        magic[0] != 'E' || magic[1] != 'M' || magic[2] != 'L' || magic[3] != '2') {
        fprintf(stderr, "Error: not an EML v2 file\n");
        exit(1);
    }
    read_u32(f, &version);
    if (version != 2) {
        fprintf(stderr, "Error: unsupported EML version %lu\n", version);
        exit(1);
    }

    w = (V2ModelWeights *)calloc(1, sizeof(V2ModelWeights));

    /* Config */
    read_u32(f, &val); w->config.vocab_size   = (int)val;
    read_u32(f, &val); w->config.n_layers     = (int)val;
    read_u32(f, &val); w->config.n_heads      = (int)val;
    read_u32(f, &val); w->config.n_kv_heads   = (int)val;
    read_u32(f, &val); w->config.d_model      = (int)val;
    read_u32(f, &val); w->config.d_ff         = (int)val;
    read_f64(f, &w->config.rope_freq_base);
    read_f64(f, &w->config.rms_norm_eps);
    read_u32(f, &val); w->config.max_seq_len  = (int)val;
    read_u32(f, &val); w->config.d_head       = (int)val;

    /* Sparsity stats (skip) */
    { unsigned long sp; double spt; read_u64(f, &sp); read_u64(f, &sp); read_f64(f, &spt); }

    /* Tokenizer */
    read_u32(f, &vocab_size_file);
    read_u32(f, &merges_count);
    read_u32(f, &bos_id);
    read_u32(f, &eos_id);

    tok->vocab_size = (int)vocab_size_file;
    tok->bos_id = (int)bos_id;
    tok->eos_id = (int)eos_id;
    tok->vocab = (char **)malloc((eml_size_t)vocab_size_file * sizeof(char *));
    for (i = 0; i < (int)vocab_size_file; i++) {
        tok->vocab[i] = read_string(f);
        if (!tok->vocab[i]) { fprintf(stderr, "Error reading vocab\n"); exit(1); }
    }

    tok->merges_count = (int)merges_count;
    tok->merges = (MergeEntry *)malloc((eml_size_t)merges_count * sizeof(MergeEntry));
    for (i = 0; i < (int)merges_count; i++) {
        tok->merges[i].left  = read_string(f);
        tok->merges[i].right = read_string(f);
        tok->merges[i].rank  = i;
    }

    tokenizer_init_tables(tok);

    /* Execution graph (skip -- not needed at runtime) */
    skip_exec_graph(f);

    /* Global weights */
    w->token_embd  = read_f64_array(f, &dummy_len);
    w->output_norm = read_f64_array(f, &dummy_len);
    if (read_sm_tensor(f, &w->sm_output) != 0) {
        fprintf(stderr, "Error reading sm_output\n"); exit(1);
    }

    /* Per-layer weights */
    w->n_layers = w->config.n_layers;
    w->layers = (V2LayerWeights *)calloc((eml_size_t)w->n_layers, sizeof(V2LayerWeights));

    for (i = 0; i < w->n_layers; i++) {
        V2LayerWeights *lw = &w->layers[i];
        if (read_sm_tensor(f, &lw->sm_qkv)     != 0 ||
            read_sm_tensor(f, &lw->sm_o)        != 0 ||
            read_sm_tensor(f, &lw->sm_gate_up)  != 0 ||
            read_sm_tensor(f, &lw->sm_down)     != 0) {
            fprintf(stderr, "Error reading layer %d sm tensors\n", i); exit(1);
        }
        lw->q_bias    = read_f64_array(f, &dummy_len);
        lw->k_bias    = read_f64_array(f, &dummy_len);
        lw->v_bias    = read_f64_array(f, &dummy_len);
        lw->attn_norm = read_f64_array(f, &dummy_len);
        lw->ffn_norm  = read_f64_array(f, &dummy_len);

        if ((i + 1) % 8 == 0 || i == w->n_layers - 1) {
            fprintf(stderr, "  Loaded layer %d/%d\n", i + 1, w->n_layers);
        }
    }

    fclose(f);
    return w;
}

void free_v2_weights(V2ModelWeights *w)
{
    int i;
    if (!w) return;
    free(w->token_embd);
    free(w->output_norm);
    free(w->sm_output.magnitudes);
    free(w->sm_output.signs);
    for (i = 0; i < w->n_layers; i++) {
        V2LayerWeights *lw = &w->layers[i];
        free(lw->sm_qkv.magnitudes); free(lw->sm_qkv.signs);
        free(lw->sm_o.magnitudes);   free(lw->sm_o.signs);
        free(lw->sm_gate_up.magnitudes); free(lw->sm_gate_up.signs);
        free(lw->sm_down.magnitudes); free(lw->sm_down.signs);
        free(lw->q_bias); free(lw->k_bias); free(lw->v_bias);
        free(lw->attn_norm); free(lw->ffn_norm);
    }
    free(w->layers);
    free(w);
}


/* ========================================================================
 * SECTION 12: main()
 * Guarded by EML_NO_MAIN so the test harness can provide its own main().
 * ======================================================================== */

#ifndef EML_NO_MAIN
int main(int argc, char **argv)
{
    const char *model_path;
    const char *prompt_text;
    int max_tokens;
    V2ModelWeights *weights;
    Tokenizer tok;
    RopeCache *rope;
    int *prompt_ids, *gen_ids;
    int prompt_len, gen_len;
    char *output_text;

    if (argc < 3) {
        printf("Usage: %s <model.eml> <prompt> [max_tokens]\n", argv[0]);
        printf("\nemilio MOV-only inference engine\n");
        printf("  Compiled with M/o/Vfuscator -- all instructions are MOV.\n");
        printf("  Reference: Odrzywołek (2026) arXiv:2603.21852\n");
        return 1;
    }

    model_path = argv[1];
    prompt_text = argv[2];
    max_tokens = (argc >= 4) ? 0 : 64;

    /* Parse max_tokens from argv[3] if provided */
    if (argc >= 4) {
        const char *s = argv[3];
        int v = 0;
        while (*s >= '0' && *s <= '9') { v = v * 10 + (*s - '0'); s++; }
        max_tokens = v;
        if (max_tokens <= 0) max_tokens = 64;
    }

    fprintf(stderr, "emilio MOV-only inference engine\n");
    fprintf(stderr, "Loading model: %s\n", model_path);

    /* Load model + tokenizer */
    memset(&tok, 0, sizeof(Tokenizer));
    weights = load_eml_v2(model_path, &tok);

    fprintf(stderr, "Model loaded:\n");
    fprintf(stderr, "  vocab_size: %d\n", weights->config.vocab_size);
    fprintf(stderr, "  n_layers:   %d\n", weights->config.n_layers);
    fprintf(stderr, "  d_model:    %d\n", weights->config.d_model);
    fprintf(stderr, "  n_heads:    %d\n", weights->config.n_heads);
    fprintf(stderr, "  d_head:     %d\n", weights->config.d_head);

    /* Build RoPE cache */
    fprintf(stderr, "Building RoPE cache...\n");
    rope = rope_cache_new(weights->config.d_head,
                          weights->config.max_seq_len,
                          weights->config.rope_freq_base);

    /* Encode prompt using ChatML template */
    fprintf(stderr, "Encoding prompt: \"%s\"\n", prompt_text);
    prompt_ids = tokenizer_encode_chat(&tok, prompt_text, &prompt_len);
    fprintf(stderr, "  %d tokens\n", prompt_len);

    /* Generate */
    fprintf(stderr, "Generating up to %d tokens...\n", max_tokens);
    gen_ids = v2_generate(prompt_ids, prompt_len, weights, rope,
                          max_tokens, &gen_len);

    /* Decode and print */
    output_text = tokenizer_decode(&tok, gen_ids + prompt_len,
                                   gen_len - prompt_len);
    printf("%s\n", output_text);

    /* Cleanup */
    free(output_text);
    free(gen_ids);
    free(prompt_ids);
    rope_cache_free(rope);
    free_v2_weights(weights);
    tokenizer_free(&tok);

    return 0;
}
#endif /* EML_NO_MAIN */
