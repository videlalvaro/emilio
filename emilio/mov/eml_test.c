/*
 * eml_test.c -- Self-test for the MOV-only emilio inference engine
 *
 * Validates:
 *   1. Software math functions (exp, log, sin, cos, sqrt)
 *   2. EML fused operations (mul, div, sqrt)
 *   3. Matmul kernel (sign+magnitude)
 *   4. Softmax
 *   5. RMSNorm
 *   6. SiLU
 *
 * Compile: gcc -std=c89 -Wall -o eml_test eml_test.c eml_mov.c eml_tokenizer.c -lm
 * Or:      movcc eml_test.c eml_mov.c eml_tokenizer.c -o eml_test_mov
 *
 * Exit code 0 = all tests pass.
 */

#include "eml_mov.h"

static int n_pass = 0;
static int n_fail = 0;

static double ref_fabs(double x) { return x < 0 ? -x : x; }

static void check(const char *name, double got, double expected, double tol)
{
    double err = ref_fabs(got - expected);
    if (err < tol) {
        n_pass++;
    } else {
        printf("FAIL: %s: got %.15g, expected %.15g, err=%.2e (tol=%.0e)\n",
               name, got, expected, err, tol);
        n_fail++;
    }
}

static void test_sw_math(void)
{
    printf("--- Software math ---\n");

    check("fabs(-3.5)", sw_fabs(-3.5), 3.5, 1e-15);
    check("fabs(2.0)",  sw_fabs(2.0),  2.0, 1e-15);

    check("exp(0)",   sw_exp(0.0),  1.0, 1e-12);
    check("exp(1)",   sw_exp(1.0),  EML_E, 1e-12);
    check("exp(-1)",  sw_exp(-1.0), 1.0/EML_E, 1e-12);
    check("exp(2)",   sw_exp(2.0),  EML_E*EML_E, 1e-10);
    check("exp(10)",  sw_exp(10.0), 22026.4657948067, 1e-6);
    check("exp(-10)", sw_exp(-10.0), 4.53999297625e-05, 1e-14);

    check("log(1)",   sw_log(1.0),  0.0, 1e-12);
    check("log(e)",   sw_log(EML_E), 1.0, 1e-12);
    check("log(2)",   sw_log(2.0),  EML_LN2, 1e-12);
    check("log(10)",  sw_log(10.0), 2.302585093, 1e-9);
    check("log(0.5)", sw_log(0.5), -EML_LN2, 1e-12);

    check("sqrt(4)",  sw_sqrt(4.0),  2.0, 1e-12);
    check("sqrt(2)",  sw_sqrt(2.0),  1.41421356237, 1e-10);
    check("sqrt(9)",  sw_sqrt(9.0),  3.0, 1e-12);
    check("sqrt(0.25)", sw_sqrt(0.25), 0.5, 1e-12);

    check("sin(0)",      sw_sin(0.0),         0.0, 1e-12);
    check("sin(pi/2)",   sw_sin(EML_PI/2.0),  1.0, 1e-12);
    check("sin(pi)",     sw_sin(EML_PI),       0.0, 1e-10);
    check("sin(-pi/2)",  sw_sin(-EML_PI/2.0), -1.0, 1e-12);

    check("cos(0)",      sw_cos(0.0),         1.0, 1e-12);
    check("cos(pi/2)",   sw_cos(EML_PI/2.0),  0.0, 1e-12);
    check("cos(pi)",     sw_cos(EML_PI),      -1.0, 1e-10);

    /* exp(log(x)) == x roundtrip */
    check("exp(log(42))", sw_exp(sw_log(42.0)), 42.0, 1e-9);
    check("exp(log(0.001))", sw_exp(sw_log(0.001)), 0.001, 1e-12);
}

static void test_eml_ops(void)
{
    printf("--- EML fused ops ---\n");

    check("add(1,2)",  eml_add_f(1.0, 2.0), 3.0, 1e-15);
    check("sub(5,3)",  eml_sub_f(5.0, 3.0), 2.0, 1e-15);
    check("neg(5)",    eml_neg_f(5.0), -5.0, 1e-15);
    check("neg(-3)",   eml_neg_f(-3.0), 3.0, 1e-15);

    check("mul(3,5)",  eml_mul_f(3.0, 5.0),  15.0, 1e-8);
    check("mul(0.5,4)", eml_mul_f(0.5, 4.0), 2.0, 1e-8);
    check("mul(-1,5)", eml_mul_f(-1.0, 5.0), -5.0, 1e-8);
    check("mul(-2,-3)", eml_mul_f(-2.0, -3.0), 6.0, 1e-8);
    check("mul(0,5)",  eml_mul_f(0.0, 5.0),  0.0, 1e-15);

    check("div(6,3)",  eml_div_f(6.0, 3.0), 2.0, 1e-8);
    check("div(1,4)",  eml_div_f(1.0, 4.0), 0.25, 1e-8);
    check("div(-6,3)", eml_div_f(-6.0, 3.0), -2.0, 1e-8);

    check("inv(2)",  eml_inv_f(2.0), 0.5, 1e-15);
    check("inv(4)",  eml_inv_f(4.0), 0.25, 1e-15);

    check("sqrt(9)",  eml_sqrt_f(9.0), 3.0, 1e-8);
    check("sqrt(2)",  eml_sqrt_f(2.0), 1.41421356237, 1e-8);
}

static void test_matmul(void)
{
    /* [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]] */
    double a[4] = {1.0, 2.0, 3.0, 4.0};
    double result[4];
    SmTensor w;
    double w_mag[4], w_sign[4];

    printf("--- Matmul (sign+mag) ---\n");

    /* Prepare sign+magnitude for B = [[5,6],[7,8]] transposed to (cols,inner) = (2,2)
     * B^T = [[5,7],[6,8]]
     * For movfuscator matmul, w layout is (cols, inner):
     *   w[0] = {ln(5), +1}, w[1] = {ln(7), +1}  (col 0)
     *   w[2] = {ln(6), +1}, w[3] = {ln(8), +1}  (col 1)
     */
    w_mag[0] = sw_log(5.0); w_sign[0] = 1.0;
    w_mag[1] = sw_log(7.0); w_sign[1] = 1.0;
    w_mag[2] = sw_log(6.0); w_sign[2] = 1.0;
    w_mag[3] = sw_log(8.0); w_sign[3] = 1.0;

    w.magnitudes = w_mag;
    w.signs = w_sign;
    w.len = 4;

    build_matmul_sm(result, a, &w, 2, 2, 2);

    check("matmul[0,0]", result[0], 19.0, 1e-5);
    check("matmul[0,1]", result[1], 22.0, 1e-5);
    check("matmul[1,0]", result[2], 43.0, 1e-5);
    check("matmul[1,1]", result[3], 50.0, 1e-5);
}

static void test_softmax(void)
{
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double out[4];
    double sum = 0.0;

    printf("--- Softmax ---\n");

    build_softmax_cse(out, x, 4);

    sum = out[0] + out[1] + out[2] + out[3];
    check("softmax sum", sum, 1.0, 1e-9);
    /* monotonic: out[3] > out[2] > out[1] > out[0] */
    check("softmax monotonic 3>2", (out[3] > out[2]) ? 1.0 : 0.0, 1.0, 0.5);
    check("softmax monotonic 2>1", (out[2] > out[1]) ? 1.0 : 0.0, 1.0, 0.5);
    check("softmax monotonic 1>0", (out[1] > out[0]) ? 1.0 : 0.0, 1.0, 0.5);
}

static void test_rms_norm(void)
{
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double gamma[4] = {1.0, 1.0, 1.0, 1.0};
    double out[4];
    double expected_rms;

    printf("--- RMSNorm ---\n");

    eml_rms_norm(out, x, gamma, 4, 1e-6);

    /* RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386... */
    expected_rms = sw_sqrt(7.5);
    /* Normalized values: x[i] / rms */
    check("rms_norm[0]", out[0], 1.0 / expected_rms, 1e-6);
    check("rms_norm[1]", out[1], 2.0 / expected_rms, 1e-6);
    check("rms_norm[2]", out[2], 3.0 / expected_rms, 1e-6);
    check("rms_norm[3]", out[3], 4.0 / expected_rms, 1e-6);
}

static void test_silu(void)
{
    double x[3] = {0.0, 1.0, -1.0};
    double out[3];

    printf("--- SiLU ---\n");

    eml_silu_vec(out, x, 3);

    /* silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0 */
    check("silu(0)", out[0], 0.0, 1e-8);
    /* silu(1) = 1 * sigmoid(1) = 1 * 0.7310585... = 0.7310585 */
    check("silu(1)", out[1], 0.7310585786, 1e-6);
    /* silu(-1) = -1 * sigmoid(-1) = -1 * 0.2689414... = -0.2689414 */
    check("silu(-1)", out[2], -0.2689414214, 1e-6);
}

static void test_rope(void)
{
    RopeCache *r;
    double x[4] = {1.0, 0.0, 0.0, 1.0};

    printf("--- RoPE ---\n");

    r = rope_cache_new(4, 10, 10000.0);

    /* pos=0: rotation by angle=0, so x unchanged */
    rope_apply(r, x, 0);
    check("rope pos0 [0]", x[0], 1.0, 1e-10);
    check("rope pos0 [1]", x[1], 0.0, 1e-10);
    check("rope pos0 [2]", x[2], 0.0, 1e-10);
    check("rope pos0 [3]", x[3], 1.0, 1e-10);

    rope_cache_free(r);
}

int main(void)
{
    printf("=== emilio MOV-only self-test ===\n\n");

    test_sw_math();
    test_eml_ops();
    test_matmul();
    test_softmax();
    test_rms_norm();
    test_silu();
    test_rope();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    if (n_fail > 0) {
        printf("FAIL\n");
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    return 0;
}
