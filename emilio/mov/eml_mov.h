/*
 * eml_mov.h -- Emilio EML inference engine (C89, MOV-only target)
 *
 * Minimal C89 port of emilio's v2 inference path for compilation
 * with the M/o/Vfuscator (mov-only x86 compiler).
 *
 * All system headers are avoided; libc prototypes declared manually
 * to bypass LCC's inability to parse modern C99 system headers.
 *
 * Reference: Odrzywołek (2026) arXiv:2603.21852
 */

#ifndef EML_MOV_H
#define EML_MOV_H

/* ==== Minimal libc prototypes (no system headers) ========================
 * LCC (movfuscator frontend) cannot parse modern system headers that use
 * C99 features. We declare only the functions we need.
 */

typedef unsigned long eml_size_t;
#define EML_NULL ((void *)0)

/* stdio */
typedef struct __eml_file_t EML_FILE;
extern EML_FILE *fopen(const char *path, const char *mode);
extern int fclose(EML_FILE *f);
extern eml_size_t fread(void *ptr, eml_size_t size, eml_size_t n, EML_FILE *f);
extern int fprintf(EML_FILE *f, const char *fmt, ...);
extern int printf(const char *fmt, ...);
extern int fputc(int c, EML_FILE *f);
extern int fflush(EML_FILE *f);

/* stderr handle -- set from linker symbol */
extern EML_FILE *stderr;

/* stdlib */
extern void *malloc(eml_size_t size);
extern void *calloc(eml_size_t n, eml_size_t size);
extern void *realloc(void *ptr, eml_size_t size);
extern void free(void *ptr);
extern void exit(int status);

/* string */
extern eml_size_t strlen(const char *s);
extern int strcmp(const char *a, const char *b);
extern int strncmp(const char *a, const char *b, eml_size_t n);
extern char *strcpy(char *dst, const char *src);
extern char *strcat(char *dst, const char *src);
extern void *memcpy(void *dst, const void *src, eml_size_t n);
extern void *memset(void *s, int c, eml_size_t n);

/* ==== Constants ========================================================= */

#define EML_NEG_INF   (-(1.0/0.0))
#define EML_POS_INF   (1.0/0.0)
#define EML_PI        3.14159265358979323846
#define EML_E         2.71828182845904523536
#define EML_LN2       0.69314718055994530942
#define EML_INV_LN2   1.44269504088896340736
#define EML_FRAC_1_PI 0.31830988618379067154

/* Qwen2.5 special token IDs */
#define EML_EOS_ID    151643
#define EML_EOT_ID    151645

/* ==== Software math (100% MOV-only -- no libm) ========================= */

double sw_fabs(double x);
double sw_fmod(double x, double y);
double sw_floor(double x);
double sw_exp(double x);
double sw_log(double x);
double sw_sqrt(double x);
double sw_sin(double x);
double sw_cos(double x);
double sw_atan2(double y, double x);
double sw_pow(double base, double exp);

/* ==== Model configuration =============================================== */

typedef struct {
    int vocab_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int d_model;
    int d_ff;
    double rope_freq_base;
    double rms_norm_eps;
    int max_seq_len;
    int d_head;
} EmlConfig;

/* ==== Sign+Magnitude Tensor ============================================ */

typedef struct {
    double *magnitudes;   /* ln|w_k| per element */
    double *signs;        /* +1.0 or -1.0 per element */
    int len;
} SmTensor;

/* ==== V2 Layer Weights ================================================== */

typedef struct {
    SmTensor sm_qkv;       /* fused Q+K+V */
    SmTensor sm_o;         /* output projection */
    SmTensor sm_gate_up;   /* fused gate+up */
    SmTensor sm_down;      /* down projection */
    double *q_bias;
    double *k_bias;
    double *v_bias;
    double *attn_norm;
    double *ffn_norm;
} V2LayerWeights;

/* ==== V2 Model Weights ================================================== */

typedef struct {
    EmlConfig config;
    double *token_embd;    /* (vocab_size, d_model) */
    double *output_norm;   /* (d_model,) */
    SmTensor sm_output;    /* LM head sign+mag */
    V2LayerWeights *layers;
    int n_layers;
} V2ModelWeights;

/* ==== RoPE Cache ======================================================== */

typedef struct {
    double *cos_cache;
    double *sin_cache;
    int d_half;
    int max_len;
} RopeCache;

/* ==== KV Cache ========================================================== */

typedef struct {
    double *k;
    double *v;
    int len;
    int max_len;
    int kv_dim;
} LayerKVCache;

typedef struct {
    LayerKVCache *layers;
    int n_layers;
} KVCache;

/* ==== Tokenizer ========================================================= */

typedef struct {
    char *str;
    int id;
} TokenEntry;

typedef struct {
    char *left;
    char *right;
    int rank;
} MergeEntry;

typedef struct {
    char **vocab;
    int vocab_size;
    TokenEntry *sorted;
    int sorted_count;
    MergeEntry *merges;
    int merges_count;
    int byte_to_unicode[256];
    int unicode_to_byte[512];
    int bos_id;
    int eos_id;
    int im_start_id;
    int im_end_id;
} Tokenizer;

/* ==== EML core ops (pure f64, fused algebraic forms) ==================== */

double eml_exp_f(double x);
double eml_ln_f(double x);
double eml_add_f(double a, double b);
double eml_sub_f(double a, double b);
double eml_neg_f(double x);
double eml_mul_f(double a, double b);
double eml_div_f(double a, double b);
double eml_inv_f(double x);
double eml_sqrt_f(double x);

/* ==== Composite ops ===================================================== */

void eml_add_vec(double *out, const double *a, const double *b, int n);
void eml_mul_vec(double *out, const double *a, const double *b, int n);

void build_matmul_sm(double *result, const double *a,
                     const SmTensor *w, int rows, int inner, int cols);
void eml_rms_norm(double *out, const double *x, const double *gamma,
                  int n, double eps);
void eml_silu_vec(double *out, const double *x, int n);
void build_softmax_cse(double *out, const double *x, int n);

/* ==== RoPE ============================================================== */

RopeCache *rope_cache_new(int d_head, int max_len, double base);
void rope_cache_free(RopeCache *r);
void rope_apply(const RopeCache *r, double *x, int pos);

/* ==== KV Cache ========================================================== */

KVCache *kv_cache_new(const EmlConfig *cfg, int max_len);
void kv_cache_free(KVCache *kv);

/* ==== Inference ========================================================= */

void v2_forward_one(double *logits, int token_id, int pos,
                    const V2ModelWeights *w, const RopeCache *rope,
                    KVCache *kv);

int *v2_generate(const int *prompt, int prompt_len,
                 const V2ModelWeights *w, const RopeCache *rope,
                 int max_new, int *out_len);

/* ==== Model I/O ========================================================= */

V2ModelWeights *load_eml_v2(const char *path, Tokenizer *tok);
void free_v2_weights(V2ModelWeights *w);

/* ==== Tokenizer ========================================================= */

void tokenizer_init_tables(Tokenizer *tok);
int *tokenizer_encode(const Tokenizer *tok, const char *text, int *out_len);
char *tokenizer_decode(const Tokenizer *tok, const int *ids, int n);
int *tokenizer_encode_chat(const Tokenizer *tok, const char *msg, int *out_len);
void tokenizer_free(Tokenizer *tok);

#endif /* EML_MOV_H */
