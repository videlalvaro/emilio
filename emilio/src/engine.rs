//! emilio — EML inference engine.
//!
//! Compiles a GGUF model into CSE-optimized EML expression DAGs,
//! then executes them through exp()/ln() only.
//!
//! Architecture support: Qwen2 (RMSNorm, SwiGLU, RoPE, GQA)
//!
//! INVARIANT: every numerical result flows through eml(x,y) = exp(x) - ln(y).
//! The only non-EML operations are:
//!   - Token lookup (discrete indexing)
//!   - Argmax / sampling (comparison)
//!   - RoPE angle computation (constant, precomputed)

use crate::eml_ops::*;
use crate::eml_optimizer::*;
use crate::gguf::GGUFFile;
use num_complex::Complex64;
use rayon::prelude::*;

// ─── Model config (extracted from GGUF metadata) ───────────────────────────

#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub vocab_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub rope_freq_base: f64,
    pub rms_norm_eps: f64,
    pub max_seq_len: usize,
    pub d_head: usize,
    pub eos_token_id: usize,
    pub eot_token_id: usize, // <|im_end|> for ChatML
}

impl QwenConfig {
    pub fn from_gguf(gguf: &GGUFFile) -> Self {
        let arch = gguf.meta_str("general.architecture").unwrap_or("qwen2");
        let d_model = gguf.meta_u32(&format!("{arch}.embedding_length")).unwrap_or(896) as usize;
        let n_heads = gguf.meta_u32(&format!("{arch}.attention.head_count")).unwrap_or(14) as usize;
        let n_kv_heads = gguf.meta_u32(&format!("{arch}.attention.head_count_kv")).unwrap_or(2) as usize;

        QwenConfig {
            vocab_size: gguf.meta_u32(&format!("{arch}.vocab_size"))
                .or_else(|| gguf.meta_u32("tokenizer.ggml.vocab_size"))
                .unwrap_or(151936) as usize,
            n_layers: gguf.meta_u32(&format!("{arch}.block_count")).unwrap_or(24) as usize,
            n_heads,
            n_kv_heads,
            d_model,
            d_ff: gguf.meta_u32(&format!("{arch}.feed_forward_length")).unwrap_or(4864) as usize,
            rope_freq_base: gguf.meta_f32(&format!("{arch}.rope.freq_base")).unwrap_or(1000000.0) as f64,
            rms_norm_eps: gguf.meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f64,
            max_seq_len: gguf.meta_u32(&format!("{arch}.context_length")).unwrap_or(32768) as usize,
            d_head: d_model / n_heads,
            eos_token_id: gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(151643) as usize,
            eot_token_id: gguf.meta_u32("tokenizer.ggml.eot_token_id")
                .or_else(|| {
                    // Qwen2.5 ChatML: <|im_end|> = 151645
                    Some(151645)
                })
                .unwrap() as usize,
        }
    }

    pub fn print(&self) {
        println!("  vocab_size:     {}", self.vocab_size);
        println!("  n_layers:       {}", self.n_layers);
        println!("  n_heads:        {}", self.n_heads);
        println!("  n_kv_heads:     {}", self.n_kv_heads);
        println!("  d_model:        {}", self.d_model);
        println!("  d_head:         {}", self.d_head);
        println!("  d_ff:           {}", self.d_ff);
        println!("  rope_freq_base: {}", self.rope_freq_base);
        println!("  rms_norm_eps:   {}", self.rms_norm_eps);
        println!("  max_seq_len:    {}", self.max_seq_len);
    }
}

// ─── Layer weights ─────────────────────────────────────────────────────────

pub struct LayerWeights {
    // Attention
    pub q_weight: Vec<f64>,  // (d_model, n_heads * d_head)
    pub k_weight: Vec<f64>,  // (d_model, n_kv_heads * d_head)
    pub v_weight: Vec<f64>,  // (d_model, n_kv_heads * d_head)
    pub q_bias: Vec<f64>,    // (n_heads * d_head,)
    pub k_bias: Vec<f64>,    // (n_kv_heads * d_head,)
    pub v_bias: Vec<f64>,    // (n_kv_heads * d_head,)
    pub o_weight: Vec<f64>,  // (n_heads * d_head, d_model)

    // Attention norm
    pub attn_norm: Vec<f64>, // (d_model,)

    // FFN (SwiGLU)
    pub gate_weight: Vec<f64>,  // (d_model, d_ff)
    pub up_weight: Vec<f64>,    // (d_model, d_ff)
    pub down_weight: Vec<f64>,  // (d_ff, d_model)

    // FFN norm
    pub ffn_norm: Vec<f64>,  // (d_model,)

    // Precomputed ln(W) in (cols, inner) layout for each weight matrix.
    // These are element-wise ln of the GGUF-layout weights (double-transpose cancels).
    pub ln_q: Vec<Complex64>,
    pub ln_k: Vec<Complex64>,
    pub ln_v: Vec<Complex64>,
    pub ln_o: Vec<Complex64>,
    pub ln_gate: Vec<Complex64>,
    pub ln_up: Vec<Complex64>,
    pub ln_down: Vec<Complex64>,
}

pub struct ModelWeights {
    pub config: QwenConfig,
    pub token_embd: Vec<f64>,   // (vocab, d_model)
    pub output_norm: Vec<f64>,  // (d_model,)
    pub output_weight: Vec<f64>, // (d_model, vocab) — may be tied to token_embd
    pub ln_output: Vec<Complex64>, // precomputed ln(output_weight) transposed: (vocab, d_model)
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    /// Load all weights from a GGUF file, dequantizing to f64.
    pub fn from_gguf(gguf: &GGUFFile) -> std::io::Result<Self> {
        let config = QwenConfig::from_gguf(gguf);
        println!("  Loading weights ({} layers)...", config.n_layers);

        // Token embeddings
        let token_embd = load_tensor(gguf, "token_embd.weight")?;

        // Output norm
        let output_norm = load_tensor(gguf, "output_norm.weight")?;

        // Output weight — may be tied to token_embd
        // GGUF stores as (vocab, d_model); we need (d_model, vocab) for matmul
        let output_weight = if gguf.tensor_info("output.weight").is_some() {
            let raw = load_tensor(gguf, "output.weight")?;
            // raw is (vocab, d_model), transpose to (d_model, vocab)
            let v = config.vocab_size;
            let d = config.d_model;
            let mut out = vec![0.0f64; d * v];
            for i in 0..v {
                for j in 0..d {
                    out[j * v + i] = raw[i * d + j];
                }
            }
            out
        } else {
            // Tied embeddings: output = token_embd^T
            // token_embd is (vocab, d_model), we need (d_model, vocab)
            let v = config.vocab_size;
            let d = config.d_model;
            let mut out = vec![0.0f64; d * v];
            for i in 0..v {
                for j in 0..d {
                    out[j * v + i] = token_embd[i * d + j];
                }
            }
            out
        };

        // Layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let pfx = format!("blk.{i}");
            let q_weight = load_tensor(gguf, &format!("{pfx}.attn_q.weight"))?;
            let k_weight = load_tensor(gguf, &format!("{pfx}.attn_k.weight"))?;
            let v_weight = load_tensor(gguf, &format!("{pfx}.attn_v.weight"))?;
            let o_weight = load_tensor(gguf, &format!("{pfx}.attn_output.weight"))?;
            let gate_weight = load_tensor(gguf, &format!("{pfx}.ffn_gate.weight"))?;
            let up_weight = load_tensor(gguf, &format!("{pfx}.ffn_up.weight"))?;
            let down_weight = load_tensor(gguf, &format!("{pfx}.ffn_down.weight"))?;

            // Precompute ln(W) — element-wise ln of GGUF layout weights.
            // For layer weights stored as (out_dim, in_dim), the double-transpose
            // (emilio transpose + matmul ln-transpose) cancels, so element-wise
            // ln gives us the correct (cols, inner) layout directly.
            let ln_q = precompute_ln_weight(&q_weight);
            let ln_k = precompute_ln_weight(&k_weight);
            let ln_v = precompute_ln_weight(&v_weight);
            let ln_o = precompute_ln_weight(&o_weight);
            let ln_gate = precompute_ln_weight(&gate_weight);
            let ln_up = precompute_ln_weight(&up_weight);
            let ln_down = precompute_ln_weight(&down_weight);

            let layer = LayerWeights {
                q_weight, k_weight, v_weight,
                q_bias: load_tensor(gguf, &format!("{pfx}.attn_q.bias"))?,
                k_bias: load_tensor(gguf, &format!("{pfx}.attn_k.bias"))?,
                v_bias: load_tensor(gguf, &format!("{pfx}.attn_v.bias"))?,
                o_weight,
                attn_norm: load_tensor(gguf, &format!("{pfx}.attn_norm.weight"))?,
                gate_weight, up_weight, down_weight,
                ffn_norm: load_tensor(gguf, &format!("{pfx}.ffn_norm.weight"))?,
                ln_q, ln_k, ln_v, ln_o, ln_gate, ln_up, ln_down,
            };
            layers.push(layer);
            if (i + 1) % 8 == 0 || i == config.n_layers - 1 {
                println!("    Loaded layer {}/{}", i + 1, config.n_layers);
            }
        }

        // Precompute ln(output_weight) — output_weight is (d_model, vocab),
        // which is (inner, cols). Needs a real transpose to (vocab, d_model).
        let ln_output = precompute_ln_weight_transposed(
            &output_weight, config.d_model, config.vocab_size,
        );
        println!("  Precomputed ln(weights) for all layers + LM head");

        Ok(ModelWeights { config, token_embd, output_norm, output_weight, ln_output, layers })
    }
}

fn load_tensor(gguf: &GGUFFile, name: &str) -> std::io::Result<Vec<f64>> {
    let info = gguf.tensor_info(name)
        .ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("tensor not found: {name}"),
        ))?;
    gguf.load_tensor_f64(info)
}

// ─── EML RMSNorm ────────────────────────────────────────────────────────────
//
// RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
//
// In EML with CSE:
//   - x² = exp(2 * ln(x))  → share ln(x)
//   - mean = sum / N        → sum is 0-cost adds, div via log domain
//   - sqrt = exp(0.5 * ln(..))
//   - final = x * gamma / std = exp(ln(x) + ln(gamma) - ln(std))
//     → ln(x) already cached, ln(gamma) constant, ln(std) computed once

pub fn eml_rms_norm(x: &[f64], gamma: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let nc = to_c(n as f64);

    // Cache ln(x_i) — shared between squaring and final division
    let ln_x: Vec<Complex64> = x.iter()
        .map(|&v| eml_ln(to_c(v)))
        .collect();

    // x² = exp(2 * ln(x))
    let two = to_c(2.0);
    let sq_sum: Complex64 = ln_x.iter()
        .map(|&lx| eml_exp(eml_mul(two, lx)))
        .fold(Complex64::new(0.0, 0.0), eml_add);

    // mean(x²) = sq_sum / N via log domain: exp(ln(sq_sum) - ln(N))
    let mean_sq = eml_div(sq_sum, nc);

    // std = sqrt(mean_sq + eps) = exp(0.5 * ln(mean_sq + eps))
    let ln_std = eml_mul(to_c(0.5), eml_ln(eml_add(mean_sq, to_c(eps))));

    // Cache ln(gamma)
    let ln_gamma: Vec<Complex64> = gamma.iter()
        .map(|&g| eml_ln(to_c(g)))
        .collect();

    // result_i = x_i * gamma_i / std
    //          = exp(ln(x_i) + ln(gamma_i) - ln(std))
    // All ln's already cached!
    (0..n)
        .map(|i| {
            to_r(eml_exp(eml_sub(eml_add(ln_x[i], ln_gamma[i]), ln_std)))
        })
        .collect()
}

// ─── EML RoPE ───────────────────────────────────────────────────────────────
//
// Rotary position embedding applies a rotation:
//   q'[2i]   = q[2i]   * cos(θ) - q[2i+1] * sin(θ)
//   q'[2i+1] = q[2i]   * sin(θ) + q[2i+1] * cos(θ)
//
// where θ_i = pos / (base ^ (2i/d_head))
//
// In EML: mul,add,sub are all EML. cos/sin can be expressed as:
//   cos(θ) = Re(exp(iθ)) — but we're in the real domain, so we precompute
//   the cos/sin tables (they're constants, not data-dependent).
//
// The actual rotation mul/add/sub is all EML.

pub struct RopeCache {
    /// cos[pos * d_half + i] and sin[pos * d_half + i]
    pub cos: Vec<f64>,
    pub sin: Vec<f64>,
    pub d_half: usize,
    pub max_len: usize,
}

impl RopeCache {
    pub fn new(d_head: usize, max_len: usize, base: f64) -> Self {
        let d_half = d_head / 2;
        let mut cos = vec![0.0; max_len * d_half];
        let mut sin = vec![0.0; max_len * d_half];

        for pos in 0..max_len {
            for i in 0..d_half {
                // freq = 1 / base^(2i/d_head)
                // = exp(neg(mul(2i/d_head, ln(base)))) via EML
                let exponent = to_c(2.0 * i as f64 / d_head as f64);
                let ln_base = eml_ln(to_c(base));
                let freq = to_r(eml_exp(eml_neg(eml_mul(exponent, ln_base))));

                // angle = pos * freq via EML mul
                let angle = to_r(eml_mul(to_c(pos as f64), to_c(freq)));

                // cos(θ) + i·sin(θ) = exp(iθ) — EML exp on complex argument
                let rot = eml_exp(Complex64::new(0.0, angle));
                cos[pos * d_half + i] = rot.re;
                sin[pos * d_half + i] = rot.im;
            }
        }

        RopeCache { cos, sin, d_half, max_len }
    }

    /// Apply RoPE to a vector of length d_head at position pos.
    /// Uses EML mul/add/sub for the rotation.
    pub fn apply(&self, x: &mut [f64], pos: usize) {
        let d_half = self.d_half;
        for i in 0..d_half {
            let cos_v = self.cos[pos * d_half + i];
            let sin_v = self.sin[pos * d_half + i];
            let x0 = x[i];
            let x1 = x[i + d_half];

            // x'[i]        = x[i] * cos - x[i+d_half] * sin
            // x'[i+d_half] = x[i] * sin + x[i+d_half] * cos
            // All via EML mul/add/sub
            let x0c = to_c(x0);
            let x1c = to_c(x1);
            let cc = to_c(cos_v);
            let sc = to_c(sin_v);

            x[i] = to_r(eml_sub(eml_mul(x0c, cc), eml_mul(x1c, sc)));
            x[i + d_half] = to_r(eml_add(eml_mul(x0c, sc), eml_mul(x1c, cc)));
        }
    }
}

// ─── EML SwiGLU FFN ─────────────────────────────────────────────────────────
//
// SwiGLU(x) = (x @ gate_w) * silu(x @ up_w)
// silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
//
// In EML:
//   sigmoid(z) = inv(1 + exp(-z)) = exp(-ln(1 + exp(-z)))
//   silu(z) = z * sigmoid(z) = exp(ln(z) + ln(sigmoid(z)))
//   final = gate * silu(up) = exp(ln(gate) + ln(silu(up)))
//
// CSE: ln(gate) and ln(silu(up)) cached per element.

pub fn eml_silu(x: &[f64]) -> Vec<f64> {
    x.par_iter()
        .map(|&v| {
            let xc = to_c(v);
            // sigmoid(x) = 1 / (1 + exp(-x)) = inv(add(1, exp(neg(x))))
            let sig = eml_inv(eml_add(ONE, eml_exp(eml_neg(xc))));
            // silu(x) = x * sigmoid(x) = exp(ln(x) + ln(sig))
            to_r(eml_mul(xc, sig))
        })
        .collect()
}

// ─── KV Cache ───────────────────────────────────────────────────────────────

/// Per-layer KV cache: stores projected, RoPE'd K and V for all past positions.
pub struct LayerKVCache {
    /// k_cache[pos * kv_dim + h * d_head + d] — cached key vectors
    pub k: Vec<f64>,
    /// v_cache[pos * kv_dim + h * d_head + d] — cached value vectors
    pub v: Vec<f64>,
    pub len: usize,  // number of positions cached
    pub kv_dim: usize,
}

impl LayerKVCache {
    pub fn new(kv_dim: usize, max_len: usize) -> Self {
        LayerKVCache {
            k: vec![0.0; max_len * kv_dim],
            v: vec![0.0; max_len * kv_dim],
            len: 0,
            kv_dim,
        }
    }

    /// Append K,V for one position.
    pub fn append(&mut self, k: &[f64], v: &[f64]) {
        let off = self.len * self.kv_dim;
        self.k[off..off + self.kv_dim].copy_from_slice(k);
        self.v[off..off + self.kv_dim].copy_from_slice(v);
        self.len += 1;
    }
}

/// Full KV cache for all layers.
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(cfg: &QwenConfig, max_len: usize) -> Self {
        let kv_dim = cfg.n_kv_heads * cfg.d_head;
        let layers = (0..cfg.n_layers)
            .map(|_| LayerKVCache::new(kv_dim, max_len))
            .collect();
        KVCache { layers }
    }
}

// ─── Single-token forward pass with KV cache ───────────────────────────────

/// Forward pass for a single new token, using KV cache for past context.
/// Returns logits for the new token only (vocab_size elements).
pub fn emilio_forward_one(
    token_id: usize,
    pos: usize,
    weights: &ModelWeights,
    rope: &RopeCache,
    kv_cache: &mut KVCache,
) -> Vec<f64> {
    let cfg = &weights.config;
    let d = cfg.d_model;

    // 1. Token embedding
    let mut x: Vec<f64> = weights.token_embd[token_id * d..(token_id + 1) * d].to_vec();

    // 2. Transformer layers (single token)
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        x = transformer_layer_one(&x, layer, cfg, rope, pos,
                                   &mut kv_cache.layers[layer_idx]);
    }

    // 3. Final RMSNorm
    let normed = eml_rms_norm(&x, &weights.output_norm, cfg.rms_norm_eps);

    // 4. LM head: (1, d_model) @ (d_model, vocab)
    build_matmul_cse_precomp(&normed, &weights.ln_output, 1, d, cfg.vocab_size)
}

// ─── Single-token transformer layer with KV cache ──────────────────────────

fn transformer_layer_one(
    x: &[f64],
    layer: &LayerWeights,
    cfg: &QwenConfig,
    rope: &RopeCache,
    pos: usize,
    kv: &mut LayerKVCache,
) -> Vec<f64> {
    // Pre-attention RMSNorm
    let normed = eml_rms_norm(x, &layer.attn_norm, cfg.rms_norm_eps);

    // Attention (single query, cached KV)
    let attn_out = eml_gqa_attention_one(&normed, layer, cfg, rope, pos, kv);

    // Residual
    let mut x2 = eml_add_vec(x, &attn_out);

    // Pre-FFN RMSNorm
    let normed2 = eml_rms_norm(&x2, &layer.ffn_norm, cfg.rms_norm_eps);

    // SwiGLU FFN (single token)
    let ffn_out = eml_swiglu_ffn(&normed2, layer, cfg, 1);

    // Residual
    x2 = eml_add_vec(&x2, &ffn_out);
    x2
}

// ─── Single-token GQA attention with KV cache ──────────────────────────────

fn eml_gqa_attention_one(
    x: &[f64],          // (d_model,) — single normalized token
    layer: &LayerWeights,
    cfg: &QwenConfig,
    rope: &RopeCache,
    pos: usize,
    kv: &mut LayerKVCache,
) -> Vec<f64> {
    let d = cfg.d_model;
    let d_head = cfg.d_head;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let heads_per_kv = n_heads / n_kv_heads;
    let q_dim = n_heads * d_head;
    let kv_dim = n_kv_heads * d_head;

    // QKV projections for single token: (1, d) @ (d, out_dim)
    // Use precomputed ln(W) — no transpose or ln needed
    let mut q = build_matmul_cse_precomp(x, &layer.ln_q, 1, d, q_dim);
    let mut k_new = build_matmul_cse_precomp(x, &layer.ln_k, 1, d, kv_dim);
    let mut v_new = build_matmul_cse_precomp(x, &layer.ln_v, 1, d, kv_dim);

    // Add bias (EML add — 0-cost, proven by cancellation)
    for (q_val, &bias) in q.iter_mut().zip(&layer.q_bias) {
        *q_val = to_r(eml_add(to_c(*q_val), to_c(bias)));
    }
    for j in 0..kv_dim {
        k_new[j] = to_r(eml_add(to_c(k_new[j]), to_c(layer.k_bias[j])));
        v_new[j] = to_r(eml_add(to_c(v_new[j]), to_c(layer.v_bias[j])));
    }

    // Apply RoPE
    for h in 0..n_heads {
        rope.apply(&mut q[h * d_head..(h + 1) * d_head], pos);
    }
    for h in 0..n_kv_heads {
        rope.apply(&mut k_new[h * d_head..(h + 1) * d_head], pos);
    }

    // Store K,V in cache
    kv.append(&k_new, &v_new);
    let t = kv.len; // total sequence length including this token

    // Attention: single query against all cached K,V
    let mut out = vec![0.0f64; d];
    let scale = to_r(eml_inv(eml_sqrt(to_c(d_head as f64))));

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;

        // Compute scores[j] = dot(Q[h,:], K_cached[kv_h, j, :]) / sqrt(d_head)
        let mut scores = Vec::with_capacity(t);
        for j in 0..t {
            let mut dot = Complex64::new(0.0, 0.0);
            for dd in 0..d_head {
                let qv = q[h * d_head + dd];
                let kv_val = kv.k[j * kv_dim + kv_h * d_head + dd];
                dot = eml_add(dot, eml_mul(to_c(qv), to_c(kv_val)));
            }
            scores.push(to_r(eml_mul(dot, to_c(scale))));
        }

        // Softmax
        let attn_w = build_softmax_cse(&scores);

        // Weighted sum over cached V
        for dd in 0..d_head {
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, &w) in attn_w.iter().enumerate() {
                let vv = kv.v[j * kv_dim + kv_h * d_head + dd];
                acc = eml_add(acc, eml_mul(to_c(w), to_c(vv)));
            }
            out[h * d_head + dd] = to_r(acc);
        }
    }

    // Output projection
    build_matmul_cse_precomp(&out, &layer.ln_o, 1, q_dim, d)
}

// ─── SwiGLU FFN ─────────────────────────────────────────────────────────────

fn eml_swiglu_ffn(
    x: &[f64],
    layer: &LayerWeights,
    cfg: &QwenConfig,
    t: usize,
) -> Vec<f64> {
    let d = cfg.d_model;
    let d_ff = cfg.d_ff;

    // gate and up projections: use precomputed ln(W)
    let gate = build_matmul_cse_precomp(x, &layer.ln_gate, t, d, d_ff);
    let up = build_matmul_cse_precomp(x, &layer.ln_up, t, d, d_ff);

    // SwiGLU: silu(gate) * up
    let gate_activated = eml_silu(&gate);

    // Element-wise mul via EML
    let hidden = eml_mul_vec(&gate_activated, &up);

    // Down projection — use precomputed ln(down_weight)
    build_matmul_cse_precomp(&hidden, &layer.ln_down, t, d_ff, d)
}

// ─── Generation ─────────────────────────────────────────────────────────────

// ─── Generation with KV cache ───────────────────────────────────────────────

pub fn emilio_generate(
    prompt: &[usize],
    weights: &ModelWeights,
    rope: &RopeCache,
    max_new: usize,
) -> Vec<usize> {
    let cfg = &weights.config;
    let mut ids = prompt.to_vec();
    let max_len = cfg.max_seq_len.min(prompt.len() + max_new + 16);
    let mut kv_cache = KVCache::new(cfg, max_len);

    // Prefill: process all prompt tokens one-by-one through KV cache path.
    // This is simpler than a batched prefill + cache population.
    eprintln!("  Prefilling {} prompt tokens...", prompt.len());
    let mut _last_logits = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        _last_logits = emilio_forward_one(tok, i, weights, rope, &mut kv_cache);
        if (i + 1) % 10 == 0 || i == prompt.len() - 1 {
            eprint!("\r  Prefilled {}/{}", i + 1, prompt.len());
        }
    }
    eprintln!();

    // Decode: generate new tokens one at a time
    for step in 0..max_new {
        let logits = if step == 0 {
            // First decode step: we already have logits from last prefill token
            _last_logits.clone()
        } else {
            // Process the most recently appended token
            let last_tok = *ids.last().unwrap();
            let pos = ids.len() - 1; // position of this token in the sequence
            emilio_forward_one(last_tok, pos, weights, rope, &mut kv_cache)
        };

        // Greedy argmax
        let next_token = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        ids.push(next_token);

        eprint!("\r  Generated {}/{} tokens", step + 1, max_new);

        // Stop on EOS or end-of-turn (<|im_end|>)
        if next_token == cfg.eos_token_id || next_token == cfg.eot_token_id { break; }
    }
    eprintln!();

    ids
}

// ═══════════════════════════════════════════════════════════════════════════
// Metal GPU-accelerated inference
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "metal")]
#[allow(clippy::too_many_arguments)]
pub mod gpu {
    use super::*;
    use crate::metal_eml::{MetalContext, GpuModelWeights, ScratchPool};

    /// GPU-accelerated single-token forward pass.
    /// All matmul projections run on GPU; bias/norm/RoPE/softmax stay on CPU.
    pub fn forward_one_gpu(
        token_id: usize,
        pos: usize,
        weights: &ModelWeights,
        gpu_w: &GpuModelWeights,
        ctx: &MetalContext,
        pool: &ScratchPool,
        rope: &RopeCache,
        kv_cache: &mut KVCache,
    ) -> Vec<f64> {
        let cfg = &weights.config;
        let d = cfg.d_model;

        // 1. Token embedding (discrete lookup — not EML)
        let mut x: Vec<f64> = weights.token_embd[token_id * d..(token_id + 1) * d].to_vec();

        // 2. Transformer layers
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let gpu_layer = &gpu_w.layers[layer_idx];
            x = transformer_layer_one_gpu(
                &x, layer, gpu_layer, ctx, pool, cfg, rope, pos,
                &mut kv_cache.layers[layer_idx],
                &gpu_w.eps_buf,
            );
        }

        // 3. Final RMSNorm (CPU — small: d_model elements)
        let normed = eml_rms_norm(&x, &weights.output_norm, cfg.rms_norm_eps);

        // 4. LM head: (1, d_model) @ (d_model, vocab) — GPU
        ctx.single_matmul(pool, &normed, &gpu_w.output, 1, d, cfg.vocab_size)
    }

    fn transformer_layer_one_gpu(
        x: &[f64],
        layer: &LayerWeights,
        gpu_layer: &crate::metal_eml::GpuLayerWeights,
        ctx: &MetalContext,
        pool: &ScratchPool,
        cfg: &QwenConfig,
        rope: &RopeCache,
        pos: usize,
        kv: &mut LayerKVCache,
        eps_buf: &metal::Buffer,
    ) -> Vec<f64> {
        let d = cfg.d_model;
        let d_ff = cfg.d_ff;
        let q_dim = cfg.n_heads * cfg.d_head;

        // Pre-attention RMSNorm (CPU)
        let normed = eml_rms_norm(x, &layer.attn_norm, cfg.rms_norm_eps);

        // Attention producing pre-O-projection weighted output (GPU QKV, CPU attention)
        let attn_weighted = gqa_attention_one_gpu(
            &normed, layer, gpu_layer, ctx, pool, cfg, rope, pos, kv,
        );

        // Fused O→residual→RMSNorm→FFN→residual (1 GPU commit)
        ctx.batch_o_to_ffn(
            pool, &attn_weighted, x,
            &gpu_layer.ffn_norm, eps_buf, gpu_layer,
            q_dim, d, d_ff,
        )
    }

    fn gqa_attention_one_gpu(
        x: &[f64],
        layer: &LayerWeights,
        gpu_layer: &crate::metal_eml::GpuLayerWeights,
        ctx: &MetalContext,
        pool: &ScratchPool,
        cfg: &QwenConfig,
        rope: &RopeCache,
        pos: usize,
        kv: &mut LayerKVCache,
    ) -> Vec<f64> {
        let d = cfg.d_model;
        let d_head = cfg.d_head;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let heads_per_kv = n_heads / n_kv_heads;
        let q_dim = n_heads * d_head;
        let kv_dim = n_kv_heads * d_head;

        // QKV projections — batched: ln(x) computed once, 3 GPU dispatches, 1 commit
        let (mut q, mut k_new, mut v_new) = ctx.batch_qkv(
            pool, x, &gpu_layer.q, &gpu_layer.k, &gpu_layer.v, d, q_dim, kv_dim,
        );

        // Add bias (CPU — tiny vectors)
        for (q_val, &bias) in q.iter_mut().zip(&layer.q_bias) {
            *q_val = to_r(eml_add(to_c(*q_val), to_c(bias)));
        }
        for j in 0..kv_dim {
            k_new[j] = to_r(eml_add(to_c(k_new[j]), to_c(layer.k_bias[j])));
            v_new[j] = to_r(eml_add(to_c(v_new[j]), to_c(layer.v_bias[j])));
        }

        // RoPE (CPU — precomputed tables)
        for h in 0..n_heads {
            rope.apply(&mut q[h * d_head..(h + 1) * d_head], pos);
        }
        for h in 0..n_kv_heads {
            rope.apply(&mut k_new[h * d_head..(h + 1) * d_head], pos);
        }

        // KV cache
        kv.append(&k_new, &v_new);
        let t = kv.len;

        // Attention scores (CPU — quadratic in seq_len but small per token)
        let mut out = vec![0.0f64; d];
        let scale = to_r(eml_inv(eml_sqrt(to_c(d_head as f64))));

        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;

            let mut scores = Vec::with_capacity(t);
            for j in 0..t {
                let mut dot = Complex64::new(0.0, 0.0);
                for dd in 0..d_head {
                    let qv = q[h * d_head + dd];
                    let kv_val = kv.k[j * kv_dim + kv_h * d_head + dd];
                    dot = eml_add(dot, eml_mul(to_c(qv), to_c(kv_val)));
                }
                scores.push(to_r(eml_mul(dot, to_c(scale))));
            }

            let attn_w = build_softmax_cse(&scores);

            for dd in 0..d_head {
                let mut acc = Complex64::new(0.0, 0.0);
                for (j, &w) in attn_w.iter().enumerate() {
                    let vv = kv.v[j * kv_dim + kv_h * d_head + dd];
                    acc = eml_add(acc, eml_mul(to_c(w), to_c(vv)));
                }
                out[h * d_head + dd] = to_r(acc);
            }
        }

        // Output projection moved to batch_o_to_ffn — return pre-O weighted output
        out
    }

    /// GPU-accelerated generation with KV cache.
    pub fn generate_gpu(
        prompt: &[usize],
        weights: &ModelWeights,
        gpu_w: &GpuModelWeights,
        ctx: &MetalContext,
        pool: &ScratchPool,
        rope: &RopeCache,
        max_new: usize,
    ) -> Vec<usize> {
        let cfg = &weights.config;
        let mut ids = prompt.to_vec();
        let max_len = cfg.max_seq_len.min(prompt.len() + max_new + 16);
        let mut kv_cache = KVCache::new(cfg, max_len);

        eprintln!("  Prefilling {} prompt tokens (GPU)...", prompt.len());
        let mut _last_logits = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            _last_logits = forward_one_gpu(
                tok, i, weights, gpu_w, ctx, pool, rope, &mut kv_cache,
            );
            if (i + 1) % 10 == 0 || i == prompt.len() - 1 {
                eprint!("\r  Prefilled {}/{}", i + 1, prompt.len());
            }
        }
        eprintln!();

        for step in 0..max_new {
            let logits = if step == 0 {
                _last_logits.clone()
            } else {
                let last_tok = *ids.last().unwrap();
                let pos = ids.len() - 1;
                forward_one_gpu(
                    last_tok, pos, weights, gpu_w, ctx, pool, rope, &mut kv_cache,
                )
            };

            let next_token = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            ids.push(next_token);
            eprint!("\r  Generated {}/{} tokens (GPU)", step + 1, max_new);

            if next_token == cfg.eos_token_id || next_token == cfg.eot_token_id { break; }
        }
        eprintln!();

        ids
    }
}
