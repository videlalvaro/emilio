//! EML v2: optimized compiled model format.
//!
//! Optimizations over v1:
//! 1. **Sign+magnitude encoding**: ln(w) stored as (f64 magnitude, 1-bit sign)
//!    instead of Complex64 (16B → 8.125B per element, ~50% file size reduction).
//!    The inner loop skips the round(im/π) computation entirely.
//!
//! 2. **QKV weight fusion**: Q, K, V projection weights concatenated into one
//!    tensor. One matmul call instead of 3, ln(activation) computed once.
//!
//! 3. **Gate+Up fusion**: SwiGLU gate and up projection weights concatenated
//!    into one tensor. One matmul call instead of 2.
//!
//! 4. **Sparse pruning**: At compile time, weights with |ln(w)| < threshold
//!    are set to -∞. IEEE 754 guarantees exp(-∞) = 0, so pruned entries
//!    contribute nothing without any runtime branching.
//!
//! 5. **Execution graph**: Stored op DAG describing the inference computation
//!    for documentation, reproducibility, and future graph-level optimization.
//!
//! Reference: Odrzywołek (2026) arXiv:2603.21852

use crate::engine::*;
use crate::eml_ops::*;
use crate::eml_optimizer::build_softmax_cse;
use num_complex::Complex64;
use rayon::prelude::*;

// ─── Sign+Magnitude Tensor ─────────────────────────────────────────────────

/// Compact sign+magnitude representation of precomputed ln(weights).
///
/// For real weight w:
///   ln(w) = ln|w| + i·π·(w < 0)
///
/// We store:
///   magnitudes[k] = ln|w_k|         (f64, 8 bytes)
///   signs[k]      = +1.0 or -1.0    (f64 in memory, 1 bit on disk)
///
/// On disk: magnitudes as f64[] + signs as packed bitmap (1 bit/element).
/// In memory: signs expanded to f64 for branch-free inner loop.
///
/// Pruned entries have magnitude = -∞. IEEE 754: exp(-∞) = 0, contributing
/// nothing to the inner product without any runtime branching.
pub struct SmTensor {
    pub magnitudes: Vec<f64>,
    pub signs: Vec<f64>,      // +1.0 or -1.0 per element (fast for inner loop)
    pub len: usize,
}

impl SmTensor {
    /// Convert from Complex64 ln(w) representation.
    pub fn from_complex(c: &[Complex64]) -> Self {
        let len = c.len();
        let mut magnitudes = Vec::with_capacity(len);
        let mut signs = Vec::with_capacity(len);
        for &v in c {
            magnitudes.push(v.re);
            // im is 0 (positive) or π (negative)
            if v.im.abs() > 1.0 {
                signs.push(-1.0);
            } else {
                signs.push(1.0);
            }
        }
        SmTensor { magnitudes, signs, len }
    }

    /// Fuse multiple tensors by concatenating their columns.
    ///
    /// Each input tensor has layout (cols_i, inner), i.e. cols_i columns
    /// each of length `inner`. The fused tensor has (sum(cols_i), inner).
    pub fn fuse(tensors: &[&SmTensor], inner: usize) -> Self {
        let total_cols: usize = tensors.iter().map(|t| t.len / inner).sum();
        let total_len = total_cols * inner;
        let mut magnitudes = Vec::with_capacity(total_len);
        let mut signs = Vec::with_capacity(total_len);
        for t in tensors {
            magnitudes.extend_from_slice(&t.magnitudes);
            signs.extend_from_slice(&t.signs);
        }
        SmTensor { magnitudes, signs, len: total_len }
    }

    /// Prune entries with magnitude below threshold by setting to -∞.
    /// Returns the number of entries pruned.
    pub fn prune(&mut self, threshold: f64) -> usize {
        let mut count = 0;
        for mag in self.magnitudes.iter_mut() {
            if *mag < threshold {
                *mag = f64::NEG_INFINITY;
                count += 1;
            }
        }
        count
    }

    /// Pack signs into bitmap: bit k%8 of byte k/8 is 1 if sign is negative.
    pub fn pack_signs(&self) -> Vec<u8> {
        let nbytes = self.len.div_ceil(8);
        let mut packed = vec![0u8; nbytes];
        for (i, &s) in self.signs.iter().enumerate() {
            if s < 0.0 {
                packed[i / 8] |= 1 << (i % 8);
            }
        }
        packed
    }

    /// Unpack signs from bitmap into f64 +1.0/-1.0.
    pub fn from_packed(magnitudes: Vec<f64>, packed: &[u8], len: usize) -> Self {
        let mut signs = Vec::with_capacity(len);
        for i in 0..len {
            let bit = (packed[i / 8] >> (i % 8)) & 1;
            signs.push(if bit == 1 { -1.0 } else { 1.0 });
        }
        SmTensor { magnitudes, signs, len }
    }
}

// ─── Sign+Magnitude Matmul Kernel ──────────────────────────────────────────

/// CSE matmul with sign+magnitude precomputed weights.
///
/// Computes C[i,j] = Σ_k a[i,k] * w[j,k]
/// where w is stored as sign+magnitude of ln(w).
///
/// Inner loop: exp(ln|a_k| + ln|w_jk|) * sign(a_k) * sign(w_jk)
///
/// vs Complex64 version:
///   exp(la.re + lb.re) * branchless_sign(la.im + lb.im)
///
/// The sign+mag version eliminates the round(im/π) computation.
pub fn build_matmul_sm_precomp(
    a: &[f64],
    w: &SmTensor,  // (cols, inner) layout
    rows: usize, inner: usize, cols: usize,
) -> Vec<f64> {
    // Phase 1: precompute ln|a| and sign(a) — once per call
    let mut la_mags = Vec::with_capacity(a.len());
    let mut la_signs = Vec::with_capacity(a.len());
    for &v in a.iter() {
        if v > 0.0 {
            la_mags.push(to_r(eml_ln(to_c(v))));
            la_signs.push(1.0f64);
        } else if v < 0.0 {
            la_mags.push(to_r(eml_ln(to_c(-v))));
            la_signs.push(-1.0f64);
        } else {
            la_mags.push(f64::NEG_INFINITY);
            la_signs.push(1.0f64);
        }
    }

    // Phase 2: parallel matmul with 4-wide unroll
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

                    let e0 = (la_mags[a_off+k] + w.magnitudes[b_off+k]).exp(); // EML_AUDIT:OK — real-exp bypass: exp(ln|a|+ln|w|), sign handled separately
                    acc0 += e0 * la_signs[a_off+k] * w.signs[b_off+k];

                    let e1 = (la_mags[a_off+k+1] + w.magnitudes[b_off+k+1]).exp(); // EML_AUDIT:OK — real-exp bypass
                    acc1 += e1 * la_signs[a_off+k+1] * w.signs[b_off+k+1];

                    let e2 = (la_mags[a_off+k+2] + w.magnitudes[b_off+k+2]).exp(); // EML_AUDIT:OK — real-exp bypass
                    acc2 += e2 * la_signs[a_off+k+2] * w.signs[b_off+k+2];

                    let e3 = (la_mags[a_off+k+3] + w.magnitudes[b_off+k+3]).exp(); // EML_AUDIT:OK — real-exp bypass
                    acc3 += e3 * la_signs[a_off+k+3] * w.signs[b_off+k+3];
                }

                for k in (chunks*4)..(chunks*4 + remainder) {
                    let e = (la_mags[a_off+k] + w.magnitudes[b_off+k]).exp(); // EML_AUDIT:OK — real-exp bypass remainder
                    acc0 += e * la_signs[a_off+k] * w.signs[b_off+k];
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
                    let e = (la_mags[a_off+k] + w.magnitudes[b_off+k]).exp(); // EML_AUDIT:OK — real-exp bypass (sequential fallback)
                    acc += e * la_signs[a_off+k] * w.signs[b_off+k];
                }
                result[i * cols + j] = acc;
            }
        }
    }

    result
}

// ─── V2 Model Weights ──────────────────────────────────────────────────────

pub struct V2LayerWeights {
    /// Fused QKV: (q_dim + kv_dim + kv_dim) columns × d_model inner
    pub sm_qkv: SmTensor,
    /// Output projection: d_model columns × q_dim inner
    pub sm_o: SmTensor,
    /// Fused gate+up: (2 × d_ff) columns × d_model inner
    pub sm_gate_up: SmTensor,
    /// Down projection: d_model columns × d_ff inner
    pub sm_down: SmTensor,
    /// Biases
    pub q_bias: Vec<f64>,
    pub k_bias: Vec<f64>,
    pub v_bias: Vec<f64>,
    /// Norms
    pub attn_norm: Vec<f64>,
    pub ffn_norm: Vec<f64>,
}

/// Execution graph op — documents one step of the inference computation.
#[derive(Clone)]
pub enum ExecOp {
    RmsNorm { layer: i32 },       // -1 = final norm
    FusedQkvMatmul { layer: u32 },
    BiasAdd { layer: u32 },
    RoPE { layer: u32 },
    Attention { layer: u32 },
    OutputProjection { layer: u32 },
    ResidualAdd { layer: u32, stage: u32 },
    FusedGateUpMatmul { layer: u32 },
    SiLU { layer: u32 },
    ElementwiseMul { layer: u32 },
    DownProjection { layer: u32 },
    LmHead,
}

pub struct SparsityStats {
    pub total_params: usize,
    pub pruned_params: usize,
    pub threshold: f64,
}

pub struct V2ModelWeights {
    pub config: QwenConfig,
    pub token_embd: Vec<f64>,
    pub output_norm: Vec<f64>,
    pub sm_output: SmTensor,
    pub layers: Vec<V2LayerWeights>,
    pub sparsity: SparsityStats,
    pub exec_graph: Vec<ExecOp>,
}

// ─── Compile: ModelWeights → V2ModelWeights ─────────────────────────────────

/// Default pruning threshold: entries with ln|w| < this are set to -∞.
/// exp(-30) ≈ 9.4e-14, negligible in a sum of 896+ terms relative to f64 eps.
pub const DEFAULT_PRUNE_THRESHOLD: f64 = -30.0;

/// Compile v1 ModelWeights into v2 format with fusion + pruning.
pub fn compile_v2(weights: &ModelWeights, threshold: f64) -> V2ModelWeights {
    let cfg = &weights.config;
    let d = cfg.d_model;

    let mut total_params: usize = 0;
    let mut pruned_params: usize = 0;

    // LM head
    let mut sm_output = SmTensor::from_complex(&weights.ln_output);
    total_params += sm_output.len;
    pruned_params += sm_output.prune(threshold);

    // Build execution graph
    let mut exec_graph = Vec::new();

    // Per-layer compilation
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for (i, layer) in weights.layers.iter().enumerate() {
        // Convert Complex64 → SignMag
        let sm_q = SmTensor::from_complex(&layer.ln_q);
        let sm_k = SmTensor::from_complex(&layer.ln_k);
        let sm_v = SmTensor::from_complex(&layer.ln_v);
        let sm_o_raw = SmTensor::from_complex(&layer.ln_o);
        let sm_gate = SmTensor::from_complex(&layer.ln_gate);
        let sm_up = SmTensor::from_complex(&layer.ln_up);
        let sm_down_raw = SmTensor::from_complex(&layer.ln_down);

        // Fuse QKV: concatenate columns
        // Q has q_dim columns, K has kv_dim, V has kv_dim — all with inner=d
        let mut sm_qkv = SmTensor::fuse(&[&sm_q, &sm_k, &sm_v], d);

        // Fuse gate+up: concatenate columns
        // gate has d_ff columns, up has d_ff — all with inner=d
        let mut sm_gate_up = SmTensor::fuse(&[&sm_gate, &sm_up], d);

        let mut sm_o = SmTensor { ..sm_o_raw };
        let mut sm_down = SmTensor { ..sm_down_raw };

        // Apply pruning
        total_params += sm_qkv.len + sm_o.len + sm_gate_up.len + sm_down.len;
        pruned_params += sm_qkv.prune(threshold);
        pruned_params += sm_o.prune(threshold);
        pruned_params += sm_gate_up.prune(threshold);
        pruned_params += sm_down.prune(threshold);

        layers.push(V2LayerWeights {
            sm_qkv,
            sm_o,
            sm_gate_up,
            sm_down,
            q_bias: layer.q_bias.clone(),
            k_bias: layer.k_bias.clone(),
            v_bias: layer.v_bias.clone(),
            attn_norm: layer.attn_norm.clone(),
            ffn_norm: layer.ffn_norm.clone(),
        });

        // Execution graph for this layer
        let li = i as u32;
        exec_graph.extend([
            ExecOp::RmsNorm { layer: i as i32 },
            ExecOp::FusedQkvMatmul { layer: li },
            ExecOp::BiasAdd { layer: li },
            ExecOp::RoPE { layer: li },
            ExecOp::Attention { layer: li },
            ExecOp::OutputProjection { layer: li },
            ExecOp::ResidualAdd { layer: li, stage: 0 },
            ExecOp::RmsNorm { layer: i as i32 },
            ExecOp::FusedGateUpMatmul { layer: li },
            ExecOp::SiLU { layer: li },
            ExecOp::ElementwiseMul { layer: li },
            ExecOp::DownProjection { layer: li },
            ExecOp::ResidualAdd { layer: li, stage: 1 },
        ]);

        if (i + 1) % 8 == 0 || i == cfg.n_layers - 1 {
            eprintln!("    Compiled layer {}/{}", i + 1, cfg.n_layers);
        }
    }

    // Final ops
    exec_graph.push(ExecOp::RmsNorm { layer: -1 });
    exec_graph.push(ExecOp::LmHead);

    let sparsity = SparsityStats {
        total_params,
        pruned_params,
        threshold,
    };

    V2ModelWeights {
        config: cfg.clone(),
        token_embd: weights.token_embd.clone(),
        output_norm: weights.output_norm.clone(),
        sm_output,
        layers,
        sparsity,
        exec_graph,
    }
}

// ─── V2 Inference ───────────────────────────────────────────────────────────

/// Single-token forward pass using v2 weights (fused QKV, fused gate+up, sign+mag).
pub fn v2_forward_one(
    token_id: usize,
    pos: usize,
    weights: &V2ModelWeights,
    rope: &RopeCache,
    kv_cache: &mut KVCache,
) -> Vec<f64> {
    let cfg = &weights.config;
    let d = cfg.d_model;

    // 1. Token embedding
    let mut x: Vec<f64> = weights.token_embd[token_id * d..(token_id + 1) * d].to_vec();

    // 2. Transformer layers
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        x = v2_transformer_layer_one(&x, layer, cfg, rope, pos,
                                      &mut kv_cache.layers[layer_idx]);
    }

    // 3. Final RMSNorm
    let normed = eml_rms_norm(&x, &weights.output_norm, cfg.rms_norm_eps);

    // 4. LM head: (1, d_model) @ (d_model, vocab)
    build_matmul_sm_precomp(&normed, &weights.sm_output, 1, d, cfg.vocab_size)
}

fn v2_transformer_layer_one(
    x: &[f64],
    layer: &V2LayerWeights,
    cfg: &QwenConfig,
    rope: &RopeCache,
    pos: usize,
    kv: &mut LayerKVCache,
) -> Vec<f64> {
    // Pre-attention RMSNorm
    let normed = eml_rms_norm(x, &layer.attn_norm, cfg.rms_norm_eps);

    // Fused QKV attention
    let attn_out = v2_gqa_attention_one(&normed, layer, cfg, rope, pos, kv);

    // Residual
    let mut x2 = eml_add_vec(x, &attn_out);

    // Pre-FFN RMSNorm
    let normed2 = eml_rms_norm(&x2, &layer.ffn_norm, cfg.rms_norm_eps);

    // Fused SwiGLU FFN
    let ffn_out = v2_swiglu_ffn(&normed2, layer, cfg);

    // Residual
    x2 = eml_add_vec(&x2, &ffn_out);
    x2
}

fn v2_gqa_attention_one(
    x: &[f64],
    layer: &V2LayerWeights,
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
    let qkv_dim = q_dim + kv_dim + kv_dim;

    // ── Fused QKV matmul: single call, ln(x) computed once ──
    let qkv = build_matmul_sm_precomp(x, &layer.sm_qkv, 1, d, qkv_dim);

    // Split into Q, K, V
    let mut q = qkv[..q_dim].to_vec();
    let mut k_new = qkv[q_dim..q_dim + kv_dim].to_vec();
    let mut v_new = qkv[q_dim + kv_dim..].to_vec();

    // Add bias
    for (q_val, &bias) in q.iter_mut().zip(&layer.q_bias) {
        *q_val += bias;
    }
    for j in 0..kv_dim {
        k_new[j] += layer.k_bias[j];
        v_new[j] += layer.v_bias[j];
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
    let t = kv.len;

    // Attention: single query against all cached K,V
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

    // Output projection
    build_matmul_sm_precomp(&out, &layer.sm_o, 1, q_dim, d)
}

fn v2_swiglu_ffn(
    x: &[f64],
    layer: &V2LayerWeights,
    cfg: &QwenConfig,
) -> Vec<f64> {
    let d = cfg.d_model;
    let d_ff = cfg.d_ff;

    // ── Fused gate+up matmul: single call, ln(x) computed once ──
    let gate_up = build_matmul_sm_precomp(x, &layer.sm_gate_up, 1, d, 2 * d_ff);

    // Split
    let gate = &gate_up[..d_ff];
    let up = &gate_up[d_ff..];

    // SwiGLU: silu(gate) * up
    let gate_activated = eml_silu(gate);
    let hidden = eml_mul_vec(&gate_activated, up);

    // Down projection
    build_matmul_sm_precomp(&hidden, &layer.sm_down, 1, d_ff, d)
}

// ─── V2 Generation ──────────────────────────────────────────────────────────

pub fn v2_generate(
    prompt: &[usize],
    weights: &V2ModelWeights,
    rope: &RopeCache,
    max_new: usize,
) -> Vec<usize> {
    let cfg = &weights.config;
    let mut ids = prompt.to_vec();
    let max_len = cfg.max_seq_len.min(prompt.len() + max_new + 16);
    let mut kv_cache = KVCache::new(cfg, max_len);

    // Prefill
    eprintln!("  Prefilling {} prompt tokens...", prompt.len());
    let mut _last_logits = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        _last_logits = v2_forward_one(tok, i, weights, rope, &mut kv_cache);
        if (i + 1) % 10 == 0 || i == prompt.len() - 1 {
            eprint!("\r  Prefilled {}/{}", i + 1, prompt.len());
        }
    }
    eprintln!();

    // Decode
    for step in 0..max_new {
        let logits = if step == 0 {
            _last_logits.clone()
        } else {
            let last_tok = *ids.last().unwrap();
            let pos = ids.len() - 1;
            v2_forward_one(last_tok, pos, weights, rope, &mut kv_cache)
        };

        let next_id = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;

        ids.push(next_id);
        eprint!("\r  Generated {}/{} tokens", step + 1, max_new);

        // EOS check
        if next_id == 151645 || next_id == 151643 {
            break;
        }
    }
    eprintln!();

    ids
}
