//! .eml compiled model format — precomputed ln(weights) for instant load.
//!
//! The .eml format stores the EML-ready representation of a model:
//! - Config (model architecture parameters)
//! - Tokenizer (vocab + BPE merges)
//! - Precomputed ln(W) as Complex64 for all weight matrices
//! - Biases, norms, embeddings as f64
//!
//! Raw weight matrices (f64) are NOT stored — they are only needed to
//! compute ln(W), which is done at compile time. This saves ~50% of the
//! file size vs storing both raw + precomputed.
//!
//! Format:
//!   [Header]  magic "EML1" | version u32 | config fields | section offsets
//!   [Tokenizer]  vocab strings + BPE merges
//!   [Weights]  precomputed tensors in fixed order
//!
//! All multi-byte values are little-endian. All arrays are length-prefixed
//! with a u64 element count.

use crate::engine::{ModelWeights, LayerWeights, QwenConfig};
use crate::eml_v2::*;
use crate::tokenizer::Tokenizer;
use num_complex::Complex64;
use std::io::{self, Read, Write, BufWriter, BufReader};
use std::fs::File;

const MAGIC: &[u8; 4] = b"EML1";
const VERSION: u32 = 1;

// ─── Writer helpers ─────────────────────────────────────────────────────────

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f64(w: &mut impl Write, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f64_array(w: &mut impl Write, data: &[f64]) -> io::Result<()> {
    write_u64(w, data.len() as u64)?;
    for &v in data {
        write_f64(w, v)?;
    }
    Ok(())
}

fn write_complex_array(w: &mut impl Write, data: &[Complex64]) -> io::Result<()> {
    write_u64(w, data.len() as u64)?;
    for c in data {
        write_f64(w, c.re)?;
        write_f64(w, c.im)?;
    }
    Ok(())
}

fn write_string(w: &mut impl Write, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    write_u32(w, bytes.len() as u32)?;
    w.write_all(bytes)
}

// ─── Reader helpers ─────────────────────────────────────────────────────────

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_f64_array(r: &mut impl Read) -> io::Result<Vec<f64>> {
    let len = read_u64(r)? as usize;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(read_f64(r)?);
    }
    Ok(data)
}

fn read_complex_array(r: &mut impl Read) -> io::Result<Vec<Complex64>> {
    let len = read_u64(r)? as usize;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        let re = read_f64(r)?;
        let im = read_f64(r)?;
        data.push(Complex64::new(re, im));
    }
    Ok(data)
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ─── Compile: GGUF → .eml ──────────────────────────────────────────────────

/// Compile a model (weights + tokenizer) to .eml format.
pub fn compile_to_eml(
    weights: &ModelWeights,
    tokenizer: &Tokenizer,
    path: &str,
) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    // ── Header ──
    w.write_all(MAGIC)?;
    write_u32(&mut w, VERSION)?;

    // Config
    let cfg = &weights.config;
    write_u32(&mut w, cfg.vocab_size as u32)?;
    write_u32(&mut w, cfg.n_layers as u32)?;
    write_u32(&mut w, cfg.n_heads as u32)?;
    write_u32(&mut w, cfg.n_kv_heads as u32)?;
    write_u32(&mut w, cfg.d_model as u32)?;
    write_u32(&mut w, cfg.d_ff as u32)?;
    write_f64(&mut w, cfg.rope_freq_base)?;
    write_f64(&mut w, cfg.rms_norm_eps)?;
    write_u32(&mut w, cfg.max_seq_len as u32)?;
    write_u32(&mut w, cfg.d_head as u32)?;

    // ── Tokenizer ──
    write_u32(&mut w, tokenizer.vocab.len() as u32)?;
    write_u32(&mut w, tokenizer.merges.len() as u32)?;
    write_u32(&mut w, tokenizer.bos_id as u32)?;
    write_u32(&mut w, tokenizer.eos_id as u32)?;

    // Vocab strings
    for tok in &tokenizer.vocab {
        write_string(&mut w, tok)?;
    }

    // Merges: need to serialize in rank order
    let mut merges_sorted: Vec<(&(String, String), &usize)> =
        tokenizer.merges.iter().collect();
    merges_sorted.sort_by_key(|(_, rank)| **rank);
    for ((left, right), _) in merges_sorted {
        write_string(&mut w, left)?;
        write_string(&mut w, right)?;
    }

    // ── Global weights ──
    write_f64_array(&mut w, &weights.token_embd)?;
    write_f64_array(&mut w, &weights.output_norm)?;
    write_complex_array(&mut w, &weights.ln_output)?;

    // ── Per-layer weights ──
    for layer in &weights.layers {
        // Precomputed ln(W) — Complex64 arrays
        write_complex_array(&mut w, &layer.ln_q)?;
        write_complex_array(&mut w, &layer.ln_k)?;
        write_complex_array(&mut w, &layer.ln_v)?;
        write_complex_array(&mut w, &layer.ln_o)?;
        write_complex_array(&mut w, &layer.ln_gate)?;
        write_complex_array(&mut w, &layer.ln_up)?;
        write_complex_array(&mut w, &layer.ln_down)?;

        // Biases — f64
        write_f64_array(&mut w, &layer.q_bias)?;
        write_f64_array(&mut w, &layer.k_bias)?;
        write_f64_array(&mut w, &layer.v_bias)?;

        // Norms — f64
        write_f64_array(&mut w, &layer.attn_norm)?;
        write_f64_array(&mut w, &layer.ffn_norm)?;
    }

    w.flush()?;
    Ok(())
}

// ─── Load: .eml → ModelWeights + Tokenizer ──────────────────────────────────

/// Load a compiled .eml model. Returns (weights, tokenizer).
pub fn load_eml(path: &str) -> io::Result<(ModelWeights, Tokenizer)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    // ── Header ──
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Not an EML file (magic: {:?})", magic),
        ));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported EML version: {version} (expected {VERSION})"),
        ));
    }

    // Config
    let config = QwenConfig {
        vocab_size: read_u32(&mut r)? as usize,
        n_layers: read_u32(&mut r)? as usize,
        n_heads: read_u32(&mut r)? as usize,
        n_kv_heads: read_u32(&mut r)? as usize,
        d_model: read_u32(&mut r)? as usize,
        d_ff: read_u32(&mut r)? as usize,
        rope_freq_base: read_f64(&mut r)?,
        rms_norm_eps: read_f64(&mut r)?,
        max_seq_len: read_u32(&mut r)? as usize,
        d_head: read_u32(&mut r)? as usize,
        eos_token_id: 151643,
        eot_token_id: 151645,
    };

    // ── Tokenizer ──
    let vocab_size = read_u32(&mut r)? as usize;
    let merges_count = read_u32(&mut r)? as usize;
    let bos_id = read_u32(&mut r)? as usize;
    let eos_id = read_u32(&mut r)? as usize;

    let mut vocab = Vec::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        vocab.push(read_string(&mut r)?);
    }

    let mut token_to_id = std::collections::HashMap::with_capacity(vocab.len());
    for (i, tok) in vocab.iter().enumerate() {
        token_to_id.insert(tok.clone(), i);
    }

    let mut merges = std::collections::HashMap::with_capacity(merges_count);
    for rank in 0..merges_count {
        let left = read_string(&mut r)?;
        let right = read_string(&mut r)?;
        merges.insert((left, right), rank);
    }

    let im_start_id = token_to_id.get("<|im_start|>").copied();
    let im_end_id = token_to_id.get("<|im_end|>").copied();

    let byte_to_char = crate::tokenizer::byte_to_unicode_pub();
    let char_to_byte = byte_to_char.iter().map(|(&b, &c)| (c, b)).collect();

    let tokenizer = Tokenizer {
        vocab,
        token_to_id,
        merges,
        byte_to_char,
        char_to_byte,
        bos_id,
        eos_id,
        im_start_id,
        im_end_id,
    };

    // ── Global weights ──
    let token_embd = read_f64_array(&mut r)?;
    let output_norm = read_f64_array(&mut r)?;
    let ln_output = read_complex_array(&mut r)?;

    // ── Per-layer weights ──
    let mut layers = Vec::with_capacity(config.n_layers);
    for _ in 0..config.n_layers {
        let ln_q = read_complex_array(&mut r)?;
        let ln_k = read_complex_array(&mut r)?;
        let ln_v = read_complex_array(&mut r)?;
        let ln_o = read_complex_array(&mut r)?;
        let ln_gate = read_complex_array(&mut r)?;
        let ln_up = read_complex_array(&mut r)?;
        let ln_down = read_complex_array(&mut r)?;

        let q_bias = read_f64_array(&mut r)?;
        let k_bias = read_f64_array(&mut r)?;
        let v_bias = read_f64_array(&mut r)?;

        let attn_norm = read_f64_array(&mut r)?;
        let ffn_norm = read_f64_array(&mut r)?;

        layers.push(LayerWeights {
            // Raw weights not needed — fill with empty vecs
            q_weight: Vec::new(),
            k_weight: Vec::new(),
            v_weight: Vec::new(),
            o_weight: Vec::new(),
            gate_weight: Vec::new(),
            up_weight: Vec::new(),
            down_weight: Vec::new(),
            q_bias, k_bias, v_bias,
            attn_norm, ffn_norm,
            ln_q, ln_k, ln_v, ln_o, ln_gate, ln_up, ln_down,
        });
    }

    let weights = ModelWeights {
        config,
        token_embd,
        output_norm,
        output_weight: Vec::new(), // not needed — only ln_output is used
        ln_output,
        layers,
    };

    Ok((weights, tokenizer))
}

/// Check if a path is an .eml file.
pub fn is_eml_file(path: &str) -> bool {
    path.ends_with(".eml")
}

// ═══════════════════════════════════════════════════════════════════════════
//  EML v2 FORMAT
// ═══════════════════════════════════════════════════════════════════════════
//
// Magic: "EML2", version: 2
//
// Sign+magnitude encoding:
//   magnitudes: f64[] (length-prefixed)
//   signs: packed bitmap u8[] (ceil(len/8) bytes)
//
// Fused weights: QKV concatenated, gate+up concatenated
// Sparsity stats: total_params, pruned_params, threshold
// Execution graph: op count + serialized ops

const MAGIC_V2: &[u8; 4] = b"EML2";
const VERSION_V2: u32 = 2;

fn write_sm_tensor(w: &mut impl Write, t: &SmTensor) -> io::Result<()> {
    write_u64(w, t.len as u64)?;
    // Magnitudes (f64 array, without redundant length prefix — len is above)
    for &m in &t.magnitudes {
        write_f64(w, m)?;
    }
    // Signs as packed bitmap
    let packed = t.pack_signs();
    write_u32(w, packed.len() as u32)?;
    w.write_all(&packed)?;
    Ok(())
}

fn read_sm_tensor(r: &mut impl Read) -> io::Result<SmTensor> {
    let len = read_u64(r)? as usize;
    let mut magnitudes = Vec::with_capacity(len);
    for _ in 0..len {
        magnitudes.push(read_f64(r)?);
    }
    let packed_len = read_u32(r)? as usize;
    let mut packed = vec![0u8; packed_len];
    r.read_exact(&mut packed)?;
    Ok(SmTensor::from_packed(magnitudes, &packed, len))
}

fn write_exec_graph(w: &mut impl Write, graph: &[ExecOp]) -> io::Result<()> {
    write_u32(w, graph.len() as u32)?;
    for op in graph {
        match op {
            ExecOp::RmsNorm { layer } => { w.write_all(&[0])?; write_u32(w, *layer as u32)?; }
            ExecOp::FusedQkvMatmul { layer } => { w.write_all(&[1])?; write_u32(w, *layer)?; }
            ExecOp::BiasAdd { layer } => { w.write_all(&[2])?; write_u32(w, *layer)?; }
            ExecOp::RoPE { layer } => { w.write_all(&[3])?; write_u32(w, *layer)?; }
            ExecOp::Attention { layer } => { w.write_all(&[4])?; write_u32(w, *layer)?; }
            ExecOp::OutputProjection { layer } => { w.write_all(&[5])?; write_u32(w, *layer)?; }
            ExecOp::ResidualAdd { layer, stage } => { w.write_all(&[6])?; write_u32(w, *layer)?; write_u32(w, *stage)?; }
            ExecOp::FusedGateUpMatmul { layer } => { w.write_all(&[7])?; write_u32(w, *layer)?; }
            ExecOp::SiLU { layer } => { w.write_all(&[8])?; write_u32(w, *layer)?; }
            ExecOp::ElementwiseMul { layer } => { w.write_all(&[9])?; write_u32(w, *layer)?; }
            ExecOp::DownProjection { layer } => { w.write_all(&[10])?; write_u32(w, *layer)?; }
            ExecOp::LmHead => { w.write_all(&[11])?; }
        }
    }
    Ok(())
}

fn read_exec_graph(r: &mut impl Read) -> io::Result<Vec<ExecOp>> {
    let count = read_u32(r)? as usize;
    let mut graph = Vec::with_capacity(count);
    for _ in 0..count {
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;
        let op = match tag[0] {
            0 => ExecOp::RmsNorm { layer: read_u32(r)? as i32 },
            1 => ExecOp::FusedQkvMatmul { layer: read_u32(r)? },
            2 => ExecOp::BiasAdd { layer: read_u32(r)? },
            3 => ExecOp::RoPE { layer: read_u32(r)? },
            4 => ExecOp::Attention { layer: read_u32(r)? },
            5 => ExecOp::OutputProjection { layer: read_u32(r)? },
            6 => { let l = read_u32(r)?; let s = read_u32(r)?; ExecOp::ResidualAdd { layer: l, stage: s } }
            7 => ExecOp::FusedGateUpMatmul { layer: read_u32(r)? },
            8 => ExecOp::SiLU { layer: read_u32(r)? },
            9 => ExecOp::ElementwiseMul { layer: read_u32(r)? },
            10 => ExecOp::DownProjection { layer: read_u32(r)? },
            11 => ExecOp::LmHead,
            other => return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Unknown exec op tag: {other}"))),
        };
        graph.push(op);
    }
    Ok(graph)
}

/// Compile V2ModelWeights + Tokenizer to .eml v2 format.
pub fn compile_to_eml_v2(
    weights: &V2ModelWeights,
    tokenizer: &Tokenizer,
    path: &str,
) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    // ── Header ──
    w.write_all(MAGIC_V2)?;
    write_u32(&mut w, VERSION_V2)?;

    // Config (same as v1)
    let cfg = &weights.config;
    write_u32(&mut w, cfg.vocab_size as u32)?;
    write_u32(&mut w, cfg.n_layers as u32)?;
    write_u32(&mut w, cfg.n_heads as u32)?;
    write_u32(&mut w, cfg.n_kv_heads as u32)?;
    write_u32(&mut w, cfg.d_model as u32)?;
    write_u32(&mut w, cfg.d_ff as u32)?;
    write_f64(&mut w, cfg.rope_freq_base)?;
    write_f64(&mut w, cfg.rms_norm_eps)?;
    write_u32(&mut w, cfg.max_seq_len as u32)?;
    write_u32(&mut w, cfg.d_head as u32)?;

    // Sparsity stats
    write_u64(&mut w, weights.sparsity.total_params as u64)?;
    write_u64(&mut w, weights.sparsity.pruned_params as u64)?;
    write_f64(&mut w, weights.sparsity.threshold)?;

    // ── Tokenizer (same as v1) ──
    write_u32(&mut w, tokenizer.vocab.len() as u32)?;
    write_u32(&mut w, tokenizer.merges.len() as u32)?;
    write_u32(&mut w, tokenizer.bos_id as u32)?;
    write_u32(&mut w, tokenizer.eos_id as u32)?;

    for tok in &tokenizer.vocab {
        write_string(&mut w, tok)?;
    }

    let mut merges_sorted: Vec<(&(String, String), &usize)> =
        tokenizer.merges.iter().collect();
    merges_sorted.sort_by_key(|(_, rank)| **rank);
    for ((left, right), _) in merges_sorted {
        write_string(&mut w, left)?;
        write_string(&mut w, right)?;
    }

    // ── Execution graph ──
    write_exec_graph(&mut w, &weights.exec_graph)?;

    // ── Global weights ──
    write_f64_array(&mut w, &weights.token_embd)?;
    write_f64_array(&mut w, &weights.output_norm)?;
    write_sm_tensor(&mut w, &weights.sm_output)?;

    // ── Per-layer weights (fused, sign+mag) ──
    for layer in &weights.layers {
        write_sm_tensor(&mut w, &layer.sm_qkv)?;
        write_sm_tensor(&mut w, &layer.sm_o)?;
        write_sm_tensor(&mut w, &layer.sm_gate_up)?;
        write_sm_tensor(&mut w, &layer.sm_down)?;

        write_f64_array(&mut w, &layer.q_bias)?;
        write_f64_array(&mut w, &layer.k_bias)?;
        write_f64_array(&mut w, &layer.v_bias)?;
        write_f64_array(&mut w, &layer.attn_norm)?;
        write_f64_array(&mut w, &layer.ffn_norm)?;
    }

    w.flush()?;
    Ok(())
}

/// Load a compiled .eml v2 model. Returns (V2ModelWeights, Tokenizer).
pub fn load_eml_v2(path: &str) -> io::Result<(V2ModelWeights, Tokenizer)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    // ── Header ──
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC_V2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Not an EML v2 file (magic: {:?})", magic),
        ));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION_V2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported EML version: {version} (expected {VERSION_V2})"),
        ));
    }

    let config = QwenConfig {
        vocab_size: read_u32(&mut r)? as usize,
        n_layers: read_u32(&mut r)? as usize,
        n_heads: read_u32(&mut r)? as usize,
        n_kv_heads: read_u32(&mut r)? as usize,
        d_model: read_u32(&mut r)? as usize,
        d_ff: read_u32(&mut r)? as usize,
        rope_freq_base: read_f64(&mut r)?,
        rms_norm_eps: read_f64(&mut r)?,
        max_seq_len: read_u32(&mut r)? as usize,
        d_head: read_u32(&mut r)? as usize,
        eos_token_id: 151643,
        eot_token_id: 151645,
    };

    let sparsity = SparsityStats {
        total_params: read_u64(&mut r)? as usize,
        pruned_params: read_u64(&mut r)? as usize,
        threshold: read_f64(&mut r)?,
    };

    // ── Tokenizer ──
    let vocab_size = read_u32(&mut r)? as usize;
    let merges_count = read_u32(&mut r)? as usize;
    let bos_id = read_u32(&mut r)? as usize;
    let eos_id = read_u32(&mut r)? as usize;

    let mut vocab = Vec::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        vocab.push(read_string(&mut r)?);
    }

    let mut token_to_id = std::collections::HashMap::with_capacity(vocab.len());
    for (i, tok) in vocab.iter().enumerate() {
        token_to_id.insert(tok.clone(), i);
    }

    let mut merges = std::collections::HashMap::with_capacity(merges_count);
    for rank in 0..merges_count {
        let left = read_string(&mut r)?;
        let right = read_string(&mut r)?;
        merges.insert((left, right), rank);
    }

    let im_start_id = token_to_id.get("<|im_start|>").copied();
    let im_end_id = token_to_id.get("<|im_end|>").copied();

    let byte_to_char = crate::tokenizer::byte_to_unicode_pub();
    let char_to_byte = byte_to_char.iter().map(|(&b, &c)| (c, b)).collect();

    let tokenizer = Tokenizer {
        vocab,
        token_to_id,
        merges,
        byte_to_char,
        char_to_byte,
        bos_id,
        eos_id,
        im_start_id,
        im_end_id,
    };

    // ── Execution graph ──
    let exec_graph = read_exec_graph(&mut r)?;

    // ── Global weights ──
    let token_embd = read_f64_array(&mut r)?;
    let output_norm = read_f64_array(&mut r)?;
    let sm_output = read_sm_tensor(&mut r)?;

    // ── Per-layer weights ──
    let mut layers = Vec::with_capacity(config.n_layers);
    for _ in 0..config.n_layers {
        let sm_qkv = read_sm_tensor(&mut r)?;
        let sm_o = read_sm_tensor(&mut r)?;
        let sm_gate_up = read_sm_tensor(&mut r)?;
        let sm_down = read_sm_tensor(&mut r)?;

        let q_bias = read_f64_array(&mut r)?;
        let k_bias = read_f64_array(&mut r)?;
        let v_bias = read_f64_array(&mut r)?;
        let attn_norm = read_f64_array(&mut r)?;
        let ffn_norm = read_f64_array(&mut r)?;

        layers.push(V2LayerWeights {
            sm_qkv, sm_o, sm_gate_up, sm_down,
            q_bias, k_bias, v_bias,
            attn_norm, ffn_norm,
        });
    }

    let weights = V2ModelWeights {
        config,
        token_embd,
        output_norm,
        sm_output,
        layers,
        sparsity,
        exec_graph,
    };

    Ok((weights, tokenizer))
}

/// Detect format version from file magic bytes.
pub fn detect_eml_version(path: &str) -> io::Result<u32> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    match &magic {
        b"EML1" => Ok(1),
        b"EML2" => Ok(2),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown EML format (magic: {:?})", magic),
        )),
    }
}
