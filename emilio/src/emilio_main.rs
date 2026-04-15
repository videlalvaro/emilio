//! emilio — EML inference engine
//!
//! Usage:
//!   emilio <model.gguf|model.eml> [--explore | --generate <text> | --chat <message>]
//!   emilio <model.gguf> --compile [output.eml]       (v1 format)
//!   emilio <model.gguf> --compile-v2 [output.eml]    (v2: sign+mag, fused, pruned)
//!   emilio <model.gguf> --generate --gpu "text"       (Metal GPU acceleration)

use emilio::gguf::GGUFFile;
use emilio::engine::*;
use emilio::eml_format;
use emilio::eml_v2::{self as v2, v2_generate};
use emilio::tokenizer::Tokenizer;
use std::time::Instant;
#[cfg(feature = "metal")]
use emilio::metal_eml::{MetalContext, GpuModelWeights, ScratchPool};

// ─── Result box formatter ───────────────────────────────────────────────────

/// Count display width (number of terminal columns).
fn display_width(s: &str) -> usize {
    s.chars().count()
}

/// Escape control characters for single-line display.
fn escape_for_box(s: &str) -> String {
    s.replace('\n', " ").replace('\r', "").replace('\t', " ")
}

/// Word-wrap `text` into lines of at most `max_w` display columns.
fn wrap_lines(text: &str, max_w: usize) -> Vec<String> {
    if max_w == 0 { return vec![text.to_string()]; }
    let mut lines = Vec::new();
    let mut cur = String::new();
    let mut cur_w: usize = 0;
    for word in text.split(' ') {
        let ww = display_width(word);
        if cur.is_empty() {
            cur = word.to_string();
            cur_w = ww;
        } else if cur_w + 1 + ww <= max_w {
            cur.push(' ');
            cur.push_str(word);
            cur_w += 1 + ww;
        } else {
            lines.push(cur);
            cur = word.to_string();
            cur_w = ww;
        }
    }
    if !cur.is_empty() { lines.push(cur); }
    if lines.is_empty() { lines.push(String::new()); }
    lines
}

/// Get terminal width, falling back to 100.
fn term_width() -> usize {
    // Try $COLUMNS first, then a sensible default
    std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100)
}

/// Print a neatly formatted result box that auto-sizes to content,
/// word-wrapping long text to fit the terminal.
///
/// ```text
///   ┌─────────────────────────────────────────────────────────────────────┐
///   │  Prompt:     What is the capital of Uruguay and why is it          │
///   │              important in South American history?                   │
///   │  Generated:  The capital of Uruguay is Montevideo.                  │
///   │  Tokens:     16 (ids: [17, 10, 17, ...])                           │
///   │  Time:       7.18s  (2.23 tok/s)                                   │
///   └─────────────────────────────────────────────────────────────────────┘
/// ```
fn print_result_box(prompt: &str, generated: &str, token_ids: &[usize], secs: f64) {
    let gen_esc = escape_for_box(generated);
    let n_tok = token_ids.len();
    let tok_s = if secs > 0.0 { n_tok as f64 / secs } else { 0.0 };

    // Build token ID preview — truncate if too many
    let ids_str = if n_tok <= 8 {
        format!("{:?}", token_ids)
    } else {
        let head: Vec<String> = token_ids[..6].iter().map(|t| t.to_string()).collect();
        format!("[{}, ...]", head.join(", "))
    };
    let tok_line = format!("{n_tok} (ids: {ids_str})");
    let time_line = format!("{secs:.2}s  ({tok_s:.2} tok/s)");

    // Label column: "Generated:  " = 13 chars (widest label + padding)
    let label_w: usize = 13;
    //   Box chrome: "  │  " (5) + label_w + "  │" (3) = 21 fixed chars
    let chrome: usize = 5 + label_w + 3;
    let max_val_w = term_width().saturating_sub(chrome).max(20);

    // Rows: (label, value)
    // Prompt preserves newlines (split into sub-lines); others are single-line.
    let rows: Vec<(&str, Vec<&str>)> = vec![
        ("Prompt:", prompt.lines().collect()),
        ("Generated:", vec![&gen_esc]),
        ("Tokens:", vec![&tok_line]),
        ("Time:", vec![&time_line]),
    ];

    // Word-wrap each value and build display lines
    let mut all_lines: Vec<String> = Vec::new();
    let mut max_row_w: usize = 0;
    for (label, sub_lines) in &rows {
        let mut first = true;
        for sub in sub_lines {
            let cleaned = escape_for_box(sub);
            let wrapped = wrap_lines(&cleaned, max_val_w);
            for (i, line) in wrapped.iter().enumerate() {
                let formatted = if first && i == 0 {
                    let pad = label_w.saturating_sub(display_width(label));
                    format!("{label}{}{line}", " ".repeat(pad))
                } else {
                    format!("{}{line}", " ".repeat(label_w))
                };
                let w = display_width(&formatted);
                if w > max_row_w { max_row_w = w; }
                all_lines.push(formatted);
            }
            first = false;
        }
    }

    // Box inner width = widest row + 4 (2 spaces each side)
    let inner = max_row_w + 4;

    // Draw
    println!();
    println!("  ┌{}┐", "─".repeat(inner));
    for row in &all_lines {
        let pad = inner.saturating_sub(display_width(row) + 4);
        println!("  │  {row}{}  │", " ".repeat(pad));
    }
    println!("  └{}┘", "─".repeat(inner));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: emilio <model.gguf|model.eml> [--explore | --generate <text> | --compile-v2 [out.eml]]");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  emilio model.gguf --explore");
        eprintln!("  emilio model.gguf --generate \"Hello world\"");
        eprintln!("  emilio model.gguf --chat \"What is 2+2?\"");
        eprintln!("  emilio model.gguf --tokens \"1,2,3\"     (raw token IDs)");
        eprintln!("  emilio model.gguf --compile              (v1: writes model.eml)");
        eprintln!("  emilio model.gguf --compile-v2            (v2: sign+mag + fused + pruned)");
        eprintln!("  emilio model.eml  --generate \"Hello\"     (load compiled, auto-detect v1/v2)");
        #[cfg(feature = "metal")]
        eprintln!("  emilio model.gguf --generate --gpu \"Hello\" (Metal GPU acceleration)");
        std::process::exit(1);
    }

    // Check for --gpu flag anywhere in args
    let use_gpu = args.iter().any(|a| a == "--gpu");
    let args_filtered: Vec<String> = args.iter()
        .filter(|a| a.as_str() != "--gpu")
        .cloned()
        .collect();

    let model_path = &args_filtered[1];
    let mode = args_filtered.get(2).map(|s| s.as_str()).unwrap_or("--explore");

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  emilio -- EML inference engine                          ║");
    println!("║  Every result flows through eml(x,y) = exp(x) - ln(y)    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Detect model format ────────────────────────────────────────
    #[cfg(not(feature = "metal"))]
    if use_gpu {
        eprintln!("Error: --gpu requires the 'metal' feature. Build with: cargo build --features metal");
        std::process::exit(1);
    }

    if eml_format::is_eml_file(model_path) {
        // .eml compiled format — auto-detect v1 vs v2
        match eml_format::detect_eml_version(model_path) {
            Ok(1) => run_from_eml(model_path, mode, &args),
            Ok(2) => run_from_eml_v2(model_path, mode, &args),
            Ok(v) => {
                eprintln!("Unsupported EML version: {v}");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error reading .eml: {e}");
                std::process::exit(1);
            }
        }
    } else {
        // GGUF format — original path
        run_from_gguf(model_path, mode, &args_filtered, use_gpu);
    }
}

fn run_from_eml(model_path: &str, mode: &str, args: &[String]) {
    println!("Loading compiled EML: {model_path}");
    let t0 = Instant::now();
    let (weights, tokenizer) = match eml_format::load_eml(model_path) {
        Ok(wt) => wt,
        Err(e) => {
            eprintln!("Error loading .eml: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded in {load_ms:.0}ms ({} layers, vocab {})",
        weights.config.n_layers, weights.config.vocab_size);
    println!("  Tokenizer: {} tokens, {} merges",
        tokenizer.vocab_size(), tokenizer.merges.len());
    println!();

    match mode {
        "--generate" => {
            let text = args.get(3).map(|s| s.as_str()).unwrap_or("Hello");
            let prompt_ids = tokenizer.encode(text);
            println!("  Tokenized: \"{}\" → {} tokens: {:?}",
                text, prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(20)]);
            generate_with_weights(&weights, &prompt_ids, &tokenizer, None);
        }
        "--chat" => {
            let msg = args.get(3).map(|s| s.as_str()).unwrap_or("Hello");
            let prompt_ids = tokenizer.encode_chat(msg);
            println!("  Chat prompt: \"{msg}\"");
            println!("  Tokenized to {} tokens: {:?}...",
                prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(20)]);
            let chat_display = format!("system: You are a helpful assistant.\nuser: {msg}");
            generate_with_weights(&weights, &prompt_ids, &tokenizer, Some(&chat_display));
        }
        "--tokens" => {
            let tok_str = args.get(3).map(|s| s.as_str()).unwrap_or("1,2,3");
            let prompt: Vec<usize> = tok_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            generate_raw_with_weights(&weights, &prompt, Some(&tokenizer));
        }
        _ => {
            eprintln!("Unknown mode for .eml: {mode} (use --generate, --chat, or --tokens)");
            std::process::exit(1);
        }
    }
}

fn run_from_eml_v2(model_path: &str, mode: &str, args: &[String]) {
    println!("Loading compiled EML v2: {model_path}");
    let t0 = Instant::now();
    let (weights, tokenizer) = match eml_format::load_eml_v2(model_path) {
        Ok(wt) => wt,
        Err(e) => {
            eprintln!("Error loading .eml v2: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Loaded in {load_ms:.0}ms ({} layers, vocab {})",
        weights.config.n_layers, weights.config.vocab_size);
    println!("  Format: v2 (sign+mag, fused QKV + gate_up, pruned)");
    println!("  Sparsity: {}/{} params pruned ({:.1}%, threshold={:.0})",
        weights.sparsity.pruned_params, weights.sparsity.total_params,
        100.0 * weights.sparsity.pruned_params as f64 / weights.sparsity.total_params.max(1) as f64,
        weights.sparsity.threshold);
    println!("  Exec graph: {} ops", weights.exec_graph.len());
    println!("  Tokenizer: {} tokens, {} merges",
        tokenizer.vocab_size(), tokenizer.merges.len());
    println!();

    let rope = RopeCache::new(
        weights.config.d_head,
        weights.config.max_seq_len.min(2048),
        weights.config.rope_freq_base,
    );

    match mode {
        "--generate" => {
            let text = args.get(3).map(|s| s.as_str()).unwrap_or("Hello");
            let prompt_ids = tokenizer.encode(text);
            println!("  Tokenized: \"{}\" → {} tokens: {:?}",
                text, prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(20)]);

            println!();
            println!("Running emilio v2 (sign+mag, fused ops, compiled format)...");
            println!("  Prompt: {} tokens", prompt_ids.len());
            println!("  Config: {} layers, {} heads, d_model={}",
                weights.config.n_layers, weights.config.n_heads, weights.config.d_model);

            let max_new = 16;
            let t1 = Instant::now();
            let output = v2_generate(&prompt_ids, &weights, &rope, max_new);
            let gen_s = t1.elapsed().as_secs_f64();

            let generated = &output[prompt_ids.len()..];
            let prompt_text = tokenizer.decode(&prompt_ids);
            let generated_text = tokenizer.decode(generated);

            print_result_box(&prompt_text, &generated_text, generated, gen_s);
            println!("  v2: sign+mag kernel, fused QKV/gate_up, {:.1}% pruned.",
                100.0 * weights.sparsity.pruned_params as f64 / weights.sparsity.total_params.max(1) as f64);
        }
        "--chat" => {
            let msg = args.get(3).map(|s| s.as_str()).unwrap_or("Hello");
            let prompt_ids = tokenizer.encode_chat(msg);
            println!("  Chat prompt: \"{msg}\"");
            println!("  Tokenized to {} tokens: {:?}...",
                prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(20)]);

            println!();
            println!("Running emilio v2 (sign+mag, fused ops, compiled format)...");

            let max_new = 16;
            let t1 = Instant::now();
            let output = v2_generate(&prompt_ids, &weights, &rope, max_new);
            let gen_s = t1.elapsed().as_secs_f64();

            let generated = &output[prompt_ids.len()..];
            let generated_text = tokenizer.decode(generated);

            let chat_display = format!("system: You are a helpful assistant.\nuser: {msg}");
            print_result_box(&chat_display, &generated_text, generated, gen_s);
        }
        _ => {
            eprintln!("Unknown mode for .eml v2: {mode} (use --generate or --chat)");
            std::process::exit(1);
        }
    }
}

fn run_from_gguf(model_path: &str, mode: &str, args: &[String], _use_gpu: bool) {
    // ── Parse GGUF ─────────────────────────────────────────────────
    println!("Loading GGUF: {model_path}");
    let t0 = Instant::now();
    let gguf = match GGUFFile::parse(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error parsing GGUF: {e}");
            std::process::exit(1);
        }
    };
    println!("  Parsed header in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // ── Build tokenizer ────────────────────────────────────────────
    let tokenizer = Tokenizer::from_gguf(&gguf);
    if let Some(ref tok) = tokenizer {
        println!("  Tokenizer: {} tokens, {} merges",
            tok.vocab_size(), tok.merges.len());
    }
    println!();

    match mode {
        "--explore" => explore(&gguf),
        "--compile" => {
            let tok = tokenizer.as_ref().expect("Tokenizer not found in GGUF");
            let out_path = args.get(3).map(|s| s.as_str()).unwrap_or_else(|| {
                // Default: replace .gguf with .eml
                Box::leak(model_path.replace(".gguf", ".eml").into_boxed_str())
            });
            compile_model(&gguf, tok, out_path);
        }
        "--compile-v2" => {
            let tok = tokenizer.as_ref().expect("Tokenizer not found in GGUF");
            let out_path = args.get(3).map(|s| s.as_str()).unwrap_or_else(|| {
                Box::leak(model_path.replace(".gguf", ".eml").into_boxed_str())
            });
            compile_model_v2(&gguf, tok, out_path);
        }
        "--generate" => {
            let text = args.get(3).map(|s| s.as_str()).unwrap_or("Hello");
            let tok = tokenizer.as_ref().expect("Tokenizer not found in GGUF");
            let prompt_ids = tok.encode(text);
            println!("  Tokenized: \"{}\" → {} tokens: {:?}",
                text, prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(20)]);
            #[cfg(feature = "metal")]
            if _use_gpu {
                generate_metal(&gguf, &prompt_ids, tok, None);
                return;
            }
            generate(&gguf, &prompt_ids, tok, None);
        }
        "--chat" => {
            let msg = args.get(3).map(|s| s.as_str()).unwrap_or("Hello");
            let tok = tokenizer.as_ref().expect("Tokenizer not found in GGUF");
            let prompt_ids = tok.encode_chat(msg);
            println!("  Chat prompt: \"{msg}\"");
            println!("  Tokenized to {} tokens: {:?}...",
                prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(20)]);
            let chat_display = format!("system: You are a helpful assistant.\nuser: {msg}");
            #[cfg(feature = "metal")]
            if _use_gpu {
                generate_metal(&gguf, &prompt_ids, tok, Some(&chat_display));
                return;
            }
            generate(&gguf, &prompt_ids, tok, Some(&chat_display));
        }
        "--tokens" => {
            let tok_str = args.get(3).map(|s| s.as_str()).unwrap_or("1,2,3");
            let prompt: Vec<usize> = tok_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let tok = tokenizer.as_ref();
            generate_raw(&gguf, &prompt, tok);
        }
        _ => {
            eprintln!("Unknown mode: {mode}");
            std::process::exit(1);
        }
    }
}

fn compile_model(gguf: &GGUFFile, tok: &Tokenizer, out_path: &str) {
    println!("Compiling model to EML format...");

    let t0 = Instant::now();
    let weights = match ModelWeights::from_gguf(gguf) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error loading weights: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Dequantized + precomputed ln(W) in {load_ms:.0}ms");

    let t1 = Instant::now();
    match eml_format::compile_to_eml(&weights, tok, out_path) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error writing .eml: {e}");
            std::process::exit(1);
        }
    }
    let write_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let file_size = std::fs::metadata(out_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("  Wrote {out_path} ({:.1} MB) in {write_ms:.0}ms",
        file_size as f64 / 1_048_576.0);
    println!();
    println!("  Format: EML v1 (precomputed ln(W) as Complex64)");
    println!("  Contents:");
    println!("    - Config ({} layers, d_model={}, vocab={})",
        weights.config.n_layers, weights.config.d_model, weights.config.vocab_size);
    println!("    - Tokenizer ({} tokens, {} merges)",
        tok.vocab_size(), tok.merges.len());
    println!("    - {} precomputed ln(W) tensors (Complex64)",
        weights.config.n_layers * 7 + 1);
    println!("    - {} bias/norm tensors (f64)",
        weights.config.n_layers * 5 + 2);
    println!();
    println!("  To use: emilio {out_path} --generate \"Hello world\"");
}

fn compile_model_v2(gguf: &GGUFFile, tok: &Tokenizer, out_path: &str) {
    println!("Compiling model to EML v2 format (sign+mag, fused, pruned)...");

    // Step 1: Load GGUF → v1 ModelWeights (dequant + precompute ln)
    let t0 = Instant::now();
    let weights = match ModelWeights::from_gguf(gguf) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error loading weights: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Dequantized + precomputed ln(W) in {load_ms:.0}ms");

    // Step 2: Compile v1 → v2 (fusion + sign+mag + pruning)
    let t1 = Instant::now();
    println!("  Compiling v2 (fusing QKV + gate_up, sign+mag encoding, pruning)...");
    let v2 = v2::compile_v2(&weights, v2::DEFAULT_PRUNE_THRESHOLD);
    let compile_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("  Compiled v2 in {compile_ms:.0}ms");
    println!("    Sparsity: {}/{} params pruned ({:.2}%, threshold={:.0})",
        v2.sparsity.pruned_params, v2.sparsity.total_params,
        100.0 * v2.sparsity.pruned_params as f64 / v2.sparsity.total_params.max(1) as f64,
        v2.sparsity.threshold);
    println!("    Exec graph: {} ops", v2.exec_graph.len());

    // Step 3: Serialize to .eml v2
    let t2 = Instant::now();
    match eml_format::compile_to_eml_v2(&v2, tok, out_path) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error writing .eml v2: {e}");
            std::process::exit(1);
        }
    }
    let write_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let file_size = std::fs::metadata(out_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("  Wrote {out_path} ({:.1} MB) in {write_ms:.0}ms",
        file_size as f64 / 1_048_576.0);
    println!();
    println!("  Format: EML v2");
    println!("  Optimizations:");
    println!("    1. Sign+magnitude: 8.125 bytes/elem vs 16 (~50% file reduction)");
    println!("    2. Fused QKV: 3 matmuls → 1 (shared ln(activation))");
    println!("    3. Fused gate+up: 2 matmuls → 1 (shared ln(activation))");
    println!("    4. Sparse pruning: {:.1}% of weights pruned",
        100.0 * v2.sparsity.pruned_params as f64 / v2.sparsity.total_params.max(1) as f64);
    println!("    5. Execution graph: {} ops stored", v2.exec_graph.len());
    println!();
    println!("  To use: emilio {out_path} --generate \"Hello world\"");
}

fn explore(gguf: &GGUFFile) {
    println!("═══════════════════════════════════════════════════");
    println!("  MODEL METADATA");
    println!("═══════════════════════════════════════════════════");
    gguf.print_summary();

    let config = QwenConfig::from_gguf(gguf);
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  QWEN2 ARCHITECTURE → EML MAPPING");
    println!("═══════════════════════════════════════════════════");
    config.print();

    // Show tensor table
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  TENSOR MAP ({} tensors)", gguf.tensors.len());
    println!("═══════════════════════════════════════════════════");
    println!("  {:50} {:>12} {:>8} {:>10}",
        "Name", "Shape", "Type", "Size");
    println!("  {}", "─".repeat(84));

    for t in &gguf.tensors {
        let shape = t.dims.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("×");
        let size_mb = t.byte_size() as f64 / 1e6;
        println!("  {:50} {:>12} {:>8?} {:>8.2}MB",
            t.name, shape, t.dtype, size_mb);
    }

    // EML complexity analysis
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  EML COMPLEXITY ANALYSIS");
    println!("═══════════════════════════════════════════════════");

    let d = config.d_model;
    let d_ff = config.d_ff;
    let n_heads = config.n_heads;
    let n_kv = config.n_kv_heads;
    let d_head = config.d_head;
    let v = config.vocab_size;
    let n_layers = config.n_layers;

    println!();
    println!("  Per-layer EML ops (seq_len=T, CSE-optimized):");
    println!();

    // QKV projections: 3 matmuls
    let qkv_naive = 3 * d * (n_heads * d_head + 2 * n_kv * d_head); // per token, K=d
    let qkv_cse_per_tok = n_heads * d_head * d + 2 * n_kv * d_head * d; // exp calls
    println!("    QKV projection:  T × {} exp (CSE) vs T × {} (naive)",
        qkv_cse_per_tok / d, qkv_naive);

    // Attention: T² × d_head dot products per head
    let _attn_per_token = n_heads * d_head * 3; // per score: d_head muls
    println!("    Attention:       T² × {} transcendentals per head × {n_heads} heads",
        d_head * 3);

    // FFN: gate + up + down
    let ffn_matmul = 2 * d * d_ff + d_ff * d;
    println!("    SwiGLU FFN:      T × {} exp (3 matmuls, CSE)", ffn_matmul / d);

    // RMSNorm: per token
    let rms_per_tok = d + 3; // d ln's + div + sqrt + scale
    println!("    RMSNorm:         T × ~{rms_per_tok} transcendentals (CSE on ln(x))");

    // LM head
    let _lm_head = d * v;
    println!("    LM head:         T × {} exp (CSE)", v);

    println!();
    println!("  Total layers: {n_layers} × above, + final norm + LM head");
    println!();

    // Dequant a small tensor to show it works
    if let Some(info) = gguf.tensor_info("token_embd.weight") {
        println!("═══════════════════════════════════════════════════");
        println!("  DEQUANTIZATION SPOT CHECK");
        println!("═══════════════════════════════════════════════════");
        let t0 = Instant::now();
        match gguf.load_tensor_f64(info) {
            Ok(data) => {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let n = data.len();
                let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mean: f64 = data.iter().sum::<f64>() / n as f64;
                println!("  token_embd.weight: {} elements, dequantized in {ms:.1}ms", n);
                println!("    range: [{min:.6}, {max:.6}]");
                println!("    mean:  {mean:.6}");
                println!("    first 8: {:?}", &data[..8.min(n)]);
            }
            Err(e) => println!("  Error loading tensor: {e}"),
        }
    }
}

fn generate(gguf: &GGUFFile, prompt: &[usize], tok: &Tokenizer, display_prompt: Option<&str>) {
    println!("Loading model weights...");
    let t0 = Instant::now();
    let weights = match ModelWeights::from_gguf(gguf) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error loading weights: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Weights loaded in {load_ms:.0}ms");

    let rope = RopeCache::new(
        weights.config.d_head,
        weights.config.max_seq_len.min(2048),
        weights.config.rope_freq_base,
    );

    println!();
    println!("Running emilio (pure EML inference, KV-cached)...");
    println!("  Prompt: {} tokens", prompt.len());
    println!("  Config: {} layers, {} heads, d_model={}",
        weights.config.n_layers, weights.config.n_heads, weights.config.d_model);

    let max_new = 16;
    let t1 = Instant::now();
    let output = emilio_generate(prompt, &weights, &rope, max_new);
    let gen_s = t1.elapsed().as_secs_f64();

    let generated = &output[prompt.len()..];
    let prompt_text = display_prompt
        .map(|s| s.to_string())
        .unwrap_or_else(|| tok.decode(prompt));
    let generated_text = tok.decode(generated);

    print_result_box(&prompt_text, &generated_text, generated, gen_s);
    println!("  Every multiply was exp(ln(a) + ln(b)).");
    println!("  Every division was exp(ln(a) - ln(b)).");
}

fn generate_raw(gguf: &GGUFFile, prompt: &[usize], tok: Option<&Tokenizer>) {
    println!("Loading model weights...");
    let t0 = Instant::now();
    let weights = match ModelWeights::from_gguf(gguf) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error loading weights: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Weights loaded in {load_ms:.0}ms");

    let rope = RopeCache::new(
        weights.config.d_head,
        weights.config.max_seq_len.min(2048),
        weights.config.rope_freq_base,
    );

    println!();
    println!("Running emilio (pure EML, KV-cached)...");
    println!("  Prompt token IDs: {:?}", prompt);

    let max_new = 8;
    let t1 = Instant::now();
    let output = emilio_generate(prompt, &weights, &rope, max_new);
    let gen_s = t1.elapsed().as_secs_f64();

    let generated = &output[prompt.len()..];
    println!();
    println!("  Generated IDs: {:?}", generated);
    if let Some(tok) = tok {
        println!("  Decoded: \"{}\"", tok.decode(generated));
    }
    println!("  Time: {gen_s:.2}s ({:.4} tokens/s)",
        generated.len() as f64 / gen_s);
}

fn generate_with_weights(weights: &ModelWeights, prompt: &[usize], tok: &Tokenizer, display_prompt: Option<&str>) {
    let rope = RopeCache::new(
        weights.config.d_head,
        weights.config.max_seq_len.min(2048),
        weights.config.rope_freq_base,
    );

    println!();
    println!("Running emilio (pure EML inference, KV-cached, compiled format)...");
    println!("  Prompt: {} tokens", prompt.len());
    println!("  Config: {} layers, {} heads, d_model={}",
        weights.config.n_layers, weights.config.n_heads, weights.config.d_model);

    let max_new = 16;
    let t1 = Instant::now();
    let output = emilio_generate(prompt, weights, &rope, max_new);
    let gen_s = t1.elapsed().as_secs_f64();

    let generated = &output[prompt.len()..];
    let prompt_text = display_prompt
        .map(|s| s.to_string())
        .unwrap_or_else(|| tok.decode(prompt));
    let generated_text = tok.decode(generated);

    print_result_box(&prompt_text, &generated_text, generated, gen_s);
    println!("  Every multiply was exp(ln(a) + ln(b)).");
    println!("  Every division was exp(ln(a) - ln(b)).");
}

fn generate_raw_with_weights(weights: &ModelWeights, prompt: &[usize], tok: Option<&Tokenizer>) {
    let rope = RopeCache::new(
        weights.config.d_head,
        weights.config.max_seq_len.min(2048),
        weights.config.rope_freq_base,
    );

    println!();
    println!("Running emilio (pure EML, KV-cached, compiled format)...");
    println!("  Prompt token IDs: {:?}", prompt);

    let max_new = 8;
    let t1 = Instant::now();
    let output = emilio_generate(prompt, weights, &rope, max_new);
    let gen_s = t1.elapsed().as_secs_f64();

    let generated = &output[prompt.len()..];
    println!();
    println!("  Generated IDs: {:?}", generated);
    if let Some(tok) = tok {
        println!("  Decoded: \"{}\"", tok.decode(generated));
    }
    println!("  Time: {gen_s:.2}s ({:.4} tokens/s)",
        generated.len() as f64 / gen_s);
}

#[cfg(feature = "metal")]
fn generate_metal(gguf: &GGUFFile, prompt: &[usize], tok: &Tokenizer, display_prompt: Option<&str>) {
    println!("Loading model weights...");
    let t0 = Instant::now();
    let weights = match ModelWeights::from_gguf(gguf) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error loading weights: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Weights loaded in {load_ms:.0}ms");

    // Initialize Metal GPU
    println!();
    println!("Initializing Metal GPU...");
    let t_gpu = Instant::now();
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Metal init failed: {e}");
            std::process::exit(1);
        }
    };

    // Upload all weight matrices to GPU
    let gpu_w = GpuModelWeights::from_model_weights(&ctx, &weights);

    // Pre-allocate scratch buffers (eliminates per-call allocations)
    let max_act = weights.config.d_model.max(weights.config.d_ff);
    let max_result = weights.config.vocab_size.max(weights.config.d_ff);
    let pool = ScratchPool::new(&ctx, max_act, max_result);

    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;
    println!("  GPU ready in {gpu_ms:.0}ms");

    let rope = RopeCache::new(
        weights.config.d_head,
        weights.config.max_seq_len.min(2048),
        weights.config.rope_freq_base,
    );

    println!();
    println!("Running emilio (Metal GPU, pure EML inference)...");
    println!("  Prompt: {} tokens", prompt.len());
    println!("  Config: {} layers, {} heads, d_model={}",
        weights.config.n_layers, weights.config.n_heads, weights.config.d_model);
    println!("  Backend: Metal GPU (float32 exp kernel)");

    let max_new = 16;
    let t1 = Instant::now();
    let output = emilio::engine::gpu::generate_gpu(
        prompt, &weights, &gpu_w, &ctx, &pool, &rope, max_new,
    );
    let gen_s = t1.elapsed().as_secs_f64();

    let generated = &output[prompt.len()..];
    let prompt_text = display_prompt
        .map(|s| s.to_string())
        .unwrap_or_else(|| tok.decode(prompt));
    let generated_text = tok.decode(generated);

    print_result_box(&prompt_text, &generated_text, generated, gen_s);
    println!("  Metal GPU: exp() kernel on GPU, ln()/norm/RoPE on CPU.");
    println!("  Every multiply was exp(ln(a) + ln(b)) -- on the GPU.");
}
