//! Optimized Metal GPU backend for EML matmul.
//!
//! Key optimizations:
//!   1. half-precision weights — halves bandwidth for the dominant memory read.
//!   2. Precomputed signs — ±1.0 extracted at upload time (no rint/branch in kernel).
//!   3. Pre-allocated scratch buffers — zero per-call buffer allocations.
//!   4. Shared ln(A) — QKV and gate/up share the same input activation.
//!   5. Batched command buffers — QKV in one, fused FFN in one.
//!   6. Fused FFN — gate+up matmuls → silu_mul_ln → down matmul, all on GPU
//!      in a single command buffer (eliminates CPU silu+mul round-trip).

use metal::*;
use num_complex::Complex64;
use std::mem;

// ─── f32 → f16 conversion ──────────────────────────────────────────────────

fn f32_to_f16(val: f32) -> u16 {
    let b = val.to_bits();
    let sign = (b >> 16) & 0x8000;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x007F_FFFF;

    if exp == 0 {
        sign as u16
    } else if exp == 0xFF {
        (sign | 0x7C00 | if man != 0 { 0x200 } else { 0 }) as u16
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            (sign | 0x7C00) as u16
        } else if new_exp <= 0 {
            if new_exp >= -10 {
                let shifted = (man | 0x0080_0000) >> (14 - new_exp);
                (sign | shifted) as u16
            } else {
                sign as u16
            }
        } else {
            (sign | ((new_exp as u32) << 10) | (man >> 13)) as u16
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn write_u32(buf: &Buffer, val: u32) {
    unsafe { *(buf.contents() as *mut u32) = val; }
}

fn read_buffer_f64(buf: &Buffer, n: usize) -> Vec<f64> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }
        .iter()
        .map(|&v| v as f64)
        .collect()
}

fn upload_f32(buf: &Buffer, data: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents() as *mut f32,
            data.len(),
        );
    }
}

// ─── Metal context ──────────────────────────────────────────────────────────

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipe_matmul_v4: ComputePipelineState,
    pipe_silu_mul_ln: ComputePipelineState,
    pipe_ln_split: ComputePipelineState,
    pipe_residual_rms_norm_ln: ComputePipelineState,
}

/// GPU-resident weight buffers: half-precision mag + packed sign bits.
pub struct GpuWeights {
    pub buf_mag: Buffer,    // half[cols * inner]: ln(|weight|)
    pub buf_sign: Buffer,   // u32[ceil(cols*inner/32)]: packed sign bits (0=+, 1=-)
    pub inner: usize,
    pub cols: usize,
}

#[allow(clippy::too_many_arguments)]
impl MetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found".to_string())?;

        eprintln!("  Metal GPU: {}", device.name());

        let queue = device.new_command_queue();

        let shader_src = include_str!("eml_matmul.metal");
        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(shader_src, &opts)
            .map_err(|e| format!("Metal shader compilation failed: {e}"))?;

        let make_pipe = |name: &str| -> Result<ComputePipelineState, String> {
            let f = library.get_function(name, None)
                .map_err(|e| format!("Function '{name}' not found: {e}"))?;
            device.new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("Pipeline '{name}' failed: {e}"))
        };

        let pipe_matmul_v4 = make_pipe("eml_matmul_v4")?;
        let pipe_silu_mul_ln = make_pipe("eml_silu_mul_ln")?;
        let pipe_ln_split = make_pipe("eml_ln_split")?;
        let pipe_residual_rms_norm_ln = make_pipe("eml_residual_rms_norm_ln_split")?;

        Ok(MetalContext { device, queue, pipe_matmul_v4, pipe_silu_mul_ln,
            pipe_ln_split, pipe_residual_rms_norm_ln })
    }

    // ── Buffer helpers ──────────────────────────────────────────────

    fn new_buffer_with_data<T: Copy>(&self, data: &[T]) -> Buffer {
        let byte_len = std::mem::size_of_val(data);
        self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn alloc_f32(&self, n: usize) -> Buffer {
        self.device.new_buffer(
            (n * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    // ── Weight upload (half-precision) ──────────────────────────────

    pub fn upload_weights(
        &self,
        ln_b_t: &[Complex64],
        inner: usize,
        cols: usize,
    ) -> GpuWeights {
        let n = cols * inner;
        assert_eq!(ln_b_t.len(), n, "ln_b_t size mismatch");

        let mut mag_h: Vec<u16> = Vec::with_capacity(n);
        let n_words = n.div_ceil(32);
        let mut sign_packed: Vec<u32> = vec![0u32; n_words];
        for (i, c) in ln_b_t.iter().enumerate() {
            mag_h.push(f32_to_f16(c.re as f32));
            if c.im.abs() >= 1.0 {
                // negative weight: set bit
                sign_packed[i >> 5] |= 1 << (i & 31);
            }
        }

        GpuWeights {
            buf_mag: self.new_buffer_with_data(&mag_h),
            buf_sign: self.new_buffer_with_data(&sign_packed),
            inner,
            cols,
        }
    }

    // ── Encode a single matmul dispatch (v2: half weights) ──────────

    fn encode_matmul(
        &self,
        enc: &ComputeCommandEncoderRef,
        a_mag: &Buffer,
        a_sign: &Buffer,
        w: &GpuWeights,
        result: &Buffer,
        params: &ParamSet,
        rows: usize,
        cols: usize,
    ) {
        enc.set_compute_pipeline_state(&self.pipe_matmul_v4);
        enc.set_buffer(0, Some(a_mag), 0);
        enc.set_buffer(1, Some(a_sign), 0);
        enc.set_buffer(2, Some(&w.buf_mag), 0);
        enc.set_buffer(3, Some(&w.buf_sign), 0);
        enc.set_buffer(4, Some(result), 0);
        enc.set_buffer(5, Some(&params.rows), 0);
        enc.set_buffer(6, Some(&params.inner), 0);
        enc.set_buffer(7, Some(&params.cols), 0);

        // V3: one SIMD group (32 threads) per output element
        let threadgroups = MTLSize::new(cols as u64, rows as u64, 1);
        let tg_size = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(threadgroups, tg_size);
    }

    // ── Encode ln_split dispatch ─────────────────────────────────────

    fn encode_ln_split(
        &self,
        enc: &ComputeCommandEncoderRef,
        input: &Buffer,
        out_mag: &Buffer,
        out_sign: &Buffer,
        count_buf: &Buffer,
        count: usize,
    ) {
        enc.set_compute_pipeline_state(&self.pipe_ln_split);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(out_mag), 0);
        enc.set_buffer(2, Some(out_sign), 0);
        enc.set_buffer(3, Some(count_buf), 0);

        let grid = MTLSize::new(count as u64, 1, 1);
        let tg = MTLSize::new(count.min(256) as u64, 1, 1);
        enc.dispatch_threads(grid, tg);
    }

    // ── Encode silu_mul_ln dispatch ─────────────────────────────────

    fn encode_silu_mul_ln(
        &self,
        enc: &ComputeCommandEncoderRef,
        gate: &Buffer,
        up: &Buffer,
        out_mag: &Buffer,
        out_sign: &Buffer,
        count_buf: &Buffer,
        count: usize,
    ) {
        enc.set_compute_pipeline_state(&self.pipe_silu_mul_ln);
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up), 0);
        enc.set_buffer(2, Some(out_mag), 0);
        enc.set_buffer(3, Some(out_sign), 0);
        enc.set_buffer(4, Some(count_buf), 0);

        let grid = MTLSize::new(count as u64, 1, 1);
        let tg = MTLSize::new(count.min(256) as u64, 1, 1);
        enc.dispatch_threads(grid, tg);
    }

    // ── Encode fused residual_add + rms_norm + ln_split ─────────────

    fn encode_residual_rms_norm_ln_split(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,         // first residual input
        b: &Buffer,         // second residual input
        gamma: &Buffer,     // norm scale
        out_mag: &Buffer,
        out_sign: &Buffer,
        z_out: &Buffer,     // a+b written here (for later use)
        n_buf: &Buffer,
        eps_buf: &Buffer,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(&self.pipe_residual_rms_norm_ln);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(gamma), 0);
        enc.set_buffer(3, Some(out_mag), 0);
        enc.set_buffer(4, Some(out_sign), 0);
        enc.set_buffer(5, Some(z_out), 0);
        enc.set_buffer(6, Some(n_buf), 0);
        enc.set_buffer(7, Some(eps_buf), 0);

        let nth = n.min(256) as u64;
        let tg = MTLSize::new(nth, 1, 1);
        enc.set_threadgroup_memory_length(0, 32 * mem::size_of::<f32>() as u64);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), tg);
    }

    /// Upload raw f64 activation to GPU as f32, used before GPU-side ln_split.
    fn upload_act_raw(&self, pool: &ScratchPool, x: &[f64]) {
        let f32s: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        upload_f32(&pool.act_raw, &f32s);
    }
    // ── Batched operations ──────────────────────────────────────────

    /// Batch QKV projections: upload raw activation, GPU ln_split, 3 matmul dispatches, 1 commit.
    pub fn batch_qkv(
        &self,
        pool: &ScratchPool,
        x: &[f64],
        wq: &GpuWeights, wk: &GpuWeights, wv: &GpuWeights,
        d: usize, q_dim: usize, kv_dim: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.upload_act_raw(pool, x);
        write_u32(&pool.count_buf, d as u32);

        pool.params[0].write(1, d, q_dim);
        pool.params[1].write(1, d, kv_dim);

        let cmd = self.queue.new_command_buffer();

        // Phase 0: GPU ln_split — compute mag+sign from raw activation
        {
            let enc = cmd.new_compute_command_encoder();
            self.encode_ln_split(enc, &pool.act_raw,
                &pool.act_mag, &pool.act_sign, &pool.count_buf, d);
            enc.end_encoding();
        }

        // Phase 1: 3 matmul dispatches (all read from act_mag/act_sign)
        {
            let enc = cmd.new_compute_command_encoder();
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                wq, &pool.result[0], &pool.params[0], 1, q_dim);
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                wk, &pool.result[1], &pool.params[1], 1, kv_dim);
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                wv, &pool.result[2], &pool.params[1], 1, kv_dim);
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        (
            read_buffer_f64(&pool.result[0], q_dim),
            read_buffer_f64(&pool.result[1], kv_dim),
            read_buffer_f64(&pool.result[2], kv_dim),
        )
    }

    /// Fused FFN: GPU ln_split → gate+up matmuls → silu_mul_ln → down matmul.
    /// All on GPU in a single command buffer (4 encoders for data deps).
    pub fn batch_ffn_fused(
        &self,
        pool: &ScratchPool,
        x: &[f64],
        wgate: &GpuWeights, wup: &GpuWeights, wdown: &GpuWeights,
        d: usize, d_ff: usize,
    ) -> Vec<f64> {
        self.upload_act_raw(pool, x);
        write_u32(&pool.count_buf, d as u32);

        pool.params[0].write(1, d, d_ff);

        let cmd = self.queue.new_command_buffer();

        // Phase 0: GPU ln_split on raw activation
        {
            let enc = cmd.new_compute_command_encoder();
            self.encode_ln_split(enc, &pool.act_raw,
                &pool.act_mag, &pool.act_sign, &pool.count_buf, d);
            enc.end_encoding();
        }

        // Phase 1: gate + up matmuls (independent, can overlap)
        {
            let enc = cmd.new_compute_command_encoder();
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                wgate, &pool.result[0], &pool.params[0], 1, d_ff);
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                wup, &pool.result[1], &pool.params[0], 1, d_ff);
            enc.end_encoding();
        }

        // Phase 2: fused silu(gate) × up → mag+sign
        {
            write_u32(&pool.count_buf, d_ff as u32);
            let enc = cmd.new_compute_command_encoder();
            self.encode_silu_mul_ln(
                enc, &pool.result[0], &pool.result[1],
                &pool.act_mag, &pool.act_sign, &pool.count_buf, d_ff,
            );
            enc.end_encoding();
        }

        // Phase 3: down matmul
        {
            pool.params[1].write(1, d_ff, d);
            let enc = cmd.new_compute_command_encoder();
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                wdown, &pool.result[2], &pool.params[1], 1, d);
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        read_buffer_f64(&pool.result[2], d)
    }

    /// Fused O_proj → residual_add → RMSNorm → ln_split → FFN → residual_add.
    /// All on GPU in a single command buffer.  Saves 2 commit+waits per layer
    /// (the old separate O and FFN commits).
    ///
    /// Input: `attn_weighted` — attention-weighted output (before O projection),
    ///        `x_residual` — layer input (for both residual connections).
    /// Output: layer output = (x + O(attn_weighted)) + ffn(norm(x + O(attn_weighted))) as Vec<f64>.
    pub fn batch_o_to_ffn(
        &self,
        pool: &ScratchPool,
        attn_weighted: &[f64],  // pre-O-projection attention output
        x_residual: &[f64],     // layer input (for residual adds)
        ffn_norm_gpu: &Buffer,  // RMSNorm gamma weights on GPU
        eps_buf: &Buffer,       // epsilon value
        gpu_layer: &GpuLayerWeights,
        q_dim: usize, d: usize, d_ff: usize,
    ) -> Vec<f64> {
        // Upload both vectors before creating command buffer
        let attn_f32: Vec<f32> = attn_weighted.iter().map(|&v| v as f32).collect();
        let x_f32: Vec<f32> = x_residual.iter().map(|&v| v as f32).collect();
        upload_f32(&pool.act_raw, &attn_f32);   // attn_weighted → act_raw
        upload_f32(&pool.result[3], &x_f32);     // x_residual → result[3]

        // Pre-write count values into separate buffers (GPU sees final state at commit time)
        write_u32(&pool.count_buf, d as u32);       // used by ln_split, add, rmsnorm
        write_u32(&pool.count_buf2, d_ff as u32);   // used by silu_mul_ln

        let cmd = self.queue.new_command_buffer();

        // Phase 0: ln_split(attn_weighted) → act_mag/act_sign
        {
            let enc = cmd.new_compute_command_encoder();
            self.encode_ln_split(enc, &pool.act_raw,
                &pool.act_mag, &pool.act_sign, &pool.count_buf, q_dim);
            enc.end_encoding();
        }

        // Phase 1: O matmul → result[0]  (q_dim → d)
        {
            pool.params[0].write(1, q_dim, d);
            let enc = cmd.new_compute_command_encoder();
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                &gpu_layer.o, &pool.result[0], &pool.params[0], 1, d);
            enc.end_encoding();
        }

        // Phase 2: fused residual_add + RMSNorm + ln_split
        //   result[4] = x_residual + O_output  (z_out, kept for final residual)
        //   act_mag/act_sign = ln_split(RMSNorm(result[4]))
        {
            let enc = cmd.new_compute_command_encoder();
            self.encode_residual_rms_norm_ln_split(
                enc, &pool.result[3], &pool.result[0],
                ffn_norm_gpu, &pool.act_mag, &pool.act_sign,
                &pool.result[4], &pool.count_buf, eps_buf, d,
            );
            enc.end_encoding();
        }

        // Phase 3: gate + up matmuls
        {
            pool.params[1].write(1, d, d_ff);
            let enc = cmd.new_compute_command_encoder();
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                &gpu_layer.gate, &pool.result[0], &pool.params[1], 1, d_ff);
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                &gpu_layer.up, &pool.result[1], &pool.params[1], 1, d_ff);
            enc.end_encoding();
        }

        // Phase 4: silu_mul_ln
        {
            let enc = cmd.new_compute_command_encoder();
            self.encode_silu_mul_ln(
                enc, &pool.result[0], &pool.result[1],
                &pool.act_mag, &pool.act_sign, &pool.count_buf2, d_ff,
            );
            enc.end_encoding();
        }

        // Phase 5: down matmul → result[2]
        {
            pool.params[2].write(1, d_ff, d);
            let enc = cmd.new_compute_command_encoder();
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                &gpu_layer.down, &pool.result[2], &pool.params[2], 1, d);
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        // Final residual add on CPU (avoids extra encoder for tiny 896-element add)
        let ffn_out = read_buffer_f64(&pool.result[2], d);
        let residual = read_buffer_f64(&pool.result[4], d);
        ffn_out.iter().zip(residual.iter()).map(|(&f, &r)| f + r).collect()
    }

    /// Single matmul for projections that can't batch (O, LM head).
    pub fn single_matmul(
        &self,
        pool: &ScratchPool,
        x: &[f64],
        w: &GpuWeights,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Vec<f64> {
        self.upload_act_raw(pool, x);
        write_u32(&pool.count_buf, (rows * inner) as u32);

        pool.params[0].write(rows, inner, cols);

        let cmd = self.queue.new_command_buffer();

        // Phase 0: GPU ln_split
        {
            let enc = cmd.new_compute_command_encoder();
            self.encode_ln_split(enc, &pool.act_raw,
                &pool.act_mag, &pool.act_sign, &pool.count_buf, rows * inner);
            enc.end_encoding();
        }

        // Phase 1: matmul
        {
            let enc = cmd.new_compute_command_encoder();
            self.encode_matmul(enc, &pool.act_mag, &pool.act_sign,
                w, &pool.result[0], &pool.params[0], rows, cols);
            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();

        read_buffer_f64(&pool.result[0], rows * cols)
    }
}

// ─── Param set ──────────────────────────────────────────────────────────────

pub struct ParamSet {
    rows: Buffer,
    inner: Buffer,
    cols: Buffer,
}

impl ParamSet {
    fn new(device: &Device) -> Self {
        let alloc = || device.new_buffer(
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        ParamSet { rows: alloc(), inner: alloc(), cols: alloc() }
    }

    fn write(&self, rows: usize, inner: usize, cols: usize) {
        write_u32(&self.rows, rows as u32);
        write_u32(&self.inner, inner as u32);
        write_u32(&self.cols, cols as u32);
    }
}

// ─── Scratch pool ───────────────────────────────────────────────────────────

pub struct ScratchPool {
    act_raw: Buffer,     // f32[max_act]: raw activation values (before ln_split)
    act_mag: Buffer,     // f32[max_act]: ln(|activation|) (output of ln_split)
    act_sign: Buffer,    // f32[max_act]: sign(activation) (output of ln_split)
    pub result: [Buffer; 5],
    params: [ParamSet; 3],
    count_buf: Buffer,
    count_buf2: Buffer,  // second count buffer for fused paths needing two counts
}

impl ScratchPool {
    pub fn new(ctx: &MetalContext, max_act: usize, max_result: usize) -> Self {
        ScratchPool {
            act_raw: ctx.alloc_f32(max_act),
            act_mag: ctx.alloc_f32(max_act),
            act_sign: ctx.alloc_f32(max_act),
            result: [
                ctx.alloc_f32(max_result),
                ctx.alloc_f32(max_result),
                ctx.alloc_f32(max_result),
                ctx.alloc_f32(max_result),
                ctx.alloc_f32(max_result),
            ],
            params: [
                ParamSet::new(&ctx.device),
                ParamSet::new(&ctx.device),
                ParamSet::new(&ctx.device),
            ],
            count_buf: ctx.device.new_buffer(
                mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            count_buf2: ctx.device.new_buffer(
                mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            ),
        }
    }
}

// ─── GPU model weight storage ───────────────────────────────────────────────

pub struct GpuLayerWeights {
    pub q: GpuWeights,
    pub k: GpuWeights,
    pub v: GpuWeights,
    pub o: GpuWeights,
    pub gate: GpuWeights,
    pub up: GpuWeights,
    pub down: GpuWeights,
    pub ffn_norm: Buffer,   // f32 RMSNorm gamma weights for FFN
}

pub struct GpuModelWeights {
    pub layers: Vec<GpuLayerWeights>,
    pub output: GpuWeights,
    pub eps_buf: Buffer,  // f32 epsilon for RMSNorm
}

impl GpuModelWeights {
    pub fn from_model_weights(
        ctx: &MetalContext,
        weights: &crate::engine::ModelWeights,
    ) -> Self {
        let cfg = &weights.config;
        let d = cfg.d_model;
        let d_head = cfg.d_head;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * d_head;
        let kv_dim = n_kv_heads * d_head;
        let d_ff = cfg.d_ff;

        eprintln!("  Uploading {} layers to GPU (half-precision)...", cfg.n_layers);

        let layers: Vec<GpuLayerWeights> = weights.layers.iter().map(|layer| {
            // Upload ffn_norm gamma as f32
            let norm_f32: Vec<f32> = layer.ffn_norm.iter().map(|&v| v as f32).collect();
            let norm_buf = ctx.alloc_f32(d);
            upload_f32(&norm_buf, &norm_f32);

            GpuLayerWeights {
                q:    ctx.upload_weights(&layer.ln_q, d, q_dim),
                k:    ctx.upload_weights(&layer.ln_k, d, kv_dim),
                v:    ctx.upload_weights(&layer.ln_v, d, kv_dim),
                o:    ctx.upload_weights(&layer.ln_o, q_dim, d),
                gate: ctx.upload_weights(&layer.ln_gate, d, d_ff),
                up:   ctx.upload_weights(&layer.ln_up, d, d_ff),
                down: ctx.upload_weights(&layer.ln_down, d_ff, d),
                ffn_norm: norm_buf,
            }
        }).collect();

        let output = ctx.upload_weights(&weights.ln_output, d, cfg.vocab_size);

        // Epsilon buffer
        let eps_buf = ctx.device.new_buffer(
            mem::size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        {
            let eps_f32 = cfg.rms_norm_eps as f32;
            unsafe {
                *(eps_buf.contents() as *mut f32) = eps_f32;
            }
        }

        eprintln!("  Uploaded {} weight matrices (half-precision, {:.0}MB)",
            cfg.n_layers * 7 + 1,
            Self::total_bytes(&layers, &output) as f64 / 1_048_576.0);

        GpuModelWeights { layers, output, eps_buf }
    }

    fn total_bytes(layers: &[GpuLayerWeights], output: &GpuWeights) -> usize {
        let w_bytes = |w: &GpuWeights| {
            w.inner * w.cols * 2  // half mag
            + (w.inner * w.cols).div_ceil(32) * 4  // packed sign bits
        };
        let layer_bytes: usize = layers.iter()
            .map(|l| w_bytes(&l.q) + w_bytes(&l.k) + w_bytes(&l.v) + w_bytes(&l.o)
                + w_bytes(&l.gate) + w_bytes(&l.up) + w_bytes(&l.down))
            .sum();
        layer_bytes + w_bytes(output)
    }
}
