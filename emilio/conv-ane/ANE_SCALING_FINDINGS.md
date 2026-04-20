# Scaling Conv-Only Transformers on Apple Neural Engine

**Date:** April 2026  
**Hardware:** Apple M4 Max, macOS  
**Stack:** coremltools 9.0, CoreML mlprogram (iOS18+), CPU_AND_NE compute units

## Summary

We attempted to push our conv-only Qwen2.5 inference pipeline from 0.5B to larger
model sizes (1.5B, 3B) on the Apple Neural Engine. The 0.5B int8 model remains the
only configuration that delivers both quality and speed. Larger models hit hard ANE
limits on weight size and numerical precision.

## Results Matrix

| Model | Quant | Weights | Compiles | Runs | Speed | Quality |
|-------|-------|---------|----------|------|-------|---------|
| **0.5B** (24L, d=896) | int8 | 472 MB | Yes | Yes | **160 tok/s** | **Good** |
| 1.5B (28L, d=1536) | int4 g32 | 828 MB | Yes | Yes | ~24 tok/s | Degrades after ~15 tokens |
| 1.5B (28L, d=1536) | int4 g64 | 828 MB | Yes | Yes | ~24 tok/s | Same as g32 — no improvement |
| 1.5B (28L, d=1536) | int8 | ~1.5 GB | **Error -14** | No | — | — |
| 1.5B (28L, d=1536) | fp16 | 2.9 GB | Yes* | NaN on fwd 2 | — | Broken |
| 3B (36L, d=2048) | int4 g32 | 1.7 GB | **Error -14** | No | — | — |

*fp16 1.5B: `xcrun coremlcompiler compile` succeeds, but at runtime produces all-zero
logits on the first forward, valid logits on the second, then **all NaN** from the
third forward onward.

## Key Findings

### 1. ANE has a hard weight-size ceiling around ~1 GB

The ANE compiler (ANEF) fails with error -14 when the compiled model weights exceed
roughly 1 GB. This is a compiler-level rejection, not a runtime memory limit.

- 828 MB (1.5B int4): compiles and runs
- ~1.5 GB (1.5B int8): error -14 at compile time
- ~1.7 GB (3B int4): error -14 at compile time

Note: `coremltools` reports error -14 during its internal validation pass (inside
`.save()`), and `xcrun coremlcompiler compile` may still produce an mlmodelc — but
the model will crash with error -14 at runtime if ANE can't execute it.

### 2. fp16 causes numerical overflow at d=1536

The 1.5B model has head dimension dh=128 (vs dh=64 for 0.5B). In fp16, the attention
dot products and softmax accumulate enough error over 28 layers to produce NaN by
the third forward pass. This is a fundamental fp16 dynamic range issue — the
activations overflow, not the weights.

The 0.5B model (dh=64, 24L) stays within fp16 range.

### 3. int4 quantization degrades 1.5B quality catastrophically

While int4 fits the ANE weight budget, the quantization error at 1.5B scale destroys
coherence past ~15 generated tokens:

- **Works:** "Capital of France?" → "Paris" (single-token factual recall)
- **Fails:** "3 largest planets" → "Jupiter (mass: 1.111111111..."
- **Fails:** "Berlin Wall" → "November . . . . . . ."
- **Fails:** Any multi-sentence generation → repetition loops

Increasing group size from 32 to 64 did not help. The issue is that 1.5B's wider
layers (d=1536, dff=8960) are more sensitive to 4-bit quantization noise than
0.5B's narrower layers (d=896, dff=4864).

### 4. The 0.5B sweet spot

The 0.5B int8 model hits a favorable point in the ANE design space:

- **Weight size** (472 MB): well under the ~1 GB ceiling
- **Head dim** (64): safe for fp16 activations
- **int8 quantization**: negligible quality loss at this model scale
- **160 tok/s decode**: ~6 ms per token, fully ANE-accelerated
- **Quality**: coherent multi-paragraph generation, correct code, accurate math

## Architecture Notes

All models use the same conv-only architecture:
- Every matmul replaced with Conv2d(1×1) — ANE's conv engine does all compute
- Fused QKV and Gate+Up projections
- RMSNorm with float32 upcast for stability
- Stateful KV cache (MLState, zero host copy)
- BPE tokenizer from GGUF vocabulary

## What's Left in the Repo

```
emilio/conv-ane/
├── QwenANE_24L_stateful.*          # 0.5B int8 — the demo model ✓
├── QwenANE_28L_stateful_q4.*       # 1.5B int4 g32 — works, poor quality
├── QwenANE_28L_stateful_q4g64.*    # 1.5B int4 g64 — works, no better
├── QwenANE_28L_stateful.*          # symlinks → q4 variants
├── gguf_to_ane.py                  # GGUF → CoreML converter
├── qwen_ane.swift                  # Swift inference host
├── build_model.sh                  # Full build pipeline script
└── build.sh                        # Legacy build script

models/
├── qwen2.5-0.5b-instruct-q8_0.gguf    # 644 MB — keep
├── qwen2.5-1.5b-instruct-q8_0.gguf    # 1.8 GB — keep for future experiments
└── qwen2.5-3b-instruct-q8_0.gguf      # 3.4 GB — can delete (dead end on ANE)
```

## Possible Future Directions

- **Mixed precision**: fp32 accumulation for attention, fp16 for FFN — if coremltools
  exposes per-op dtype control
- **Model splitting**: split 1.5B across multiple ANE segments to fit under weight ceiling
- **int4 with calibration**: use activation-aware quantization (GPTQ/AWQ-style) instead
  of naive post-training quantization — may recover quality
- **Speculative decoding**: use 0.5B as draft model, verify with CPU-side 1.5B fp32
