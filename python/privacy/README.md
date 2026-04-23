# Privacy Filter on Apple's Neural Engine

**15× faster. 19× less energy. Same accuracy.**

OpenAI's [privacy-filter](https://github.com/openai/openai-privacy-filter) is a
Mixture-of-Experts NER model (8 layers, 128 experts, top-4 routing, ~1.5B total
parameters). The canonical implementation runs on CPU via PyTorch at **1.65 sentences/sec**,
drawing **15.4 J per sentence** of package power.

This port runs the same model entirely on Apple's Neural Engine — no GPU, no cloud —
at **24.6 sentences/sec** and **0.8 J per sentence**. Every entity span matches the
reference output exactly (span F1 = 100%).

| | CPU (PyTorch) | ANE (this repo) | |
|---|---|---|---|
| Throughput | 1.65 sent/s | **24.6 sent/s** | **15× faster** |
| Energy per sentence | 15,385 mJ | **812 mJ** | **19× less** |
| ANE power draw | 0 W | 0.22 W | — |
| Span F1 | 100% | 100% | — |

> Measured on M4 Max (48 GB), macOS 15, 30-second sustained run via `powermetrics`.

The trick: instead of one monolithic graph (which hits the ANE's 96 MB cliff and
falls back to GPU), each of the 1,024 experts is compiled as its own CoreML model.
At runtime, Swift dispatches the router's top-4 picks concurrently. Result: 1,033
compiled `.mlmodelc` files, 2.75 ms per MoE layer, full ANE residency.

---

## Build & Run Guide

**Hardware**: Apple Silicon Mac with ANE (tested on M4 Max, 48 GB).
**macOS**: 15+ (CoreML 9 / Xcode 16+).

---

## 0. Clone & Prerequisites

```bash
git clone <repo-url> && cd em2

# Xcode command-line tools (provides coremltools 9 via Xcode python3)
xcode-select --install
```

Verify the Xcode python3 has coremltools:
```bash
/Applications/Xcode.app/Contents/Developer/usr/bin/python3 -c "import coremltools; print(coremltools.__version__)"
# Should print 9.x
```

## 1. Python Environment

Two separate envs are needed:

| Env | Python | Purpose |
|-----|--------|---------|
| `.venv313` | 3.13 | Golden capture, weight extraction, inference |
| Xcode `python3` | System (Py 3.9) | CoreML conversion only |

```bash
# Create .venv313
python3.13 -m venv .venv313
.venv313/bin/pip install torch numpy safetensors huggingface_hub tiktoken

# Install the vendored opf package (editable)
cd python/privacy/_vendor_src/opf_src
../../../../.venv313/bin/pip install -e .
cd ../../../..
```

> **Note**: The vendored `opf_src/` is not included in the repo. Obtain it from
> [openai/openai-privacy-filter](https://github.com/openai/openai-privacy-filter)
> and place it at `python/privacy/_vendor_src/opf_src/`.

## 2. Download Model Weights

The reference script auto-downloads weights from HuggingFace on first run
(~2.8 GB → `python/privacy/_vendor_src/weights/model.safetensors`).

You also need the config files in `python/privacy/_vendor_src/`:
- `config.json`, `tokenizer.json`, `tokenizer_config.json`

These come from the HuggingFace repo or the vendored opf package.

## 3. Capture Golden Reference

```bash
PYTHONPATH=python/privacy/_vendor_src/opf_src \
  .venv313/bin/python -m python.privacy.pf_ref
```

**Produces**: `python/privacy/out/pf_golden.npz` — reference logits, labels,
tokenized inputs, and `id2label` mapping for 8 test sentences.

## 4. Capture MoE Routing Goldens

```bash
PYTHONPATH=python/privacy/_vendor_src/opf_src \
  .venv313/bin/python python/privacy/pf_moe_goldens.py
```

**Produces**: `python/privacy/out/pf_layer0_moe.npz` — per-layer router logits,
top-K indices and weights.

## 5. Extract Weights for Swift

```bash
PYTHONPATH=python/privacy/_vendor_src/opf_src \
  .venv313/bin/python python/privacy/extract_pf_swift_weights.py
```

**Produces**: `emilio/conv-ane/PF_swift/` — binary weight files (fp16 embeddings,
fp32 norms/gates), test fixtures, and `manifest.json`.

## 6. CoreML Conversion

All three steps use **Xcode python3** (coremltools 9).

```bash
XPYTHON=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
```

### 6a. Fused Attention + Router Packs (8 layers)

```bash
$XPYTHON python/privacy/build_pf_fused_attn_router_ane.py --all-layers --force
```

Produces one `.mlpackage` per layer: `PF_fused_L{0..7}_T128.mlpackage`.

### 6b. Per-Expert Packs (128 experts × 8 layers = 1,024 packs)

```bash
for L in 0 1 2 3 4 5 6 7; do
  $XPYTHON python/privacy/build_pf_per_expert_all.py --layer $L
done
```

Produces compiled `.mlmodelc` per expert (auto-compiles inline).
~6 min per layer, ~50 min total.

### 6c. Tail Pack (final norm + classifier head)

```bash
$XPYTHON python/privacy/build_pf_tail_ane.py --force
```

Produces `PF_tail_T128.mlpackage`.

### 6d. Compile .mlpackage → .mlmodelc

The expert packs (6b) compile inline. For the fused and tail packs:

```bash
cd emilio/conv-ane
for f in PF_fused_L*_T128.mlpackage PF_tail_T128.mlpackage; do
  xcrun coremlcompiler compile "$(pwd)/$f" "$(pwd)"
done
cd ../..
```

> **Important**: `xcrun coremlcompiler` requires **absolute paths**.

## 7. Build the Swift Driver

```bash
cd emilio/conv-ane
swiftc -O -framework CoreML -framework Accelerate -o pf_e2e pf_e2e.swift
cd ../..
```

## 8. Run Inference

```bash
# Single sentence
.venv313/bin/python python/privacy/pf_infer.py "My email is alice@example.com"

# From a file (one sentence per line)
.venv313/bin/python python/privacy/pf_infer.py -f input.txt

# From stdin
cat document.txt | .venv313/bin/python python/privacy/pf_infer.py -

# Verbose mode (shows entity details)
.venv313/bin/python python/privacy/pf_infer.py -v "Call Bob at 555-0142."
```

Output replaces detected PII with `[TYPE]` markers:

```
My email is [PRIVATE_EMAIL]
Call [PRIVATE_PERSON] at [PRIVATE_PHONE].
```

## 9. Run the Demo

```bash
# Slow (1.5 s between lines, default)
bash demo/demo_redact.sh

# Fast
DEMO_DELAY=0.1 bash demo/demo_redact.sh
```

---

## Quick Reference

| Step | Env | Time (M4 Max) | Output |
|------|-----|--------------|--------|
| Golden capture | `.venv313` | ~30 s | `out/pf_golden.npz` |
| MoE goldens | `.venv313` | ~30 s | `out/pf_layer0_moe.npz` |
| Weight extract | `.venv313` | ~10 s | `PF_swift/` (~250 MB) |
| Fused packs (8) | Xcode py3 | ~5 min | 8 `.mlpackage` |
| Expert packs (1024) | Xcode py3 | ~50 min | 1024 `.mlmodelc` |
| Tail pack | Xcode py3 | ~30 s | 1 `.mlpackage` |
| Compile packs | xcrun | ~2 min | `.mlmodelc` |
| Swift build | swiftc | ~5 s | `pf_e2e` binary |
| **Total** | | **~60 min** | |

## Performance

Measured on M4 Max (48 GB, 16 ANE cores), macOS 15, 30-second sustained run
via `scripts/pf_energy_probe.sh` with `sudo powermetrics`.

### Throughput & Accuracy

|  | CPU (PyTorch) | ANE (Swift, fused packs) | Δ |
|--|--------------|--------------------------|---|
| Throughput | 1.65 sent/s | **24.6 sent/s** | **15×** |
| Latency/sent | 606 ms | 40.7 ms | 15× faster |
| Span F1 | 100% | 100% | — |

### Energy

|  | CPU (PyTorch) | ANE (Swift, fused packs) | Δ |
|--|--------------|--------------------------|---|
| PKG energy/sent | 15,385 mJ | **812 mJ** | **19× less** |
| ANE power | 0 W | 0.22 W | — |
| GPU power | idle | idle | — |

The per-expert dispatch strategy (1,024 small graphs + 8 fused attention/router
packs + 1 tail pack = 1,033 CoreML models) keeps every expert prediction on ANE.
The alternative — a single gather-based dense graph — falls off ANE to GPU once
the expert dimension exceeds the 96 MB single-graph cliff.

### Model Architecture (for context)

| Parameter | Value |
|-----------|-------|
| Layers | 8 |
| Experts/layer | 128 (top-K = 4) |
| Hidden dim | 640 |
| Expert FFN dim | 64 |
| Active params/token | ~50 M |
| Total params | ~1.5 B |
| NER labels | 33 (BIOES scheme) |
