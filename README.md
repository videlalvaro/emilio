# emilio — EML Inference Engine

A complete LLM inference engine where **every multiplication is `exp(ln(a) + ln(b))`**.

Based on the EML primitive from [Odrzywołek (2026)](https://arxiv.org/abs/2603.21852):

```
eml(x, y) = exp(x) − ln(y)
```

This single operation — one exponentiation, one logarithm, one subtraction — can express
all of arithmetic, linear algebra, and the full transformer forward pass. **emilio** proves
this by running Qwen2.5-0.5B-Instruct (494M parameters) entirely through EML, producing
correct English at ~5.5 tok/s (CPU) and ~30 tok/s (GPU) on Apple Silicon.

Speed is not the goal — anyone wanting fast inference should use
[llama.cpp](https://github.com/ggerganov/llama.cpp). The goal is demonstrating that
a single algebraic identity suffices for every operation in a production transformer.

## What's in the repo

| Path | Description |
|---|---|
| `emilio/` | **emilio** — Rust inference engine (CPU + Metal GPU) |
| `emilio/src/eml_matmul.metal` | Metal shaders (4 pure-EML kernels) |
| `python/eml_core.py` | Python proof-of-concept with instrumented EML call counts |
| `python/eml_model.py` | Python model loader (GGUF → EML forward pass) |
| `python/verify.py` | Correctness verification (Python vs reference) |
| `compile_model.sh` | Compile GGUF → `.eml` format |
| `paper.tex` | The paper |
| `paper.pdf` | Pre-built PDF of the paper |

### Binaries

- **`emilio`** — Main inference engine. Loads GGUF or compiled `.eml` files, runs chat/generation.
- **`autoeml`** — Kernel benchmarks and the autonomous optimization loop.
- **`eml`** — EML compiler (GGUF → `.eml` v1/v2).

## Quick start

### Prerequisites

- Rust ≥ 1.75
- Python 3.10+ with NumPy (for the Python POC)
- macOS with Apple Silicon (for GPU support)
- ~1 GB disk for the GGUF model

### Build

```bash
cd emilio

# CPU-only
cargo build --bin emilio --release

# With Metal GPU support (Apple Silicon)
cargo build --bin emilio --release --features metal
```

### Download the model

```bash
mkdir -p models && cd models
curl -L -o qwen2.5-0.5b-instruct-q8_0.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
cd ..
```

### Run inference

```bash
cd emilio

# CPU (~5.5 tok/s)
cargo run --bin emilio --release -- \
  ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --chat "What is 2+2?"

# GPU (~30 tok/s, Apple Silicon)
cargo run --features metal --bin emilio --release -- \
  ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --generate --gpu "What is 2+2?"
```

### Compile to .eml format (optional)

The `.eml` format precomputes `ln(W)` at compile time, so inference skips
dequantization and log computation at load time.

```bash
# v1: Complex64 encoding (~8.4 GB)
./compile_model.sh models/qwen2.5-0.5b-instruct-q8_0.gguf

# v2: sign+magnitude, fused projections, sparse pruning (~4.9 GB)
cd emilio
cargo run --bin emilio --release -- \
  ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --compile-v2 ../models/qwen2.5-0.5b-instruct-v2.eml
```

Then run from the compiled file:

```bash
cargo run --bin emilio --release -- \
  ../models/qwen2.5-0.5b-instruct-v2.eml \
  --chat "What is 2+2?"
```

## How it works

Standard matrix multiplication: `c[i,j] = Σ a[i,k] × b[k,j]`

EML matrix multiplication: `c[i,j] = Σ exp(ln|a| + ln|b|) × sign`

Every multiply in the entire forward pass — attention QKV projections, output projections,
FFN gate/up/down, RMSNorm scaling — goes through `exp(ln(a) + ln(b))`. Addition and
subtraction are free (they're native EML operations). The only non-EML operation is
argmax for token sampling.

### GPU kernels (Metal)

All 4 active Metal kernels are pure EML:

| Kernel | Purpose |
|---|---|
| `eml_matmul_v4` | SIMD matmul with packed sign bits |
| `eml_silu_mul_ln` | Log-domain SiLU activation |
| `eml_ln_split` | Decompose to `ln|x|` and `sign(x)` |
| `eml_residual_rms_norm_ln_split` | Fused residual add + RMSNorm in log domain |

### Optimization levels

| Level | Technique | Throughput |
|---|---|---|
| Python POC | Naive EML composition | ~0.002 tok/s |
| Rust baseline | Direct translation | ~0.5 tok/s |
| Precomputed ln(W) | Compile-time log | ~2.5 tok/s |
| CPU optimized | Transpose + SIMD + Rayon | ~5.5 tok/s |
| Metal GPU | Fused kernel chains | ~30 tok/s |

## Benchmarks

```bash
cd emilio

# Kernel microbenchmarks
cargo run --bin autoeml --release -- bench --transposed --iters 10

# At specific model dimensions (K=128, 896, 4864)
cargo run --bin autoeml --release -- bench --transposed --size 896 --iters 10
```

## Verification

```bash
# Python: verify EML forward pass matches reference
cd python
python3 verify.py

# Rust: verify Rust engine matches Python reference
python3 verify_rust.py
```

## The paper

The paper (`paper.tex` / `paper.pdf`) covers:
- Full EML derivation from Odrzywołek's primitive
- Call-count analysis (636K EML calls per token)
- Three algebraic optimization strategies
- Pure-EML RMSNorm derivation
- Metal GPU backend with kernel evolution
- Purity audit (every kernel verified EML-pure)

## License

[MIT](LICENSE)
