# emilio MOV-only inference engine

A complete port of [emilio](../)'s v2 inference path to C89, compiled
using the [M/o/Vfuscator](https://github.com/xoreaxeaxeax/movfuscator)
to produce a binary that uses **only the x86 MOV instruction**.

This proves that a transformer-based LLM can run using nothing but data
movement — no arithmetic instructions, no jumps, no comparisons — just MOV.

Based on Stephen Dolan's proof that MOV is Turing-complete, and Chris Domas's
M/o/Vfuscator compiler.

## What's included

| File | Purpose |
|------|---------|
| `eml_mov.h` | C89 structs, constants, function declarations |
| `eml_mov.c` | Full inference engine (~1200 lines): software math, EML ops, matmul, attention, generation, model loader |
| `eml_tokenizer.c` | BPE tokenizer: encode + decode with ChatML support |
| `eml_test.c` | Self-test: validates software math, EML ops, matmul, softmax, RMSNorm, SiLU, RoPE (61 checks) |
| `build_mov.sh` | Build script: clones movfuscator, builds, compiles |
| `verify_mov.sh` | Verification: disassembles binary and checks for MOV-only |
| `test_mov.sh` | Combined CI test: GCC compile + self-test + smoke test + optional MOV verify |
| `Dockerfile` | Reproducible 32-bit build environment |

## How it works

### The EML primitive

All arithmetic in emilio derives from a single gate: `eml(x, y) = exp(x) - ln(y)`.

Addition and subtraction are free (algebraic cancellation). Multiplication is
`exp(ln(a) + ln(b))`. The only irreducible operations are `exp()` and `ln()`.

### Software math

The `exp()` and `ln()` functions are implemented in pure C89 using Taylor series
and range reduction — **no libm dependency**. When compiled by the movfuscator,
these become sequences of MOV instructions operating through lookup tables.

`sin()`, `cos()`, `sqrt()`, `atan2()` are also implemented in software for
RoPE positional encoding precomputation.

### Sign+magnitude matmul

The v2 format pre-encodes weights as `(ln|w|, sign(w))` pairs. The inner
matmul loop is pure `f64`:

```c
e = exp(la_mag[k] + w_mag[k]);
acc += e * la_sign[k] * w_sign[k];
```

No complex numbers in the hot path.

## Building

### Option 1: Docker (recommended)

```bash
docker build -t emilio-mov -f mov/Dockerfile mov/
docker run --rm -v $(pwd)/mov/build:/out emilio-mov
# Binary: ./mov/build/emilio_mov
```

### Option 2: Native Linux (x86/x86_64)

Requirements:
- GCC, make, git
- 32-bit libc: `sudo apt-get install gcc-multilib libc6-dev-i386`

```bash
# Full build (clones movfuscator, builds, compiles)
./build_mov.sh

# Test with gcc first (sanity check)
./build_mov.sh --gcc-test

# Build + verify MOV-only
./build_mov.sh --verify
```

### Sanity check with GCC

Before attempting the movfuscator build, verify the C89 code compiles:

```bash
gcc -std=c89 -pedantic -Wall -m32 eml_mov.c eml_tokenizer.c -o emilio_test -lm
```

## Testing

Run the combined test script (no movfuscator needed):

```bash
./test_mov.sh
```

This runs:
1. **GCC C89 strict compile** -- both main and test binaries, zero warnings required
2. **Self-test** -- 61 checks covering software math, EML ops, matmul, softmax, RMSNorm, SiLU, RoPE
3. **Smoke test** -- main binary prints usage message with correct branding

For the full pipeline including movfuscator build + MOV-only verification:

```bash
./test_mov.sh --full
```

Expected output:
```
[PASS] Main binary: gcc -std=c89 -pedantic -Wall -Wextra (zero warnings)
[PASS] Test binary: gcc -std=c89 -pedantic -Wall -Wextra -DEML_NO_MAIN (zero warnings)
[PASS] Self-test: 61 passed, 0 failed
[PASS] Main binary: usage message displayed
[PASS] Main binary: MOV-only branding present
ALL TESTS PASSED
```

## Usage

```bash
./build/emilio_mov model.eml "What is 2+2?" 32
```

Arguments:
1. Path to a `.eml` v2 model file (compiled from GGUF by the Rust emilio)
2. Prompt text
3. Max tokens to generate (default: 64)

## Verification

```bash
./verify_mov.sh ./build/emilio_mov
```

This disassembles the binary and checks that >90% of instructions in `.text`
are MOV variants. Expected output:

```
=== MOV-Only Verification ===
Total instructions in .text: ~637000
MOV instructions: ~603000
MOV ratio: 94%
PASS: Binary is a valid M/o/Vfuscator binary (94% MOV)
```

The M/o/Vfuscator embeds large data lookup tables in `.text` for computed
branching. `objdump` linearly disassembles these data bytes as random x86
instructions (~5% of `.text`). These are never executed. A typical
movfuscator binary is 93-97% MOV; a normal GCC binary is <30% MOV.

## Performance expectations

This is a proof-of-concept, not a practical inference engine.

- **Binary size**: ~13MB (lookup tables for MOV-based arithmetic)
- **Speed**: Very slow. Each `exp()` call expands to thousands of MOV
  instructions. A single forward pass requires ~136 million `exp()` calls.
- **Estimated time per token**: Hours to days (vs. milliseconds for the
  native Rust version)

## Architecture

```
Rust emilio                          C89 MOV-only emilio
─────────────────                    ─────────────────────
eml_ops.rs          ──────────>      eml_mov.c (EML ops)
eml_v2.rs           ──────────>      eml_mov.c (matmul, attention)
engine.rs           ──────────>      eml_mov.c (RMSNorm, RoPE, KV cache)
eml_format.rs       ──────────>      eml_mov.c (v2 loader)
tokenizer.rs        ──────────>      eml_tokenizer.c
<math.h>            ──────────>      eml_mov.c (sw_exp, sw_log, sw_sin, ...)
Rayon par_iter      ──────────>      Sequential loops
Complex64           ──────────>      f64 sign+magnitude (v2 path)
Vec<f64>            ──────────>      malloc/free
HashMap             ──────────>      Sorted array + binary search
```

## References

- Dolan, S. "mov is Turing-complete." https://drwho.virtadpt.net/files/mov.pdf
- Domas, C. M/o/Vfuscator. https://github.com/xoreaxeaxeax/movfuscator
- Odrzywołek (2026) arXiv:2603.21852
