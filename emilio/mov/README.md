# emilio — 100% MOV inference engine

A complete port of [emilio](../)'s v2 inference path to C89, compiled
using the [M/o/Vfuscator](https://github.com/xoreaxeaxeax/movfuscator)
with `--no-mov-flow` to produce a binary where **every single user
instruction is MOV**.

No jumps. No arithmetic. No comparisons. No calls. Just MOV.

This proves that a transformer-based LLM can run using nothing but data
movement — control flow is via SIGSEGV fault handler, arithmetic is via
lookup tables, and branching is via computed memory addressing. All MOV.

Based on Stephen Dolan's proof that MOV is Turing-complete, and Chris Domas's
M/o/Vfuscator compiler.

## What's included

| File | Purpose |
|------|---------|
| `eml_mov.h` | C89 structs, constants, function declarations |
| `eml_mov.c` | Full inference engine (~1200 lines): software math, EML ops, matmul, attention, generation, model loader |
| `eml_tokenizer.c` | BPE tokenizer: encode + decode with ChatML support |
| `eml_test.c` | Self-test: validates software math, EML ops, matmul, softmax, RMSNorm, SiLU, RoPE (61 checks) |
| `build_mov.sh` | Build script: clones movfuscator, builds, compiles (100% MOV by default) |
| `verify_mov.sh` | Verification: disassembles binary, checks 100% MOV user code |
| `test_mov.sh` | Combined CI test: GCC compile + self-test + smoke test + optional MOV verify |
| `Dockerfile` | Reproducible 32-bit build environment |

## How it works

### The EML primitive

All arithmetic in emilio derives from a single gate: `eml(x, y) = exp(x) - ln(y)`.

Addition and subtraction are free (algebraic cancellation). Multiplication is
`exp(ln(a) + ln(b))`. The only irreducible operations are `exp()` and `ln()`.

### Software math

The `exp()` and `ln()` functions are implemented in pure C89 using Taylor series
and range reduction — **no libm dependency**. When compiled by the movfuscator
with `--no-mov-flow`, these become pure MOV sequences — arithmetic is performed
through software float library calls (also MOV-only), and control flow is via
SIGSEGV fault handler dispatch.

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

### Option 1: Docker (recommended — works on any platform)

```bash
# x86 host (Linux or Intel Mac)
docker build -t emilio-mov -f mov/Dockerfile mov/

# Apple Silicon / ARM host (uses QEMU emulation)
docker build --platform linux/386 -t emilio-mov -f mov/Dockerfile mov/

# Extract the binary
docker run --rm -v $(pwd)/mov/build:/out emilio-mov
# Binary: ./mov/build/emilio_mov
```

### Option 2: Native Linux (x86/x86_64)

Requirements:
- GCC, make, git
- 32-bit libc: `sudo apt-get install gcc-multilib libc6-dev-i386`

```bash
# Full build -- 100% MOV (fault-based flow, default)
./build_mov.sh

# Test with gcc first (sanity check)
./build_mov.sh --gcc-test

# Build + verify 100% MOV
./build_mov.sh --verify

# Fast mode (~94% MOV, jmp-table flow, faster execution)
./build_mov.sh --fast
```

### Compile test on any platform (macOS, Linux, WSL)

The C89 source compiles and runs natively on any platform with a C compiler
(GCC, Clang, MSVC). This exercises the full inference logic — the only
difference is that the resulting binary uses normal instructions instead of MOV:

```bash
# macOS (Apple Silicon or Intel)
clang -std=c89 -pedantic -Wall eml_mov.c eml_tokenizer.c -o emilio_test -lm

# Linux (any arch)
gcc -std=c89 -pedantic -Wall eml_mov.c eml_tokenizer.c -o emilio_test -lm
```

## Testing

Run the combined test script (works on macOS, Linux, WSL):

```bash
./test_mov.sh
```

This runs:
1. **C89 strict compile** (GCC or Clang) -- both main and test binaries, zero warnings required
2. **Self-test** -- 61 checks covering software math, EML ops, matmul, softmax, RMSNorm, SiLU, RoPE
3. **Smoke test** -- main binary prints usage message with correct branding

For the full MOV pipeline via Docker (any platform):

```bash
./test_mov.sh --docker
```

For the full MOV pipeline natively (needs 32-bit Linux):

```bash
./test_mov.sh --full
```

Expected output:
```
[PASS] Main binary: gcc -std=c89 -pedantic -Wall -Wextra (zero warnings)
[PASS] Test binary: gcc -std=c89 -pedantic -Wall -Wextra -DEML_NO_MAIN (zero warnings)
[PASS] Self-test: 61 passed, 0 failed
[PASS] Main binary: usage message displayed
[PASS] Main binary: MOV branding present
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

The verification script analyzes the binary at two levels:

1. **User code** (emilio functions + softfloat): must be **100% MOV**
2. **Full binary** (includes PLT stubs + CRT bootstrap): typically ~98% MOV

Expected output:
```
=== User Code Analysis ===
  User code instructions: ~580000
  User code MOV:          ~580000
  User code non-MOV:      0
  User code MOV ratio:    100%

PASS: 100% MOV -- every user instruction is MOV (580000 instructions)
  Full binary: 98% (120 non-MOV in PLT/CRT only)
```

The few non-MOV instructions in the full binary come from:
- **PLT stubs** (~3 per libc function: jmp/push for dynamic linking)
- **CRT bootstrap** (signal handler setup for fault-based control flow)

These are linker/OS infrastructure, not emilio code. Every instruction that
the movfuscator compiled from our C89 source is a MOV.

### How --no-mov-flow works

The default movfuscator uses computed `jmp` tables for control flow, which
embeds large data lookup tables in `.text` (~5% of the binary). These are
never executed but objdump counts them as non-MOV instructions.

With `--no-mov-flow`, the movfuscator uses a different strategy: it MOVs
to an unmapped address, triggering SIGSEGV. A signal handler (set up once
in CRT) dispatches to the correct continuation using only MOV. This
eliminates all jump tables and produces pure MOV user code.

## Performance expectations

This is a proof-of-concept, not a practical inference engine.

- **Binary size**: ~13MB (MOV-based arithmetic + softfloat tables)
- **Speed**: Very slow. Each `exp()` call expands to thousands of MOV
  instructions. A single forward pass requires ~136 million `exp()` calls.
  The `--no-mov-flow` mode adds additional overhead from SIGSEGV dispatch.
- **Estimated time per token**: Hours to days (vs. milliseconds for the
  native Rust version)
- **Build modes**: Use `--fast` for ~10x faster execution (jmp-table flow,
  ~94% MOV) or default for 100% MOV purity.

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
