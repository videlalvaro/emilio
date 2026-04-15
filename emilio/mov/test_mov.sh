#!/bin/bash
# test_mov.sh -- Combined CI test for the emilio MOV-only pipeline
#
# Runs:
#   1. GCC/Clang C89 compilation (sanity check, zero warnings required)
#   2. Self-test (software math, EML ops, matmul, softmax, RMSNorm, SiLU, RoPE)
#   3. Main binary smoke test (usage message)
#   4. [Optional] MOV build + verification (requires movfuscator or --full)
#
# Works on:
#   - Linux x86/x86_64 (gcc)
#   - macOS x86_64 / Apple Silicon (clang)
#
# Usage:
#   ./test_mov.sh            # Steps 1-3 only (fast, any platform)
#   ./test_mov.sh --full     # Steps 1-4 (slow, needs 32-bit Linux + movfuscator)
#   ./test_mov.sh --docker   # Steps 1-3 locally + step 4 via Docker
#
# Exit code 0 = all tests pass.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
PASS=0
FAIL=0
SKIP=0

# ---- Helpers ----

log()  { echo "[test] $*"; }
pass() { echo "[PASS] $*"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $*" >&2; FAIL=$((FAIL + 1)); }
skip() { echo "[SKIP] $*"; SKIP=$((SKIP + 1)); }

# ---- Parse arguments ----

FULL=0
DOCKER=0
for arg in "$@"; do
    case "$arg" in
        --full)   FULL=1 ;;
        --docker) DOCKER=1 ;;
        --help|-h)
            echo "Usage: $0 [--full] [--docker]"
            echo "  --full     Build with movfuscator and verify (needs 32-bit Linux)"
            echo "  --docker   Build and verify via Docker (works on macOS + Linux)"
            exit 0
            ;;
    esac
done

# ---- Detect platform and compiler ----

OS="$(uname -s)"
ARCH="$(uname -m)"
CC="${CC:-cc}"

# Find a working C89 compiler
if command -v gcc &>/dev/null; then
    CC=gcc
elif command -v clang &>/dev/null; then
    CC=clang
elif command -v cc &>/dev/null; then
    CC=cc
else
    log "ERROR: No C compiler found (tried gcc, clang, cc)"
    exit 1
fi

log "Platform: $OS / $ARCH"
log "Compiler: $CC ($($CC --version 2>&1 | head -1))"

# ---- Step 1: C89 strict compilation ----

log "Step 1: C89 strict compilation ($CC)"

mkdir -p "$BUILD_DIR"

# 1a. Main binary (no EML_NO_MAIN)
WARN_MAIN=$($CC -std=c89 -pedantic -Wall -Wextra \
    -o "$BUILD_DIR/emilio_gcc" \
    "$SCRIPT_DIR/eml_mov.c" \
    "$SCRIPT_DIR/eml_tokenizer.c" \
    -lm 2>&1) || {
    fail "Main binary failed to compile"
    echo "$WARN_MAIN"
}

if [ -z "$WARN_MAIN" ]; then
    pass "Main binary: $CC -std=c89 -pedantic -Wall -Wextra (zero warnings)"
else
    fail "Main binary compiled with warnings:"
    echo "$WARN_MAIN"
fi

# 1b. Test binary (EML_NO_MAIN)
WARN_TEST=$($CC -std=c89 -pedantic -Wall -Wextra -DEML_NO_MAIN \
    -o "$BUILD_DIR/eml_test" \
    "$SCRIPT_DIR/eml_test.c" \
    "$SCRIPT_DIR/eml_mov.c" \
    "$SCRIPT_DIR/eml_tokenizer.c" \
    -lm 2>&1) || {
    fail "Test binary failed to compile"
    echo "$WARN_TEST"
}

if [ -z "$WARN_TEST" ]; then
    pass "Test binary: $CC -std=c89 -pedantic -Wall -Wextra -DEML_NO_MAIN (zero warnings)"
else
    fail "Test binary compiled with warnings:"
    echo "$WARN_TEST"
fi

# ---- Step 2: Self-test ----

log "Step 2: Self-test"

if [ -f "$BUILD_DIR/eml_test" ]; then
    TEST_OUTPUT=$("$BUILD_DIR/eml_test" 2>&1)
    TEST_EXIT=$?
    echo "$TEST_OUTPUT"

    if [ "$TEST_EXIT" -eq 0 ]; then
        # Extract pass/fail counts (portable -- no grep -P)
        TPASS=$(echo "$TEST_OUTPUT" | sed -n 's/.*[^0-9]\([0-9][0-9]*\) passed.*/\1/p' | tail -1)
        TFAIL=$(echo "$TEST_OUTPUT" | sed -n 's/.*[^0-9]\([0-9][0-9]*\) failed.*/\1/p' | tail -1)
        pass "Self-test: $TPASS passed, $TFAIL failed"
    else
        fail "Self-test exited with code $TEST_EXIT"
    fi
else
    skip "Self-test binary not found (compilation failed)"
fi

# ---- Step 3: Main binary smoke test ----

log "Step 3: Main binary smoke test"

if [ -f "$BUILD_DIR/emilio_gcc" ]; then
    USAGE=$("$BUILD_DIR/emilio_gcc" 2>&1 || true)

    if echo "$USAGE" | grep -q "Usage:"; then
        pass "Main binary: usage message displayed"
    else
        fail "Main binary: no usage message"
        echo "$USAGE"
    fi

    if echo "$USAGE" | grep -qE "100.*MOV|MOV-only"; then
        pass "Main binary: MOV branding present"
    else
        fail "Main binary: missing MOV branding"
    fi
else
    skip "Main binary not found (compilation failed)"
fi

# ---- Step 4: MOV build + verification (optional) ----

log "Step 4: MOV build + verification"

if [ "$DOCKER" = "1" ]; then
    # Docker-based build: works on macOS (Intel + Apple Silicon) and Linux.
    # Uses QEMU emulation for i386 on non-x86 hosts.
    log "Building 100%% MOV binary via Docker..."
    DOCKER_PLATFORM=""
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        DOCKER_PLATFORM="--platform linux/386"
        log "  (ARM host detected -- using QEMU emulation, this will be slow)"
    fi

    if docker build $DOCKER_PLATFORM -t emilio-mov -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR" 2>&1; then
        pass "Docker build succeeded (includes movfuscator build + verify)"

        # Extract binary
        mkdir -p "$BUILD_DIR"
        if docker run --rm $DOCKER_PLATFORM -v "$BUILD_DIR:/out" emilio-mov 2>&1; then
            pass "MOV binary extracted: $(ls -lh "$BUILD_DIR/emilio_mov" 2>/dev/null | awk '{print $5}')"
        else
            fail "Failed to extract MOV binary from Docker"
        fi
    else
        fail "Docker build failed"
    fi

elif [ "$FULL" = "1" ]; then
    MOVCC="$BUILD_DIR/movfuscator/movcc"
    if [ ! -f "$MOVCC" ]; then
        log "Building movfuscator (this takes a few minutes)..."
        "$SCRIPT_DIR/build_mov.sh"
    fi

    if [ -f "$MOVCC" ]; then
        log "Compiling with movfuscator --no-mov-flow (this takes a LONG time)..."
        "$MOVCC" \
            -Wf--no-mov-flow \
            -DEML_MOVCC \
            "$SCRIPT_DIR/eml_mov.c" \
            "$SCRIPT_DIR/eml_tokenizer.c" \
            -o "$BUILD_DIR/emilio_mov"

        if [ -f "$BUILD_DIR/emilio_mov" ]; then
            pass "MOV binary compiled: $(ls -lh "$BUILD_DIR/emilio_mov" | awk '{print $5}')"

            log "Verifying MOV-only..."
            if "$SCRIPT_DIR/verify_mov.sh" "$BUILD_DIR/emilio_mov"; then
                pass "MOV-only verification passed"
            else
                fail "MOV-only verification failed"
            fi
        else
            fail "MOV binary not produced"
        fi
    else
        fail "movcc not found after build"
    fi
elif [ -f "$BUILD_DIR/emilio_mov" ]; then
    log "Found existing MOV binary, running verification..."
    if "$SCRIPT_DIR/verify_mov.sh" "$BUILD_DIR/emilio_mov"; then
        pass "MOV-only verification passed (existing binary)"
    else
        fail "MOV-only verification failed (existing binary)"
    fi
else
    skip "MOV build (use --full on Linux or --docker on any platform)"
fi

# ---- Summary ----

echo ""
echo "=============================="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"
echo "=============================="

if [ "$FAIL" -gt 0 ]; then
    echo "SOME TESTS FAILED"
    exit 1
fi

echo "ALL TESTS PASSED"
exit 0
