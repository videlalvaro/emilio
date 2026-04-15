#!/bin/bash
# build_mov.sh -- Build emilio MOV-only binary using the M/o/Vfuscator
#
# This script:
#   1. Clones and builds the movfuscator (with LCC frontend + softfloat)
#   2. Compiles eml_mov.c + eml_tokenizer.c with movcc
#   3. Produces emilio_mov (32-bit ELF, 100% MOV instructions)
#
# By default, builds in pure-MOV mode (--no-mov-flow): all control flow
# is via SIGSEGV fault handler, producing 100% MOV user code with zero
# computed jumps.  Use --fast for the jmp-table mode (~94% MOV).
#
# The movfuscator build is done inline (not via upstream build.sh) so
# that modern-GCC fixes can be applied at the right point in the
# process -- after LCC is cloned and reset, but before compilation.
#
# Requirements:
#   - Linux with 32-bit libc (native i386 or x86_64 with gcc-multilib)
#   - gcc, make, git, patch
#   - Or use the provided Dockerfile
#
# Usage:
#   ./build_mov.sh              # full build (100% MOV, fault-based flow)
#   ./build_mov.sh --fast        # build with jmp-table flow (~94% MOV)
#   ./build_mov.sh --gcc-test   # test with gcc first (sanity check)
#   ./build_mov.sh --verify     # build + verify MOV-only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
MOVFUSC_DIR="$BUILD_DIR/movfuscator"
MOVFUSC_REPO="https://github.com/xoreaxeaxeax/movfuscator.git"
LCC_REPO="https://github.com/drh/lcc"
LCC_COMMIT="3b3f01b4103cd7b519ae84bd1122c9b03233e687"

# ---- Helper functions ----

log() { echo "[build_mov] $*"; }
die() { echo "[build_mov] ERROR: $*" >&2; exit 1; }

# ---- Parse arguments ----

GCC_TEST=0
VERIFY=0
FAST_MODE=0
for arg in "$@"; do
    case "$arg" in
        --gcc-test) GCC_TEST=1 ;;
        --verify)   VERIFY=1 ;;
        --fast)     FAST_MODE=1 ;;
        --help|-h)
            echo "Usage: $0 [--gcc-test] [--verify] [--fast]"
            echo "  --gcc-test  Compile with gcc -std=c89 -m32 first (sanity check)"
            echo "  --verify    After building, verify the binary is MOV-only"
            echo "  --fast      Use jmp-table flow (~94%% MOV, faster execution)"
            echo "              Default: fault-based flow (100%% MOV user code)"
            exit 0
            ;;
    esac
done

# ---- GCC sanity check ----

if [ "$GCC_TEST" = "1" ]; then
    log "Testing compilation with gcc -std=c89 -m32..."
    mkdir -p "$BUILD_DIR"
    gcc -std=c89 -pedantic -Wall -Wextra -m32 \
        -o "$BUILD_DIR/emilio_test" \
        "$SCRIPT_DIR/eml_mov.c" \
        "$SCRIPT_DIR/eml_tokenizer.c" \
        -lm
    log "GCC test build succeeded: $BUILD_DIR/emilio_test"
    if [ "$VERIFY" = "0" ]; then exit 0; fi
fi

# ---- Check 32-bit libc ----

HAS_32BIT=0
for _p in /usr/lib32/libc.so /usr/lib/i386-linux-gnu/libc.so.6 \
          /lib/i386-linux-gnu/libc.so.6 /lib/libc.so.6; do
    [ -f "$_p" ] && HAS_32BIT=1 && break
done
if [ "$HAS_32BIT" = "0" ]; then
    log "WARNING: 32-bit libc not found. Install with:"
    log "  sudo apt-get install gcc-multilib libc6-dev-i386"
fi

# ---- Clone movfuscator ----

mkdir -p "$BUILD_DIR"
if [ ! -d "$MOVFUSC_DIR" ]; then
    log "Cloning movfuscator..."
    git clone "$MOVFUSC_REPO" "$MOVFUSC_DIR"
else
    log "Movfuscator already cloned"
fi

# ---- Build movfuscator (inline, replaces upstream build.sh) ----
#
# We replicate the steps from the upstream build.sh so that we can
# inject modern-GCC fixes between the LCC checkout (git reset --hard)
# and the compilation (make).  The upstream build.sh does these steps
# in one shot with no hook point, and the git reset would undo any
# patches applied beforehand.

MOVCC="$MOVFUSC_DIR/build/movcc"

if [ ! -f "$MOVCC" ]; then
    log "Building movfuscator..."
    cd "$MOVFUSC_DIR"

    # 1. Clone LCC frontend
    if [ ! -d "lcc" ]; then
        log "  Cloning LCC frontend..."
        git clone "$LCC_REPO"
    fi
    cd lcc && git reset --hard "$LCC_COMMIT" && cd ..

    # 2. Set up build directory
    export BUILDDIR="$(pwd)/build"
    mkdir -p "$BUILDDIR/include"
    cp -p -R lcc/include/x86/linux/* "$BUILDDIR/include"

    GCCLN=$(gcc --print-search-dirs | grep install | head -1 | cut -d " " -f 2-)
    ln -sfn "$GCCLN" "$BUILDDIR/gcc"

    # The movfuscator host.c uses -L$BUILDDIR/gcc/32 to find libgcc.
    # On native i386 there is no "32" multilib subdir, so create one
    # with a symlink back to libgcc.a in the GCC install directory.
    GCCDIR=$(readlink -f "$BUILDDIR/gcc")
    if [ ! -d "${GCCDIR}/32" ]; then
        mkdir -p "${GCCDIR}/32"
        ln -sf "${GCCDIR}/libgcc.a" "${GCCDIR}/32/libgcc.a"
    fi

    # 3. Apply movfuscator patches to LCC
    patch -N -r - lcc/src/bind.c  movfuscator/bind.patch    || true
    patch -N -r - lcc/makefile    movfuscator/makefile.patch || true
    patch -N -r - lcc/src/enode.c movfuscator/enode.patch    || true
    patch -N -r - lcc/src/gen.c   movfuscator/gen.patch      || true
    patch -N -r - lcc/src/expr.c  movfuscator/expr.patch     || true
    patch -N -r - lcc/etc/lcc.c   movfuscator/lcc.patch      || true

    # 4. Modern GCC fixes (GCC 12+ / 14+)
    #    - Suppress warnings promoted to hard errors (-w, -std=gnu90)
    #    - Add missing #include <string.h> in lburg
    log "  Applying modern-GCC compatibility fixes..."
    sed -i 's/^CC=gcc$/CC=gcc -w/'                            lcc/makefile
    sed -i 's/^CFLAGS=\(.*\)$/CFLAGS=\1 -std=gnu90 -w/'      lcc/makefile
    if ! grep -q '#include <string.h>' lcc/lburg/lburg.c; then
        sed -i '1i #include <string.h>' lcc/lburg/lburg.c
    fi

    # 5. Build LCC with the M/o/Vfuscator backend
    #    Single quotes so $(BUILDDIR) is expanded by make, not the shell.
    #    The \" produce literal quotes for the -D preprocessor flag.
    log "  Building LCC compiler..."
    make -C lcc HOSTFILE=../movfuscator/host.c \
        CFLAGS='-g -std=gnu90 -w -DLCCDIR=\"$(BUILDDIR)/\"' lcc
    make -C lcc all

    # 6. Create movcc symlink
    ln -sfn "$BUILDDIR/lcc" "$BUILDDIR/movcc"

    # 7. Build M/o/Vfuscator CRT libraries
    log "  Building CRT libraries..."
    "$BUILDDIR/movcc" movfuscator/crt0.c -o "$BUILDDIR/crt0.o" -c -Wf--crt0 -Wf--q
    "$BUILDDIR/movcc" movfuscator/crtf.c -o "$BUILDDIR/crtf.o" -c -Wf--crtf -Wf--q
    "$BUILDDIR/movcc" movfuscator/crtd.c -o "$BUILDDIR/crtd.o" -c -Wf--crtd -Wf--q

    "$BUILDDIR/movcc" movfuscator/crt0.c -o "$BUILDDIR/crt0_cf.o" -c -Wf--crt0 -Wf--q -Wf--no-mov-flow
    "$BUILDDIR/movcc" movfuscator/crtf.c -o "$BUILDDIR/crtf_cf.o" -c -Wf--crtf -Wf--q -Wf--no-mov-flow
    "$BUILDDIR/movcc" movfuscator/crtd.c -o "$BUILDDIR/crtd_cf.o" -c -Wf--crtd -Wf--q -Wf--no-mov-flow

    # 8. Build softfloat library (software FP via MOV)
    #    Only build the .o files we need -- skip the timesoftfloat test
    #    binary which fails to link with movcc (cannot find -lgcc).
    log "  Building softfloat library..."
    mkdir -p movfuscator/lib
    make -C softfloat clean
    make -C softfloat CC="$BUILDDIR/movcc" \
        softfloat32.o softfloat64.o softfloatfull.o
    cp softfloat/softfloat32.o   movfuscator/lib/softfloat32.o
    cp softfloat/softfloat64.o   movfuscator/lib/softfloat64.o
    cp softfloat/softfloatfull.o movfuscator/lib/softfloatfull.o

    make -C softfloat clean
    make -C softfloat CC="$BUILDDIR/movcc -Wf--no-mov-flow" \
        softfloat32.o softfloat64.o softfloatfull.o
    cp softfloat/softfloat32.o   movfuscator/lib/softfloat32_cf.o
    cp softfloat/softfloat64.o   movfuscator/lib/softfloat64_cf.o
    cp softfloat/softfloatfull.o movfuscator/lib/softfloatfull_cf.o

    make -C softfloat clean

    cd "$SCRIPT_DIR"
    log "Movfuscator built successfully"
else
    log "Movfuscator already built"
fi

# ---- Compile emilio with movcc ----

if [ ! -f "$MOVCC" ]; then
    die "movcc not found at $MOVCC"
fi

# Select build mode: pure-MOV (fault-based) or fast (jmp-table)
if [ "$FAST_MODE" = "1" ]; then
    log "Compiling emilio with movfuscator (jmp-table mode, ~94%% MOV)..."
    MOVFLOW_FLAG=""
    SOFTFLOAT_OBJ="$MOVFUSC_DIR/movfuscator/lib/softfloat64.o"
else
    log "Compiling emilio with movfuscator (pure-MOV mode, fault-based flow)..."
    log "  100%% MOV: all control flow via SIGSEGV handler -- zero jumps."
    MOVFLOW_FLAG="-Wf--no-mov-flow"
    SOFTFLOAT_OBJ="$MOVFUSC_DIR/movfuscator/lib/softfloat64_cf.o"
fi
log "  This will take a while -- every instruction becomes MOV."

# movcc compiles C89 through LCC -> MOV-only x86 assembly -> as -> ld
#
# Note: movcc links against libc for I/O functions (fopen, printf, etc.)
# These libc functions are NOT mov-only, but the emilio code itself IS.
# The software math functions (sw_exp, sw_log, etc.) are compiled to MOV,
# so no libm dependency exists.
#
# The softfloat64.o library is needed because the movfuscator converts
# all floating-point operations (double) to software float function calls
# (float64_add, float64_mul, int32_to_float64, etc.).
#
# In pure-MOV mode (--no-mov-flow), the movfuscator uses SIGSEGV-based
# control flow instead of computed jmp tables.  This eliminates the
# ~5% of .text that was data-as-code (lookup tables for jmp targets)
# and makes every user instruction a genuine MOV.

"$MOVCC" \
    $MOVFLOW_FLAG \
    -DEML_MOVCC \
    "$SCRIPT_DIR/eml_mov.c" \
    "$SCRIPT_DIR/eml_tokenizer.c" \
    "$SOFTFLOAT_OBJ" \
    -o "$BUILD_DIR/emilio_mov"

log "Build complete: $BUILD_DIR/emilio_mov"
ls -lh "$BUILD_DIR/emilio_mov"

# ---- Verify MOV-only (optional) ----

if [ "$VERIFY" = "1" ]; then
    log "Running MOV-only verification..."
    "$SCRIPT_DIR/verify_mov.sh" "$BUILD_DIR/emilio_mov"
fi

log "Done."
