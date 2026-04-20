#!/usr/bin/env bash
set -euo pipefail

# ── build_model.sh ──────────────────────────────────────────────
# Full pipeline: GGUF → CoreML → compiled mlmodelc → Swift binary
#
# Usage:
#   ./build_model.sh <gguf_file> [--quant-bits 4] [--seq-len 4096] [--group-size 32]
#
# Examples:
#   ./build_model.sh ../../models/qwen2.5-0.5b-instruct-q8_0.gguf
#   ./build_model.sh ../../models/qwen2.5-1.5b-instruct-q8_0.gguf --quant-bits 4 --seq-len 4096
#
# The script:
#   1. Reads layer count from the GGUF file
#   2. Converts GGUF → CoreML mlpackage (stateful, with quantization)
#   3. Compiles mlpackage → mlmodelc via xcrun coremlcompiler
#   4. Creates symlinks so the Swift host can find the model
#   5. Builds the Swift inference binary (qwen_ane)
#
# After running, just:
#   ./qwen_ane --layers <N> --stateful --prompt "Hello" --max-tokens 50
# ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse args ──────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <gguf_file> [--quant-bits 0|4|8] [--seq-len N] [--group-size N]"
    echo ""
    echo "Examples:"
    echo "  $0 ../../models/qwen2.5-0.5b-instruct-q8_0.gguf"
    echo "  $0 ../../models/qwen2.5-1.5b-instruct-q8_0.gguf --quant-bits 4 --seq-len 4096"
    exit 1
fi

GGUF_PATH="$1"
shift

if [[ ! -f "$GGUF_PATH" ]]; then
    echo "ERROR: GGUF file not found: $GGUF_PATH"
    exit 1
fi

# Defaults
QUANT_BITS=0
SEQ_LEN=512
GROUP_SIZE=32

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quant-bits) QUANT_BITS="$2"; shift 2 ;;
        --seq-len)    SEQ_LEN="$2"; shift 2 ;;
        --group-size) GROUP_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Step 0: Detect layer count from GGUF ────────────────────────

echo "╔════════════════════════════════════════════════════════╗"
echo "║  build_model.sh — GGUF → ANE pipeline                 ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "  GGUF:       $GGUF_PATH"
echo "  Quant:      ${QUANT_BITS}-bit (0=fp16)"
echo "  Seq len:    $SEQ_LEN"
echo "  Group size: $GROUP_SIZE"
echo ""

# Read layer count from GGUF via Python one-liner
VENV_PYTHON="${SCRIPT_DIR}/../../.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: Python venv not found at $VENV_PYTHON"
    echo "Run: python -m venv ../../.venv && source ../../.venv/bin/activate && pip install coremltools gguf"
    exit 1
fi

N_LAYERS=$("$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '.')
from gguf_to_ane import GGUFModel
m = GGUFModel('$GGUF_PATH')
print(m.config()['n_layers'])
")

echo "  Detected:   ${N_LAYERS} layers"

# Build the suffix for filenames
SUFFIX=""
if [[ "$QUANT_BITS" == "4" ]]; then
    SUFFIX="_q4"
elif [[ "$QUANT_BITS" == "8" ]]; then
    SUFFIX="_q8"
fi
if [[ "$GROUP_SIZE" != "32" && "$QUANT_BITS" != "0" ]]; then
    SUFFIX="${SUFFIX}g${GROUP_SIZE}"
fi

BASE="QwenANE_${N_LAYERS}L_stateful${SUFFIX}"
LINK="QwenANE_${N_LAYERS}L_stateful"

echo "  Base name:  $BASE"
echo "  Symlink:    $LINK → $BASE"
echo ""

# ── Step 1: Convert GGUF → CoreML ───────────────────────────────

echo "── Step 1/4: Converting GGUF → CoreML mlpackage ──"
if [[ -d "${BASE}.mlpackage" && -f "${BASE}_embd.bin" ]]; then
    echo "  Skipping — ${BASE}.mlpackage already exists"
else
    "$VENV_PYTHON" gguf_to_ane.py "$GGUF_PATH" \
        --layers "$N_LAYERS" \
        --seq-len "$SEQ_LEN" \
        --quant-bits "$QUANT_BITS" \
        --group-size "$GROUP_SIZE" \
        --stateful
fi
echo ""

# ── Step 2: Compile mlpackage → mlmodelc ─────────────────────────

echo "── Step 2/4: Compiling mlpackage → mlmodelc ──"
if [[ -d "${BASE}.mlmodelc" ]]; then
    echo "  Skipping — ${BASE}.mlmodelc already exists"
else
    xcrun coremlcompiler compile "${BASE}.mlpackage" .
fi
echo ""

# ── Step 3: Create symlinks ──────────────────────────────────────

echo "── Step 3/4: Creating symlinks ──"
if [[ "$BASE" != "$LINK" ]]; then
    for ext in .mlpackage .mlmodelc _embd.bin _meta.json _tokenizer.json; do
        src="${BASE}${ext}"
        dst="${LINK}${ext}"
        if [[ -e "$src" || -d "$src" ]]; then
            rm -f "$dst"
            ln -sf "$src" "$dst"
            echo "  $dst → $src"
        fi
    done
else
    echo "  No symlinks needed (base == link)"
fi
echo ""

# ── Step 4: Build Swift binary ───────────────────────────────────

echo "── Step 4/4: Building Swift inference binary ──"
if [[ "qwen_ane.swift" -nt "qwen_ane" ]] || [[ ! -x "qwen_ane" ]]; then
    swiftc -O -framework CoreML -framework Foundation -o qwen_ane qwen_ane.swift
    echo "  Built qwen_ane ✓"
else
    echo "  Skipping — qwen_ane is up to date"
fi
echo ""

# ── Done ─────────────────────────────────────────────────────────

echo "Done! Run:"
echo ""
echo "  ./qwen_ane --layers $N_LAYERS --stateful --prompt \"Hello\" --max-tokens 50"
