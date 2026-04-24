#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/emilio/conv-ane"
BINARY="/tmp/gemma_ane_smoke"
META="$ROOT_DIR/python/moe/out/gemma_swift_head_meta.json"
DEFAULT_PREFIX="$ROOT_DIR/python/moe/out/gemma_swift_t414"
PREFIX="${1:-$DEFAULT_PREFIX}"
META_OUT="${PREFIX}_meta.json"
LOGITS_OUT="${PREFIX}_logits_f32.bin"
SENTINEL="$ROOT_DIR/python/moe/out/.gemma_swift_t414_logit_gate_PASS"

if [[ -e "$META_OUT" || -e "$LOGITS_OUT" || -e "$SENTINEL" ]]; then
  echo "Refusing to overwrite existing Swift logit-gate artifacts." >&2
  echo "Choose a fresh prefix, for example:" >&2
  echo "  $0 $ROOT_DIR/python/moe/out/gemma_swift_t414_run2" >&2
  exit 2
fi

echo "== Gemma Swift prompt-logit gate =="
echo "Compiling gemma_ane.swift -> $BINARY"
(
  cd "$RUNTIME_DIR"
  swiftc -O -framework CoreML -o "$BINARY" gemma_ane.swift
)

echo
echo "Dumping prompt-position logits -> $PREFIX"
(
  cd "$RUNTIME_DIR"
  "$BINARY" \
    --meta "$META" \
    --prompt-ids 2,818,5279,529,7001,563 \
    --n-new 0 \
    --dump-prompt-logits-prefix "$PREFIX"
)

echo
echo "Comparing Swift dump against HF REAP golden"
(
  cd "$ROOT_DIR"
  python3 python/moe/gemma_swift_logit_gate.py --prefix "$PREFIX"
)

echo
echo "Swift prompt-logit gate PASS"