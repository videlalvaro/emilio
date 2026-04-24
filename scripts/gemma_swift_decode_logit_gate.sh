#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/emilio/conv-ane"
BINARY="/tmp/gemma_ane_smoke"
META="$ROOT_DIR/python/moe/out/gemma_swift_head_meta.json"
DEFAULT_PREFIX="/tmp/gemma_swift_t415_decode_smoke"
PREFIX="${1:-$DEFAULT_PREFIX}"
META_OUT="${PREFIX}_meta.json"
LOGITS_OUT="${PREFIX}_logits_f32.bin"

if [[ -e "$META_OUT" || -e "$LOGITS_OUT" ]]; then
  echo "Refusing to overwrite existing Swift decode-logit smoke artifacts." >&2
  echo "Choose a fresh prefix, for example:" >&2
  echo "  $0 /tmp/gemma_swift_t415_decode_smoke_run2" >&2
  exit 2
fi

echo "== Gemma Swift decode-logit gate =="
echo "Compiling gemma_ane.swift -> $BINARY"
(
  cd "$RUNTIME_DIR"
  swiftc -O -framework CoreML -o "$BINARY" gemma_ane.swift
)

echo
echo "Dumping bounded decode logits -> $PREFIX"
(
  cd "$RUNTIME_DIR"
  "$BINARY" \
    --meta "$META" \
    --prompt-ids 2,3689,563,506,5279,529,7001,236881 \
    --n-new 2 \
    --dump-decode-logits-prefix "$PREFIX"
)

echo
echo "Comparing Swift decode dump against gemma_golden.npz"
(
  cd "$ROOT_DIR"
  python3 python/moe/gemma_swift_decode_logit_gate.py --prefix "$PREFIX"
)

echo
echo "Swift decode-logit gate PASS"