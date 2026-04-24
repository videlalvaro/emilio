#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/emilio/conv-ane"
BINARY="/tmp/gemma_ane_smoke"
META="$ROOT_DIR/python/moe/out/gemma_swift_head_meta.json"

run_case() {
  local prompt="$1"
  local expect_prompt_ids="$2"
  local expect_generated_ids="$3"

  "$BINARY" \
    --meta "$META" \
    --prompt "$prompt" \
    --n-new 4 \
    --expect-prompt-ids "$expect_prompt_ids" \
    --expect-generated-ids "$expect_generated_ids"
}

echo "== Gemma Swift closed-form parity =="
echo "Compiling gemma_ane.swift -> $BINARY"
(
  cd "$RUNTIME_DIR"
  swiftc -O -framework CoreML -o "$BINARY" gemma_ane.swift
)

echo
echo "[1/3] gold_symbol"
run_case \
  "The chemical symbol for gold is" \
  "2,818,7395,5404,573,5122,563" \
  "16107,236761,108,818"

echo
echo "[2/3] silver_symbol"
run_case \
  "The chemical symbol for silver is" \
  "2,818,7395,5404,573,10173,563" \
  "3431,236761,108,818"

echo
echo "[3/3] iron_symbol"
run_case \
  "The chemical symbol for iron is" \
  "2,818,7395,5404,573,8603,563" \
  "5741,236761,108,818"

echo
echo "Closed-form Swift parity PASS"