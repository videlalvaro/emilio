#!/usr/bin/env bash
# demo_redact.sh — feed diverse PII examples through the ANE privacy filter
# Usage: bash demo/demo_redact.sh
set -euo pipefail
cd "$(dirname "$0")/.."

INFER=".venv313/bin/python python/privacy/pf_infer.py"
INPUT="demo/pii_examples.txt"
DELAY="${DEMO_DELAY:-1.5}"   # seconds between lines (set DEMO_DELAY=0 for fast)

printf "\033[1m━━━ ANE Privacy Filter ━━━\033[0m\n\n"

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    printf "\033[90mIN:\033[0m  %s\n" "$line"
    out=$($INFER "$line" 2>/dev/null)
    printf "\033[32mOUT:\033[0m %s\n\n" "$out"
    sleep "$DELAY"
done < "$INPUT"
