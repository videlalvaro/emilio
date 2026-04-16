#!/bin/bash
# Energy benchmark for Conv-ANE (qwen_ane) vs llama.cpp
#
# Measures power consumption via powermetrics while running sustained
# inference workloads. Requires sudo for powermetrics.
#
# Usage: sudo ./energy_bench.sh [duration_seconds]
#   default duration: 120 seconds (2 minutes per engine)
#
# Output: energy_results_ane.txt, energy_results_llama.txt, energy_summary.txt

set -uo pipefail

DURATION=${1:-120}
SAMPLE_MS=1000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/energy_results"
mkdir -p "$RESULTS_DIR"

QWEN_ANE="$SCRIPT_DIR/qwen_ane"
LLAMA_BENCH="$REPO_DIR/../llama.cpp/build/bin/llama-bench"
MODEL_GGUF="$REPO_DIR/models/qwen2.5-0.5b-instruct-q8_0.gguf"

echo "╔══════════════════════════════════════════════════════╗"
echo "║   Energy Benchmark: Conv-ANE vs llama.cpp            ║"
echo "║   Duration: ${DURATION}s per engine                          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check binaries exist
for bin in "$QWEN_ANE" "$LLAMA_BENCH"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: $bin not found or not executable"
        exit 1
    fi
done

if [[ ! -f "$MODEL_GGUF" ]]; then
    echo "ERROR: $MODEL_GGUF not found"
    exit 1
fi

# ─── Helper: run workload with powermetrics ─────────────────────────
run_bench() {
    local label="$1"
    local power_log="$RESULTS_DIR/power_${label}.txt"
    local workload_log="$RESULTS_DIR/workload_${label}.txt"
    shift

    echo "==> [$label] Starting powermetrics (sampling every ${SAMPLE_MS}ms)..."
    powermetrics -s cpu_power,gpu_power \
        -i "$SAMPLE_MS" \
        -o "$power_log" &
    local PM_PID=$!

    # Give powermetrics a moment to start
    sleep 2

    echo "==> [$label] Running workload for ${DURATION}s..."
    local START_TIME=$(date +%s)
    local TOKEN_COUNT=0
    local RUN_COUNT=0

    while true; do
        local NOW=$(date +%s)
        local ELAPSED=$((NOW - START_TIME))
        if [[ $ELAPSED -ge $DURATION ]]; then
            break
        fi

        # Run inference, capture token count (|| true to not abort on crash)
        "$@" >> "$workload_log" 2>&1 || true
        RUN_COUNT=$((RUN_COUNT + 1))
        echo "    Run $RUN_COUNT completed (${ELAPSED}s / ${DURATION}s)"
    done

    local END_TIME=$(date +%s)
    local ACTUAL_DURATION=$((END_TIME - START_TIME))

    echo "==> [$label] Stopping powermetrics..."
    kill "$PM_PID" 2>/dev/null || true
    wait "$PM_PID" 2>/dev/null || true

    # Skip the first 2 samples (warmup) and parse power readings
    echo "==> [$label] Parsing power data..."

    local CPU_AVG=$(grep "CPU Power:" "$power_log" | tail -n +3 | \
        sed 's/.*: \([0-9]*\) mW/\1/' | \
        awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')

    local GPU_AVG=$(grep "^GPU Power:" "$power_log" | tail -n +3 | \
        sed 's/.*: \([0-9]*\) mW/\1/' | \
        awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')

    local ANE_AVG=$(grep "ANE Power:" "$power_log" | tail -n +3 | \
        sed 's/.*: \([0-9]*\) mW/\1/' | \
        awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')

    local COMBINED_AVG=$(grep "Combined Power" "$power_log" | tail -n +3 | \
        sed 's/.*: \([0-9]*\) mW/\1/' | \
        awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')

    local N_SAMPLES=$(grep "CPU Power:" "$power_log" | tail -n +3 | wc -l | tr -d ' ')

    echo ""
    echo "  ┌─────────────────────────────────────────┐"
    echo "  │ $label Results                          │"
    echo "  ├─────────────────────────────────────────┤"
    printf "  │ Duration:     %4ds                      │\n" "$ACTUAL_DURATION"
    printf "  │ Samples:      %4d                       │\n" "$N_SAMPLES"
    printf "  │ Runs:         %4d                       │\n" "$RUN_COUNT"
    echo "  │                                         │"
    printf "  │ CPU Power:    %5s mW                  │\n" "$CPU_AVG"
    printf "  │ GPU Power:    %5s mW                  │\n" "$GPU_AVG"
    printf "  │ ANE Power:    %5s mW                  │\n" "$ANE_AVG"
    printf "  │ Combined:     %5s mW                  │\n" "$COMBINED_AVG"
    echo "  └─────────────────────────────────────────┘"
    echo ""

    # Write summary line for later parsing
    echo "${label},${ACTUAL_DURATION},${N_SAMPLES},${RUN_COUNT},${CPU_AVG},${GPU_AVG},${ANE_AVG},${COMBINED_AVG}" \
        >> "$RESULTS_DIR/summary.csv"
}

# ─── Baseline (idle) ────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "Phase 0: Idle baseline (10s)"
echo "════════════════════════════════════════════════════════"

echo "label,duration_s,samples,runs,cpu_mw,gpu_mw,ane_mw,combined_mw" \
    > "$RESULTS_DIR/summary.csv"

IDLE_LOG="$RESULTS_DIR/power_idle.txt"
powermetrics -s cpu_power,gpu_power -i "$SAMPLE_MS" -n 12 -o "$IDLE_LOG"

IDLE_CPU=$(grep "CPU Power:" "$IDLE_LOG" | tail -n +3 | \
    sed 's/.*: \([0-9]*\) mW/\1/' | \
    awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')
IDLE_GPU=$(grep "^GPU Power:" "$IDLE_LOG" | tail -n +3 | \
    sed 's/.*: \([0-9]*\) mW/\1/' | \
    awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')
IDLE_ANE=$(grep "ANE Power:" "$IDLE_LOG" | tail -n +3 | \
    sed 's/.*: \([0-9]*\) mW/\1/' | \
    awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')
IDLE_COMBINED=$(grep "Combined Power" "$IDLE_LOG" | tail -n +3 | \
    sed 's/.*: \([0-9]*\) mW/\1/' | \
    awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')

echo "idle,10,10,0,${IDLE_CPU},${IDLE_GPU},${IDLE_ANE},${IDLE_COMBINED}" \
    >> "$RESULTS_DIR/summary.csv"

echo ""
echo "  Idle baseline: CPU=${IDLE_CPU}mW GPU=${IDLE_GPU}mW ANE=${IDLE_ANE}mW Combined=${IDLE_COMBINED}mW"
echo ""

# ─── Phase 1: Conv-ANE ─────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "Phase 1: Conv-ANE (qwen_ane on ANE) — ${DURATION}s"
echo "════════════════════════════════════════════════════════"

# Cool-down between tests
sleep 5

# qwen_ane uses relative paths — must run from its directory
pushd "$SCRIPT_DIR" > /dev/null
run_bench "conv_ane" \
    "$QWEN_ANE" --prompt "Explain the theory of computation, starting from Turing machines, through lambda calculus, to modern type theory. Cover the Church-Turing thesis, decidability, complexity classes P and NP, and their implications for practical computing." --max-tokens 512
popd > /dev/null

# ─── Phase 2: llama.cpp ────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo "Phase 2: llama.cpp (Metal GPU) — ${DURATION}s"
echo "════════════════════════════════════════════════════════"

# Cool-down between tests
sleep 10

run_bench "llama_cpp" \
    "$LLAMA_BENCH" -m "$MODEL_GGUF" \
    -t 4 -n 512 -p 64 -r 100

# ─── Final Summary ─────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                 ENERGY SUMMARY                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Raw data: $RESULTS_DIR/summary.csv"
echo ""
column -t -s',' "$RESULTS_DIR/summary.csv"
echo ""
echo "Done. Power logs saved in $RESULTS_DIR/"
