#!/usr/bin/env bash
# Energy bench for the privacy-filter end-to-end pipeline.
#
# Compares two backends on the same 8-sentence workload, sustained for
# ~30 s each so powermetrics can integrate a steady-state window:
#
#   ane  — pf_e2e Swift driver (ANE attn packs + per-expert MoE on ANE)
#   cpu  — PyTorch fp32 reference forward (Python, CPU only)
#
# Reports mJ/sentence per compute unit (CPU/GPU/ANE/PKG) for each backend,
# so we can compute the speedup AND the energy-per-result ratio.
#
# Requires sudo (powermetrics samples ane_power, cpu_power, gpu_power).
# Outputs land in tmp/energy/pf_*.
#
# Run from repo root:
#     bash scripts/pf_energy_probe.sh
#
# To dry-run (no sudo, no powermetrics — just exercise the workloads briefly):
#     PF_DRY_RUN=1 bash scripts/pf_energy_probe.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PY="/Applications/Xcode.app/Contents/Developer/usr/bin/python3"
PY313="$REPO/.venv313/bin/python"
SUSTAIN_S="${PF_SUSTAIN_S:-30}"

mkdir -p tmp/energy

# ---------------------------------------------------------------------------
# Workload: ANE (Swift pf_e2e in sustain mode)
# ---------------------------------------------------------------------------
workload_ane() {
    local log="$1"
    PF_SUSTAIN_S="$SUSTAIN_S" "$REPO/emilio/conv-ane/pf_e2e" 2>&1 | tee "$log"
}

# ---------------------------------------------------------------------------
# Workload: CPU (PyTorch reference) — sustain-loop wrapper writes BEGIN/END/ITERS
# ---------------------------------------------------------------------------
workload_cpu() {
    local log="$1"
    PYTHONPATH="$REPO/python/privacy/_vendor_src/opf_src" \
        PF_SUSTAIN_S="$SUSTAIN_S" \
        "$PY313" "$REPO/python/privacy/pf_cpu_sustain.py" 2>&1 | tee "$log"
}

# ---------------------------------------------------------------------------
# Dry run: just invoke each workload briefly to verify it works.
# ---------------------------------------------------------------------------
if [ -n "${PF_DRY_RUN:-}" ]; then
    echo "=== DRY RUN (no powermetrics, no sudo) ==="
    SUSTAIN_S=3
    echo "--- ane (3 s) ---"
    workload_ane tmp/energy/dry_ane.log
    echo "--- cpu (3 s) ---"
    workload_cpu tmp/energy/dry_cpu.log
    exit 0
fi

# ---------------------------------------------------------------------------
# Real run: powermetrics under sudo around each workload.
# ---------------------------------------------------------------------------
echo "=== priming sudo (you'll be prompted once) ==="
sudo -v
( while true; do sudo -n true; sleep 50; done ) &
SUDO_KEEPALIVE=$!
trap 'kill $SUDO_KEEPALIVE 2>/dev/null || true' EXIT

run_one() {
    local backend="$1"          # ane | cpu
    local plist="tmp/energy/pf_power_${backend}.plist"
    local log="tmp/energy/pf_${backend}.log"
    local window=$((SUSTAIN_S + 15))   # cover startup + sustain + tail

    echo
    echo "=== ${backend} workload + powermetrics (${window}s window) ==="
    rm -f "$plist" "$log"

    sudo powermetrics --samplers cpu_power,gpu_power,ane_power \
        -i 1000 -n "$window" --format plist > "$plist" &
    local pm=$!

    sleep 4   # let powermetrics settle

    if [ "$backend" = "ane" ]; then
        workload_ane "$log"
    else
        workload_cpu "$log"
    fi

    wait "$pm"

    echo "--- parse ${backend} ---"
    "$PY" "$REPO/python/moe/ane_energy_parse.py" "$plist" "$log"
}

run_one ane
run_one cpu

echo
echo "=== done. logs in tmp/energy/pf_* ==="
echo "summary: per-sentence mJ shown above for each backend; divide ANE by CPU for energy speedup."
