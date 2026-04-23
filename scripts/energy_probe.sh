#!/usr/bin/env bash
# Probe D': energy baseline (ANE vs MPS) via powermetrics.
#
# Requires sudo (powermetrics needs root for ane_power).
# Run from the repo root:
#     bash scripts/energy_probe.sh
#
# Outputs land in tmp/energy/.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# coremltools + torch live in the Xcode-bundled python on this box,
# not in .venv / .venv313. Use it directly.
PY="/Applications/Xcode.app/Contents/Developer/usr/bin/python3"

mkdir -p tmp/energy

# Prime sudo once up-front so the background powermetrics doesn't stall.
echo "=== priming sudo (you'll be prompted once) ==="
sudo -v

# Keep sudo alive in the background until this script exits.
( while true; do sudo -n true; sleep 50; done ) &
SUDO_KEEPALIVE=$!
trap 'kill $SUDO_KEEPALIVE 2>/dev/null || true' EXIT

run_one() {
    local backend="$1"        # ane | mps
    local plist="tmp/energy/power_${backend}.plist"
    local log="tmp/energy/${backend}.log"

    echo
    echo "=== ${backend} workload + powermetrics ==="
    rm -f "$plist" "$log"

    # 45 samples * 1s = 45s window; workload runs ~30s inside it.
    sudo powermetrics --samplers cpu_power,gpu_power,ane_power \
        -i 1000 -n 45 --format plist > "$plist" &
    local pm=$!

    sleep 4  # let powermetrics settle

    "$PY" python/moe/ane_energy_probe.py "$backend" > "$log"

    wait "$pm"

    "$PY" python/moe/ane_energy_parse.py "$plist" "$log"
}

run_one ane
run_one mps

echo
echo "=== done. logs in tmp/energy/ ==="
