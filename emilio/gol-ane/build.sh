#!/bin/bash
# GOL-ANE: Build and run Game of Life on Apple Neural Engine
#
# Usage: ./build.sh [grid_size] [n_generations]
#   grid_size:     side length of square grid (default: 64)
#   n_generations: GOL generations per model call (default: 32)

set -euo pipefail
cd "$(dirname "$0")"

GRID=${1:-64}
GENS=${2:-32}

echo "╔══════════════════════════════════════════════════════╗"
echo "║   GOL-ANE: Game of Life on Apple Neural Engine       ║"
echo "║   Grid: ${GRID}×${GRID}  ·  Generations: ${GENS}                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Step 0: Check dependencies
echo "==> Checking dependencies..."
python3 -c "import torch; import coremltools" 2>/dev/null || {
    echo "    Installing torch and coremltools (this may take a minute)..."
    pip install -q torch coremltools
}
echo "    ✓ torch + coremltools available"

# Step 1: Build CoreML model from PyTorch
echo ""
echo "==> Step 1: Building CoreML model (PyTorch → CoreML)..."
python3 build_model.py "$GRID" "$GENS"

# Step 2: Compile .mlpackage → .mlmodelc for on-device execution
echo ""
echo "==> Step 2: Compiling for ANE (xcrun coremlcompiler)..."
rm -rf GOL.mlmodelc
xcrun coremlcompiler compile GOL.mlpackage .
echo "    ✓ GOL.mlmodelc ready"

# Step 3: Compile Swift host
echo ""
echo "==> Step 3: Compiling Swift host..."
swiftc -O -framework CoreML -o gol_ane main.swift
echo "    ✓ gol_ane binary ready"

# Step 4: Run
echo ""
echo "==> Step 4: Running..."
echo "════════════════════════════════════════════════════════"
./gol_ane "$GENS"
