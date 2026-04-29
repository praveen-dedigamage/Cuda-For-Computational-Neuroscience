#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p ../../../bin

echo "=== Compiling hh_simulation.cu ==="
nvcc -O2 -o ../../../bin/hh_simulation hh_simulation.cu -lm

echo "Running Euler solver: 1000 neurons, 100 ms..."
../../../bin/hh_simulation 1000 100 euler

echo ""
echo "Running RK4 solver: 1000 neurons, 100 ms..."
../../../bin/hh_simulation 1000 100 rk4
