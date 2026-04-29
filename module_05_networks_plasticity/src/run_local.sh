#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p ../../../bin
echo "=== Compiling network_sim.cu ==="
nvcc -O2 -o ../../../bin/network_sim network_sim.cu -lm
echo "Running: 2000 neurons, 500 ms..."
../../../bin/network_sim 2000 500
