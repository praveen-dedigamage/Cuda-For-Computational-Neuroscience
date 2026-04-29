#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p ../../../bin ../../../results

echo "=== Compiling lif_simulation.cu ==="
nvcc -O2 -o ../../../bin/lif_simulation lif_simulation.cu -lm

echo "Running: 10,000 neurons, 1000 ms simulation..."
../../../bin/lif_simulation 10000 1000 0.1

echo ""
echo "Spike log written to spikes.txt"
echo "To visualize: run module_03_lif_neurons/03_spike_analysis.ipynb in Jupyter"
