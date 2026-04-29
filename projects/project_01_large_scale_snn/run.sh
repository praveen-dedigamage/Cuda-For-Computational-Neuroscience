#!/usr/bin/env bash
set -e
echo "=== Project 01: Large-Scale SNN ==="
echo "Compiling..."
nvcc -O2 -o large_snn large_snn.cu -lm

echo ""
echo "--- Run 1: No STDP, N=10000, T=2000 ms ---"
./large_snn 10000 2000 0

echo ""
echo "--- Run 2: With STDP, N=10000, T=2000 ms ---"
./large_snn 10000 2000 1

echo ""
echo "--- Scaling benchmark (no STDP) ---"
for N in 5000 10000 20000 50000; do
    echo -n "N=$N: "
    ./large_snn $N 500 0 | grep Throughput
done
