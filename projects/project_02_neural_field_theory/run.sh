#!/usr/bin/env bash
set -e
echo "=== Project 02: Neural Field Theory ==="
echo "Compiling..."
nvcc -O2 -o neural_field neural_field.cu -lcufft -lm

echo ""
echo "--- Run: 256×256 grid, 500 ms ---"
./neural_field 256 500

echo ""
echo "--- Scaling benchmark ---"
for G in 64 128 256 512; do
    echo -n "Grid ${G}×${G}: "
    ./neural_field $G 200 | grep Throughput
done
