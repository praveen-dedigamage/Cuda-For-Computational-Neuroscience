#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p ../../../bin

echo "=== Compiling matrix_mul.cu ==="
nvcc -O2 -o ../../../bin/matrix_mul matrix_mul.cu -lm
echo "Running..."
../../../bin/matrix_mul
echo ""
echo "Module 02 programs complete."
