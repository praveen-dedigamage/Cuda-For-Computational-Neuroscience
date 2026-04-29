#!/usr/bin/env bash
# Module 01 — compile and run all src programs locally
# Run from the repo root:  bash module_01_cuda_fundamentals/src/run_local.sh

set -e
cd "$(dirname "$0")"

mkdir -p ../../../bin

echo "=== Compiling hello_cuda.cu ==="
nvcc -O2 -o ../../../bin/hello_cuda hello_cuda.cu
echo "Running..."
../../../bin/hello_cuda
echo ""

echo "=== Compiling vector_add.cu ==="
nvcc -O2 -o ../../../bin/vector_add vector_add.cu -lm
echo "Running..."
../../../bin/vector_add
echo ""

echo "All Module 01 programs complete."
