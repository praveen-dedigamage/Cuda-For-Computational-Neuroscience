#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p ../../../bin

echo "=== Compiling cublas_demo.cu ==="
nvcc -O2 -o ../../../bin/cublas_demo cublas_demo.cu -lcublas -lm
echo "Running cuBLAS demo..."
../../../bin/cublas_demo

echo ""
echo "=== Compiling cufft_lfp.cu ==="
nvcc -O2 -o ../../../bin/cufft_lfp cufft_lfp.cu -lcufft -lm
echo "Running cuFFT LFP analysis..."
../../../bin/cufft_lfp
