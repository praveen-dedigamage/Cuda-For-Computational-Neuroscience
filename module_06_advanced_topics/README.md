# Module 06 — Advanced GPU Topics for Neuroscience

## Overview

This module covers GPU libraries and scaling strategies that unlock large-scale neuroscience simulations beyond what hand-written kernels alone can achieve.

## Topics

### cuBLAS — GPU-Accelerated Linear Algebra
Dense weight matrix operations: forward pass of rate-coded networks, covariance computation, PCA of spike trains. cuBLAS delivers near-peak FLOPS with a single API call.

### cuFFT — Spectral Analysis on GPU
Local field potential (LFP) power spectra, coherence between brain regions, and spectrogram computation. cuFFT achieves GPU-accelerated FFT with the same interface as CPU FFTW.

### Multi-GPU Scaling
Domain decomposition for networks too large for a single GPU. Communication via cudaMemcpyPeer (NVLink) or MPI. Benchmark scaling efficiency up to 4 GPUs.

## Learning Objectives

1. Use cuBLAS `cublasSgemm` for batched weight matrix multiplies
2. Compute LFP power spectra with cuFFT on real recorded data
3. Partition a spiking network across 2+ GPUs using domain decomposition
4. Understand the communication-to-computation ratio and Amdahl's Law for GPU scaling

## Notebooks

| Notebook | Topic |
|----------|-------|
| [01_cublas_neural_models.ipynb](01_cublas_neural_models.ipynb) | cuBLAS for rate networks and covariance |
| [02_cufft_signal_analysis.ipynb](02_cufft_signal_analysis.ipynb) | cuFFT for LFP spectral analysis |
| [03_multi_gpu_scaling.ipynb](03_multi_gpu_scaling.ipynb) | Multi-GPU domain decomposition |

## Exercises

| File | Description |
|------|-------------|
| [exercises/ex06_stub.ipynb](exercises/ex06_stub.ipynb) | Implement GPU-accelerated spike-train cross-correlation |
| [exercises/ex06_solution.ipynb](exercises/ex06_solution.ipynb) | Complete solution |

## Prerequisites

- Modules 01–05 (especially Module 02 memory optimization)
- cuBLAS and cuFFT ship with CUDA Toolkit — no extra install needed
- Multi-GPU section requires access to a machine with ≥2 NVIDIA GPUs
