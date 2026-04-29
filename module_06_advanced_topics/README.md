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

<table>
<thead><tr><th>Notebook</th><th>Topic</th></tr></thead>
<tbody>
<tr><td><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_06_advanced_topics/01_cublas_neural_models.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br/><a href="01_cublas_neural_models.ipynb">01_cublas_neural_models.ipynb</a></td><td>cuBLAS for rate networks and covariance</td></tr>
<tr><td><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_06_advanced_topics/02_cufft_signal_analysis.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br/><a href="02_cufft_signal_analysis.ipynb">02_cufft_signal_analysis.ipynb</a></td><td>cuFFT for LFP spectral analysis</td></tr>
<tr><td><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_06_advanced_topics/03_multi_gpu_scaling.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br/><a href="03_multi_gpu_scaling.ipynb">03_multi_gpu_scaling.ipynb</a></td><td>Multi-GPU domain decomposition</td></tr>
</tbody>
</table>

## Exercises

<table>
<thead><tr><th>Notebook</th><th>Description</th></tr></thead>
<tbody>
<tr><td><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_06_advanced_topics/exercises/ex06_stub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br/><a href="exercises/ex06_stub.ipynb">exercises/ex06_stub.ipynb</a></td><td>Implement GPU-accelerated spike-train cross-correlation</td></tr>
<tr><td><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_06_advanced_topics/exercises/ex06_solution.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br/><a href="exercises/ex06_solution.ipynb">exercises/ex06_solution.ipynb</a></td><td>Complete solution</td></tr>
</tbody>
</table>

## Prerequisites

- Modules 01–05 (especially Module 02 memory optimization)
- cuBLAS and cuFFT ship with CUDA Toolkit — no extra install needed
- Multi-GPU section requires access to a machine with ≥2 NVIDIA GPUs
