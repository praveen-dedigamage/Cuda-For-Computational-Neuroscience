# Module 01 — GPU Architecture & CUDA Fundamentals

## Overview

This module builds your mental model of how GPUs work and how to write your first CUDA C++ programs. By the end you will be able to write, compile, and run a CUDA kernel that executes in parallel across thousands of threads.

## Learning Objectives

By completing this module you will be able to:

1. Explain the architectural differences between CPUs and GPUs
2. Describe the CUDA thread hierarchy: thread → warp → block → grid
3. Write a CUDA kernel and launch it with the correct configuration
4. Compute global thread indices for 1D and 2D problems
5. Use `cudaMalloc`, `cudaMemcpy`, and `cudaFree` correctly
6. Explain why parallelism matters for neuroscience simulations

## Notebooks

| Notebook | Topic |
|----------|-------|
| [01_gpu_architecture.ipynb](01_gpu_architecture.ipynb) | CPU vs GPU, SMs, warps, memory hierarchy |
| [02_cuda_programming_model.ipynb](02_cuda_programming_model.ipynb) | Kernels, thread hierarchy, indexing |
| [03_first_cuda_programs.ipynb](03_first_cuda_programs.ipynb) | Hello CUDA, vector addition, timing |

## Exercises

| File | Description |
|------|-------------|
| [exercises/ex01_stub.ipynb](exercises/ex01_stub.ipynb) | Fill-in-the-blank: write your first kernel |
| [exercises/ex01_solution.ipynb](exercises/ex01_solution.ipynb) | Complete solution |

## Standalone Source

| File | Description |
|------|-------------|
| [src/hello_cuda.cu](src/hello_cuda.cu) | Thread identity printing |
| [src/vector_add.cu](src/vector_add.cu) | GPU vector addition with timing |
| [src/run_local.sh](src/run_local.sh) | Compile and run both programs |

## Estimated Time

- Lectures: 3–4 hours
- Exercises: 1–2 hours
