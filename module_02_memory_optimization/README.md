# Module 02 — Memory Hierarchy & Optimization

## Overview

Performance in GPU programs is almost always limited by **memory bandwidth**, not compute. This module teaches you to diagnose and fix memory bottlenecks — skills that will directly improve your neuroscience simulation speeds by 2–10×.

## Learning Objectives

1. Identify the five GPU memory types and their speed/scope trade-offs
2. Write kernels that achieve coalesced global memory access
3. Use shared memory as a software-managed cache (tiling)
4. Detect and fix shared memory bank conflicts
5. Apply constant memory for simulation parameters
6. Profile kernels with `nvprof` / Nsight Compute

## Notebooks

| Notebook | Topic |
|----------|-------|
| [01_memory_hierarchy.ipynb](01_memory_hierarchy.ipynb) | Memory types, latency, bandwidth benchmarks |
| [02_shared_memory_tiling.ipynb](02_shared_memory_tiling.ipynb) | Tiled matrix multiply, bank conflicts |
| [03_optimization_workflow.ipynb](03_optimization_workflow.ipynb) | Profiling, roofline model, constant memory |

## Exercises

| File | Description |
|------|-------------|
| [exercises/ex02_stub.ipynb](exercises/ex02_stub.ipynb) | Optimize a naive matrix multiply using shared memory |
| [exercises/ex02_solution.ipynb](exercises/ex02_solution.ipynb) | Complete solution |

## Standalone Source

| File | Description |
|------|-------------|
| [src/matrix_mul.cu](src/matrix_mul.cu) | Naive vs tiled matrix multiplication |
| [src/bandwidth_test.cu](src/bandwidth_test.cu) | Benchmark all memory types |
| [src/run_local.sh](src/run_local.sh) | Compile and run |

## Why This Matters for Neuroscience

A synaptic weight matrix for N=1000 neurons is a 1000×1000 float matrix = 4 MB. Operations on it (summing inputs, updating weights with STDP) are all matrix operations. Achieving efficient memory access patterns on this matrix is the difference between a usable simulation and one that takes hours.
