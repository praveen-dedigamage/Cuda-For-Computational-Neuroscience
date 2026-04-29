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

<table>
<thead><tr><th>Notebook</th><th>Topic</th></tr></thead>
<tbody>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_02_memory_optimization/01_memory_hierarchy.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="01_memory_hierarchy.ipynb">01_memory_hierarchy.ipynb</a></td><td valign="middle">Memory types, latency, bandwidth benchmarks</td></tr>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_02_memory_optimization/02_shared_memory_tiling.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="02_shared_memory_tiling.ipynb">02_shared_memory_tiling.ipynb</a></td><td valign="middle">Tiled matrix multiply, bank conflicts</td></tr>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_02_memory_optimization/03_optimization_workflow.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="03_optimization_workflow.ipynb">03_optimization_workflow.ipynb</a></td><td valign="middle">Profiling, roofline model, constant memory</td></tr>
</tbody>
</table>

## Exercises

<table>
<thead><tr><th>Notebook</th><th>Description</th></tr></thead>
<tbody>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_02_memory_optimization/exercises/ex02_stub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="exercises/ex02_stub.ipynb">exercises/ex02_stub.ipynb</a></td><td valign="middle">Optimize a naive matrix multiply using shared memory</td></tr>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_02_memory_optimization/exercises/ex02_solution.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="exercises/ex02_solution.ipynb">exercises/ex02_solution.ipynb</a></td><td valign="middle">Complete solution</td></tr>
</tbody>
</table>

## Standalone Source

| File | Description |
|------|-------------|
| [src/matrix_mul.cu](src/matrix_mul.cu) | Naive vs tiled matrix multiplication |
| [src/bandwidth_test.cu](src/bandwidth_test.cu) | Benchmark all memory types |
| [src/run_local.sh](src/run_local.sh) | Compile and run |

## Why This Matters for Neuroscience

A synaptic weight matrix for N=1000 neurons is a 1000×1000 float matrix = 4 MB. Operations on it (summing inputs, updating weights with STDP) are all matrix operations. Achieving efficient memory access patterns on this matrix is the difference between a usable simulation and one that takes hours.
