/*
 * vector_add.cu
 * Module 01 — GPU Architecture & CUDA Fundamentals
 *
 * GPU vector addition: C[i] = A[i] + B[i]
 * Demonstrates: cudaMalloc, cudaMemcpy, kernel indexing, cudaEvent timing.
 *
 * Compile:  nvcc -O2 -o vector_add vector_add.cu
 * Run:      ./vector_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// GPU kernel: each thread adds one pair of elements
// ─────────────────────────────────────────────────────────────────────────────
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N)
{
    // Thread's position in the 1D grid
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: the last block may have threads beyond array length
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference: serial addition for correctness check
// ─────────────────────────────────────────────────────────────────────────────
void vector_add_cpu(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 1 << 24;          // 16 million elements (~64 MB per array)
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;   // ceiling division

    size_t bytes = N * sizeof(float);

    // ── Allocate host memory ──────────────────────────────────────────────────
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);   // GPU result
    float* h_ref = (float*)malloc(bytes); // CPU reference

    // Initialize with simple values
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    // ── Allocate device memory ────────────────────────────────────────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // ── Copy host → device ────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ── Time the GPU kernel with CUDA events ──────────────────────────────────
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add_kernel<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // ── Copy device → host ────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // ── CPU reference ─────────────────────────────────────────────────────────
    vector_add_cpu(h_A, h_B, h_ref, N);

    // ── Verify correctness ────────────────────────────────────────────────────
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_C[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }

    // ── Report ────────────────────────────────────────────────────────────────
    printf("Vector size: %d elements (%.1f MB per array)\n", N, bytes / 1e6);
    printf("Launch config: %d blocks × %d threads\n", BLOCKS, THREADS);
    printf("GPU time: %.3f ms\n", gpu_ms);
    printf("Effective bandwidth: %.1f GB/s\n",
           3.0 * bytes / (gpu_ms * 1e-3) / 1e9);   // read A, B + write C
    printf("Max error vs CPU: %.2e  %s\n",
           max_err, max_err < 1e-5f ? "(PASS)" : "(FAIL)");

    // ── Cleanup ───────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);

    return 0;
}
