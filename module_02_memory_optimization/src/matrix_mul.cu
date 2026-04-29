/*
 * matrix_mul.cu
 * Module 02 — Memory Hierarchy & Optimization
 *
 * Three implementations of square matrix multiply C = A * B:
 *   1. Naive GPU  — one thread per output element, global memory only
 *   2. Tiled GPU  — shared memory tiling for cache reuse
 *   3. CPU        — serial reference
 *
 * Key insight: naive reads each row of A N times (N²×N global reads total).
 * Tiling loads tiles into shared memory once and reuses them TILE_SIZE times.
 *
 * Compile:  nvcc -O2 -o matrix_mul matrix_mul.cu
 * Run:      ./matrix_mul
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                               \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }      \
} while(0)

#define TILE_SIZE 16    // 16×16 thread block = 256 threads; tile fits in shared mem

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Naive (global memory only)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void matmul_naive(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        // Each iteration: 2 global reads (A[row,k] and B[k,col]), no reuse
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Tiled (shared memory)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N)
{
    // Shared memory tiles: each block loads TILE_SIZE×TILE_SIZE elements
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Slide a tile across the shared dimension k
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {

        // Collaboratively load a tile of A and B into shared memory
        int kA = t * TILE_SIZE + threadIdx.x;   // column of A in tile
        int kB = t * TILE_SIZE + threadIdx.y;   // row of B in tile

        sA[threadIdx.y][threadIdx.x] = (row < N && kA < N) ? A[row * N + kA] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (kB < N && col < N) ? B[kB * N + col] : 0.0f;

        __syncthreads();  // wait until all threads have loaded their element

        // Compute partial dot product using shared memory (fast!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();  // wait before loading the next tile
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU reference
// ─────────────────────────────────────────────────────────────────────────────
void matmul_cpu(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) s += A[i*N+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

float max_abs_error(const float* A, const float* B, int n) {
    float e = 0.0f;
    for (int i = 0; i < n; i++) { float d = fabsf(A[i]-B[i]); if (d>e) e=d; }
    return e;
}

int main()
{
    // Test with N=512 (512×512 matrix, fits comfortably in GPU memory)
    int N = 512;
    size_t bytes = (size_t)N * N * sizeof(float);

    float* h_A   = (float*)malloc(bytes);
    float* h_B   = (float*)malloc(bytes);
    float* h_C   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);

    // Initialize
    srand(42);
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // CPU reference
    matmul_cpu(h_A, h_B, h_ref, N);

    // Device
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    // ── Naive ─────────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(t0));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_naive; CUDA_CHECK(cudaEventElapsedTime(&ms_naive, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    float err_naive = max_abs_error(h_C, h_ref, N*N);

    double flops = 2.0 * N * N * N;   // N³ mults + N³ adds
    printf("Matrix size: %dx%d\\n\\n", N, N);
    printf("%-12s  time=%7.2f ms  GFLOPS=%6.1f  max_err=%.2e  %s\\n",
           "Naive GPU", ms_naive, flops/(ms_naive*1e-3)/1e9,
           err_naive, err_naive<1e-3f?"PASS":"FAIL");

    // ── Tiled ─────────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(t0));
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms_tiled; CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    float err_tiled = max_abs_error(h_C, h_ref, N*N);

    printf("%-12s  time=%7.2f ms  GFLOPS=%6.1f  max_err=%.2e  %s\\n",
           "Tiled GPU", ms_tiled, flops/(ms_tiled*1e-3)/1e9,
           err_tiled, err_tiled<1e-3f?"PASS":"FAIL");

    printf("\\nSpeedup (tiled vs naive): %.1fx\\n", ms_naive / ms_tiled);

    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
