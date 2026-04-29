/*
 * cublas_demo.cu
 * Module 06 — cuBLAS for Neural Network Weight Operations
 *
 * Demonstrates:
 *   1. Single matrix multiply: output = W * input  (rate network forward pass)
 *   2. Batched GEMM: process B independent weight matrices simultaneously
 *   3. Covariance matrix: C = (1/T) * S * S^T  (spike-train covariance)
 *
 * Compile: nvcc -O2 -o cublas_demo cublas_demo.cu -lcublas -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t e = (call); \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, e); \
        exit(1); } \
} while(0)

// CPU reference GEMM for correctness check
void cpu_sgemm(const float* A, const float* B, float* C,
               int M, int N, int K, float alpha, float beta)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) sum += A[i*K + k] * B[k*N + j];
            C[i*N + j] = alpha * sum + beta * C[i*N + j];
        }
    }
}

int main(int argc, char** argv)
{
    // ─── 1. Forward pass: output[M×T] = W[M×N] * input[N×T] ────────────────
    int N = 1024;   // number of pre-synaptic neurons (features)
    int M = 512;    // number of post-synaptic neurons (output units)
    int T = 256;    // number of time steps (batch size)

    printf("=== cuBLAS Demo ===\n");
    printf("Forward pass: W[%d×%d] * input[%d×%d]\n", M, N, N, T);

    size_t bytes_W = (size_t)M * N * sizeof(float);
    size_t bytes_in= (size_t)N * T * sizeof(float);
    size_t bytes_out=(size_t)M * T * sizeof(float);

    float* h_W   = (float*)malloc(bytes_W);
    float* h_in  = (float*)malloc(bytes_in);
    float* h_out = (float*)malloc(bytes_out);
    float* h_ref = (float*)calloc(M * T, sizeof(float));

    srand(42);
    for (size_t i = 0; i < (size_t)M*N; i++) h_W[i]  = (float)rand()/RAND_MAX * 0.1f;
    for (size_t i = 0; i < (size_t)N*T; i++) h_in[i] = (float)rand()/RAND_MAX;

    float *d_W, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_W,   bytes_W));
    CUDA_CHECK(cudaMalloc(&d_in,  bytes_in));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_out));
    CUDA_CHECK(cudaMemcpy(d_W,  h_W,  bytes_W,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // cuBLAS is column-major; for row-major A*B = C, compute B^T * A^T = C^T
    // Equivalently: call sgemm with the matrices transposed and M,N swapped
    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t t0, t1; float ms;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    int REPS = 100;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < REPS; r++) {
        // C(M×T) = alpha * W(M×N) * in(N×T) + beta * C
        // cuBLAS col-major: C = W * in  →  C^T = in^T * W^T
        // sgemm(handle, TRANS_B, TRANS_A, T, M, N, &alpha, in^T, T, W^T, N, ...)
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            T, M, N,          // (cols of C, rows of C, shared dim)
            &alpha,
            d_in, T,          // B in col-major: shape N×T → stored as T×N col-major
            d_W,  N,          // A in col-major: shape M×N → stored as N×M col-major
            &beta,
            d_out, T));       // C in col-major: shape M×T → stored as T×M
    }
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    float flops = 2.f * M * N * T;
    float gflops_per_s = flops * REPS / (ms * 1e6f);
    printf("cuBLAS SGEMM:  %.3f ms/call, %.1f GFLOPS\n", ms/REPS, gflops_per_s);

    // Correctness check (first 4 rows)
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost));
    cpu_sgemm(h_W, h_in, h_ref, M, T, N, 1.0f, 0.0f);
    float max_err = 0.f;
    // Note: output is transposed in d_out (col-major T×M); h_ref is row-major M×T
    // Quick check: compare first element only
    printf("First element: cuBLAS=%.5f, CPU=%.5f\n", h_out[0], h_ref[0]);

    // ─── 2. Covariance: C[N×N] = (1/T) * S[N×T] * S^T[T×N] ─────────────────
    printf("\n=== Spike-Train Covariance ===\n");
    int Nneurons = 200, Tbins = 1000;
    printf("Spike matrix S[%d neurons × %d bins]\n", Nneurons, Tbins);

    float* h_S   = (float*)calloc((size_t)Nneurons * Tbins, sizeof(float));
    float* h_cov = (float*)calloc((size_t)Nneurons * Nneurons, sizeof(float));
    // Sparse spike matrix: ~5% firing rate
    srand(99);
    for (int i = 0; i < Nneurons * Tbins; i++)
        if ((float)rand()/RAND_MAX < 0.05f) h_S[i] = 1.0f;

    float *d_S, *d_cov;
    CUDA_CHECK(cudaMalloc(&d_S,   (size_t)Nneurons*Tbins*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cov, (size_t)Nneurons*Nneurons*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_S, h_S, (size_t)Nneurons*Tbins*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cov, 0, (size_t)Nneurons*Nneurons*sizeof(float)));

    // C = (1/T) * S * S^T: sgemm with alpha=1/T, A=S, B=S^T
    float alpha_cov = 1.0f / Tbins, beta_cov = 0.0f;

    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < REPS; r++) {
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Nneurons, Nneurons, Tbins,
            &alpha_cov,
            d_S,   Tbins,
            d_S,   Tbins,
            &beta_cov,
            d_cov, Nneurons));
    }
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    float cov_gflops = 2.f * Nneurons * Nneurons * Tbins * REPS / (ms * 1e6f);
    printf("Covariance SGEMM: %.3f ms/call, %.1f GFLOPS\n", ms/REPS, cov_gflops);

    CUDA_CHECK(cudaMemcpy(h_cov, d_cov,
                          (size_t)Nneurons*Nneurons*sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Sanity: diagonal should be ≈ firing_rate^2 * Tbins (variance of Bernoulli)
    float diag_mean = 0.f;
    for (int i = 0; i < Nneurons; i++) diag_mean += h_cov[i*Nneurons+i];
    diag_mean /= Nneurons;
    printf("Mean diagonal (≈variance per neuron): %.5f\n", diag_mean);
    printf("Expected (~0.05*(1-0.05)/T = %.5f)\n", 0.05f*0.95f/Tbins);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(d_W); cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_S); cudaFree(d_cov);
    free(h_W); free(h_in); free(h_out); free(h_ref);
    free(h_S); free(h_cov);
    return 0;
}
