/*
 * neural_field.cu
 * Project 02 — Wilson-Cowan Neural Field Theory on GPU
 *
 * Uses cuFFT for convolution with a Mexican-hat connectivity kernel.
 * Thread (ix, iy) owns one cortical pixel.
 *
 * Compile:  nvcc -O2 -o neural_field neural_field.cu -lcufft -lm
 * Run:      ./neural_field [grid_size] [T_ms]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(call) do {                                               \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }      \
} while(0)

#define CUFFT_CHECK(call) do {                                              \
    cufftResult e = (call);                                                 \
    if (e != CUFFT_SUCCESS) {                                               \
        fprintf(stderr, "cuFFT error at %s:%d: %d\n",                      \
                __FILE__, __LINE__, e); exit(1); }                          \
} while(0)

__constant__ float c_dt, c_tau_E, c_tau_I;
__constant__ float c_I_E, c_I_I;          // external inputs
__constant__ int   c_Nx, c_Ny;            // grid dimensions

// Sigmoid activation function
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ─── Pointwise multiply spectra: A = A * B (complex × complex) ───────────────
__global__ void multiply_spectra(cufftComplex* A, const cufftComplex* B,
                                  int n_elem) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_elem) return;
    float ar = A[k].x, ai = A[k].y;
    float br = B[k].x, bi = B[k].y;
    A[k].x = ar*br - ai*bi;
    A[k].y = ar*bi + ai*br;
}

// ─── Scale after IFFT (cuFFT does not normalize) ─────────────────────────────
__global__ void scale(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// ─── Wilson-Cowan Euler update ────────────────────────────────────────────────
// u[i], v[i]: excitatory and inhibitory activities
// conv_EE[i]: convolution of w_EE kernel with u (i.e., excitatory input)
// conv_IE[i]: convolution of w_IE kernel with v (inhibitory input)
__global__ void wilson_cowan_step(
    float* u, float* v,
    const float* conv_EE,   // w_EE * u
    const float* conv_IE,   // w_IE * u  (excitatory drive to I)
    const float* noise,     // spatial noise for I_E
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Excitatory population input
    float input_E = conv_EE[i] - 0.5f * v[i] + c_I_E + noise[i];
    // Inhibitory population input
    float input_I = conv_IE[i] - 0.5f * v[i] + c_I_I;

    float du = c_dt / c_tau_E * (-u[i] + sigmoid(input_E));
    float dv = c_dt / c_tau_I * (-v[i] + sigmoid(input_I));

    u[i] += du;
    v[i] += dv;

    // Clip to [0,1]
    u[i] = fminf(fmaxf(u[i], 0.f), 1.f);
    v[i] = fminf(fmaxf(v[i], 0.f), 1.f);
}

// ─── Build Mexican-hat kernel on CPU ─────────────────────────────────────────
// w(r) = A * exp(-r²/2σ_E²) - B * exp(-r²/2σ_I²)
// Stored as 2D grid, then FFT'd once at startup
void build_mexican_hat(float* kernel, int Nx, int Ny,
                       float A, float sigma_E, float B, float sigma_I)
{
    for (int iy = 0; iy < Ny; iy++) {
        for (int ix = 0; ix < Nx; ix++) {
            // Circular distance (for periodic boundary)
            float dx = (float)(ix <= Nx/2 ? ix : ix - Nx);
            float dy = (float)(iy <= Ny/2 ? iy : iy - Ny);
            float r2 = dx*dx + dy*dy;
            kernel[iy * Nx + ix] =
                A * expf(-r2 / (2.f * sigma_E * sigma_E)) -
                B * expf(-r2 / (2.f * sigma_I * sigma_I));
        }
    }
}

// ─── Simple LCG noise generator on GPU ───────────────────────────────────────
__global__ void gen_noise(float* noise, unsigned int seed, float amplitude, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned int s = seed * 1664525u + (unsigned int)i * 1013904223u;
    noise[i] = amplitude * ((float)(s & 0xFFFFFF) / 0xFFFFFF - 0.5f);
}

int main(int argc, char** argv)
{
    int   Nx   = (argc > 1) ? atoi(argv[1]) : 256;
    int   Ny   = Nx;
    float T_ms = (argc > 2) ? atof(argv[2]) : 500.f;
    float dt   = 0.1f;
    int   T    = (int)(T_ms / dt);
    int   N    = Nx * Ny;
    int   n_freq = Nx * (Ny / 2 + 1);

    printf("Neural Field: %d×%d grid, T=%.0f ms\n", Nx, Ny, T_ms);

    // Parameters
    float tau_E=10.f, tau_I=20.f;
    float I_E=0.5f, I_I=0.5f;
    float A_EE=4.f, sigma_EE=2.f, B_EE=2.f, sigma_IE=8.f;

    CUDA_CHECK(cudaMemcpyToSymbol(c_dt,    &dt,    sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_E, &tau_E, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_I, &tau_I, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_I_E,   &I_E,   sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_I_I,   &I_I,   sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Nx,    &Nx,    sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Ny,    &Ny,    sizeof(int)));

    // Build kernel and FFT it
    float* h_kernel = (float*)calloc(N, sizeof(float));
    build_mexican_hat(h_kernel, Nx, Ny, A_EE, sigma_EE, B_EE, sigma_IE);

    // Normalize kernel
    float ksum = 0.f;
    for (int i = 0; i < N; i++) ksum += fabsf(h_kernel[i]);
    for (int i = 0; i < N; i++) h_kernel[i] /= ksum;

    float *d_kernel_space;
    cufftComplex *d_kernel_freq;
    CUDA_CHECK(cudaMalloc(&d_kernel_space, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel_freq,  n_freq * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemcpy(d_kernel_space, h_kernel, N*sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_kernel);

    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftPlan2d(&plan_r2c, Nx, Ny, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan2d(&plan_c2r, Nx, Ny, CUFFT_C2R));

    // Precompute kernel FFT
    CUFFT_CHECK(cufftExecR2C(plan_r2c, d_kernel_space, d_kernel_freq));

    // Simulation state
    float *d_u, *d_v, *d_conv, *d_noise;
    cufftComplex *d_u_freq;
    CUDA_CHECK(cudaMalloc(&d_u,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_noise,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_freq, n_freq * sizeof(cufftComplex)));

    // Init: small random perturbation around u=0.1, v=0.05
    float* h_u = (float*)malloc(N * sizeof(float));
    float* h_v = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) {
        h_u[i] = 0.1f + 0.05f * ((float)rand()/RAND_MAX - 0.5f);
        h_v[i] = 0.05f + 0.02f * ((float)rand()/RAND_MAX - 0.5f);
    }
    CUDA_CHECK(cudaMemcpy(d_u, h_u, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, N*sizeof(float), cudaMemcpyHostToDevice));
    free(h_u); free(h_v);

    int thr = 256;
    int blk = (N + thr - 1) / thr;
    int blk_freq = (n_freq + thr - 1) / thr;
    float inv_N = 1.0f / N;

    // Snapshot times (ms) → steps
    int snap_steps[] = {0, (int)(100/dt), (int)(250/dt), (int)(500/dt)};
    int n_snap = 4;
    int snap_idx = 0;

    float* h_snap = (float*)malloc(N * sizeof(float));
    FILE* fout = fopen("neural_field_snapshots.txt", "w");

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    for (int step = 0; step < T; step++) {
        float t_ms = step * dt;

        // Spatial noise (refresh every 10 steps)
        if (step % 10 == 0)
            gen_noise<<<blk, thr>>>(d_noise, (unsigned int)step, 0.05f, N);

        // Convolve u with Mexican-hat kernel via FFT
        // 1. FFT(u)
        CUFFT_CHECK(cufftExecR2C(plan_r2c, d_u, d_u_freq));
        // 2. Multiply spectrum
        multiply_spectra<<<blk_freq, thr>>>(d_u_freq, d_kernel_freq, n_freq);
        // 3. IFFT
        CUFFT_CHECK(cufftExecC2R(plan_c2r, d_u_freq, d_conv));
        // 4. Normalize
        scale<<<blk, thr>>>(d_conv, inv_N, N);

        // Wilson-Cowan update
        wilson_cowan_step<<<blk, thr>>>(d_u, d_v, d_conv, d_conv, d_noise, N);

        // Snapshots
        if (snap_idx < n_snap && step == snap_steps[snap_idx]) {
            CUDA_CHECK(cudaMemcpy(h_snap, d_u, N*sizeof(float), cudaMemcpyDeviceToHost));
            fprintf(fout, "# t=%.0f ms\n", t_ms);
            for (int iy = 0; iy < Ny; iy++) {
                for (int ix = 0; ix < Nx; ix++)
                    fprintf(fout, "%.4f ", h_snap[iy*Nx + ix]);
                fprintf(fout, "\n");
            }
            snap_idx++;
        }
    }

    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    float sim_ms;
    CUDA_CHECK(cudaEventElapsedTime(&sim_ms, ev0, ev1));
    printf("GPU time: %.1f ms (%.1fx real-time)\n", sim_ms, T_ms/sim_ms);
    printf("Throughput: %.0f M pixel-steps/s\n",
           (float)T * N / (sim_ms * 1e3f));

    fclose(fout);
    CUDA_CHECK(cudaEventDestroy(ev0)); CUDA_CHECK(cudaEventDestroy(ev1));
    CUFFT_CHECK(cufftDestroy(plan_r2c)); CUFFT_CHECK(cufftDestroy(plan_c2r));
    cudaFree(d_kernel_space); cudaFree(d_kernel_freq);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_conv);
    cudaFree(d_noise); cudaFree(d_u_freq);
    free(h_snap);
    return 0;
}
