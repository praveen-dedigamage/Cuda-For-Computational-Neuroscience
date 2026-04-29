/*
 * hh_simulation.cu
 * Module 04 — Hodgkin-Huxley Neuron Model on GPU
 *
 * Implements the full HH model for N neurons in parallel.
 * State per neuron: (V, m, h, n)  — 4 floats in SoA layout
 *
 * ODE system:
 *   Cm * dV/dt = -gNa*m³h*(V-ENa) - gK*n⁴*(V-EK) - gL*(V-EL) + I
 *   dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m
 *   dh/dt = alpha_h(V)*(1-h) - beta_h(V)*h
 *   dn/dt = alpha_n(V)*(1-n) - beta_n(V)*n
 *
 * Two solvers available: Euler (fast, less accurate) and RK4 (accurate).
 *
 * Usage:
 *   nvcc -O2 -o hh_simulation hh_simulation.cu -lm
 *   ./hh_simulation [N] [T_ms] [solver: euler|rk4]
 *
 * Outputs:
 *   hh_voltage.txt   — V(t) for first 10 neurons, all timesteps
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                               \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }      \
} while(0)

// HH parameters in constant memory
__constant__ float c_Cm;     // membrane capacitance (μF/cm²)
__constant__ float c_gNa;    // max Na conductance
__constant__ float c_gK;     // max K conductance
__constant__ float c_gL;     // leak conductance
__constant__ float c_ENa;    // Na reversal potential (mV)
__constant__ float c_EK;     // K reversal potential
__constant__ float c_EL;     // leak reversal potential
__constant__ float c_dt;     // timestep (ms)

// ─────────────────────────────────────────────────────────────────────────────
// Voltage-dependent rate constants — device functions
// These are called millions of times per second, so they must be fast
// ─────────────────────────────────────────────────────────────────────────────

// Sodium activation (m) rate constants
__device__ float alpha_m(float V) {
    float dv = V + 40.0f;
    if (fabsf(dv) < 1e-5f) return 1.0f;  // L'Hopital limit
    return 0.1f * dv / (1.0f - expf(-dv / 10.0f));
}
__device__ float beta_m(float V) {
    return 4.0f * expf(-(V + 65.0f) / 18.0f);
}

// Sodium inactivation (h)
__device__ float alpha_h(float V) {
    return 0.07f * expf(-(V + 65.0f) / 20.0f);
}
__device__ float beta_h(float V) {
    return 1.0f / (1.0f + expf(-(V + 35.0f) / 10.0f));
}

// Potassium activation (n)
__device__ float alpha_n(float V) {
    float dv = V + 55.0f;
    if (fabsf(dv) < 1e-5f) return 0.1f;
    return 0.01f * dv / (1.0f - expf(-dv / 10.0f));
}
__device__ float beta_n(float V) {
    return 0.125f * expf(-(V + 65.0f) / 80.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute derivatives for the full HH system
// ─────────────────────────────────────────────────────────────────────────────
__device__ void hh_derivatives(
    float V, float m, float h, float n, float I_ext,
    float* dV, float* dm, float* dh, float* dn
) {
    float I_Na = c_gNa * m * m * m * h * (V - c_ENa);
    float I_K  = c_gK  * n * n * n * n * (V - c_EK);
    float I_L  = c_gL  * (V - c_EL);

    *dV = (I_ext - I_Na - I_K - I_L) / c_Cm;
    *dm = alpha_m(V) * (1.0f - m) - beta_m(V) * m;
    *dh = alpha_h(V) * (1.0f - h) - beta_h(V) * h;
    *dn = alpha_n(V) * (1.0f - n) - beta_n(V) * n;
}

// ─────────────────────────────────────────────────────────────────────────────
// Euler solver kernel
// ─────────────────────────────────────────────────────────────────────────────
__global__ void hh_euler_step(
    float* V, float* m, float* h, float* n,
    const float* I_ext, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float dV, dm, dh, dn;
    hh_derivatives(V[i], m[i], h[i], n[i], I_ext[i], &dV, &dm, &dh, &dn);

    V[i] += c_dt * dV;
    m[i] += c_dt * dm;
    h[i] += c_dt * dh;
    n[i] += c_dt * dn;

    // Clamp gating variables to [0,1]
    m[i] = fmaxf(0.0f, fminf(1.0f, m[i]));
    h[i] = fmaxf(0.0f, fminf(1.0f, h[i]));
    n[i] = fmaxf(0.0f, fminf(1.0f, n[i]));
}

// ─────────────────────────────────────────────────────────────────────────────
// RK4 solver kernel (more accurate, ~4× more compute per step)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void hh_rk4_step(
    float* V, float* m, float* h, float* n,
    const float* I_ext, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float Vi = V[i], mi = m[i], hi = h[i], ni = n[i];
    float I  = I_ext[i];
    float dt = c_dt;

    // k1
    float k1V, k1m, k1h, k1n;
    hh_derivatives(Vi, mi, hi, ni, I, &k1V, &k1m, &k1h, &k1n);

    // k2
    float k2V, k2m, k2h, k2n;
    hh_derivatives(Vi + 0.5f*dt*k1V, mi + 0.5f*dt*k1m,
                   hi + 0.5f*dt*k1h, ni + 0.5f*dt*k1n, I,
                   &k2V, &k2m, &k2h, &k2n);

    // k3
    float k3V, k3m, k3h, k3n;
    hh_derivatives(Vi + 0.5f*dt*k2V, mi + 0.5f*dt*k2m,
                   hi + 0.5f*dt*k2h, ni + 0.5f*dt*k2n, I,
                   &k3V, &k3m, &k3h, &k3n);

    // k4
    float k4V, k4m, k4h, k4n;
    hh_derivatives(Vi + dt*k3V, mi + dt*k3m,
                   hi + dt*k3h, ni + dt*k3n, I,
                   &k4V, &k4m, &k4h, &k4n);

    // Combine
    V[i] = Vi + dt / 6.0f * (k1V + 2.0f*k2V + 2.0f*k3V + k4V);
    m[i] = mi + dt / 6.0f * (k1m + 2.0f*k2m + 2.0f*k3m + k4m);
    h[i] = hi + dt / 6.0f * (k1h + 2.0f*k2h + 2.0f*k3h + k4h);
    n[i] = ni + dt / 6.0f * (k1n + 2.0f*k2n + 2.0f*k3n + k4n);

    m[i] = fmaxf(0.0f, fminf(1.0f, m[i]));
    h[i] = fmaxf(0.0f, fminf(1.0f, h[i]));
    n[i] = fmaxf(0.0f, fminf(1.0f, n[i]));
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialization: set to steady state at resting potential
// ─────────────────────────────────────────────────────────────────────────────
__global__ void init_hh(float* V, float* m, float* h, float* n,
                         float* I_ext, int N, float V_rest, float I_mean, float I_range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    V[i] = V_rest;
    // Steady-state gating at V_rest
    float am = alpha_m(V_rest), bm = beta_m(V_rest);
    float ah = alpha_h(V_rest), bh = beta_h(V_rest);
    float an = alpha_n(V_rest), bn = beta_n(V_rest);
    m[i] = am / (am + bm);
    h[i] = ah / (ah + bh);
    n[i] = an / (an + bn);

    // Parameter sweep: vary I_ext linearly across neurons
    I_ext[i] = I_mean + I_range * ((float)i / (float)(N - 1) - 0.5f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    int   N      = (argc > 1) ? atoi(argv[1]) : 1000;
    float T_ms   = (argc > 2) ? atof(argv[2]) : 100.0f;
    int   use_rk4= (argc > 3 && strcmp(argv[3], "rk4") == 0) ? 1 : 0;
    float dt     = 0.01f;   // HH requires smaller timestep than LIF
    int   T_steps= (int)(T_ms / dt);

    // HH standard parameters (Hodgkin & Huxley, 1952)
    float Cm = 1.0f, gNa = 120.0f, gK = 36.0f, gL = 0.3f;
    float ENa = 50.0f, EK = -77.0f, EL = -54.4f;

    CUDA_CHECK(cudaMemcpyToSymbol(c_Cm,  &Cm,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gNa, &gNa, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gK,  &gK,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gL,  &gL,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ENa, &ENa, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_EK,  &EK,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_EL,  &EL,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_dt,  &dt,  sizeof(float)));

    printf("Hodgkin-Huxley Simulation\n");
    printf("  Neurons: %d\n", N);
    printf("  Duration: %.1f ms (%d steps, dt=%.3f ms)\n", T_ms, T_steps, dt);
    printf("  Solver: %s\n\n", use_rk4 ? "RK4" : "Euler");

    size_t fb = N * sizeof(float);
    float *d_V, *d_m, *d_h, *d_n, *d_I;
    CUDA_CHECK(cudaMalloc(&d_V, fb)); CUDA_CHECK(cudaMalloc(&d_m, fb));
    CUDA_CHECK(cudaMalloc(&d_h, fb)); CUDA_CHECK(cudaMalloc(&d_n, fb));
    CUDA_CHECK(cudaMalloc(&d_I, fb));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // Initialize: I_ext swept from 5 to 15 μA/cm²
    init_hh<<<blocks, threads>>>(d_V, d_m, d_h, d_n, d_I, N, -65.0f, 10.0f, 10.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate host buffer to record voltage traces (first 10 neurons)
    int n_record = (N > 10) ? 10 : N;
    float* h_V_trace = (float*)malloc((size_t)n_record * T_steps * sizeof(float));
    float* h_V = (float*)malloc(fb);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (int step = 0; step < T_steps; step++) {
        if (use_rk4) {
            hh_rk4_step<<<blocks, threads>>>(d_V, d_m, d_h, d_n, d_I, N);
        } else {
            hh_euler_step<<<blocks, threads>>>(d_V, d_m, d_h, d_n, d_I, N);
        }

        // Record every 10 steps to reduce output size
        if (step % 10 == 0 && h_V_trace) {
            CUDA_CHECK(cudaMemcpy(h_V, d_V, fb, cudaMemcpyDeviceToHost));
            int rec_step = step / 10;
            if (rec_step < T_steps / 10) {
                for (int j = 0; j < n_record; j++) {
                    h_V_trace[j * (T_steps/10) + rec_step] = h_V[j];
                }
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    float sim_ms;
    CUDA_CHECK(cudaEventElapsedTime(&sim_ms, t0, t1));

    printf("=== Results ===\n");
    printf("  GPU time: %.2f ms\n", sim_ms);
    printf("  Throughput: %.2f M neuron-steps/s\n",
           (float)N * T_steps / sim_ms / 1000.0f);
    printf("  Real-time factor: %.1fx\n", T_ms / sim_ms);

    // Write voltage traces
    FILE* f = fopen("hh_voltage.txt", "w");
    fprintf(f, "# time_ms");
    for (int j = 0; j < n_record; j++) fprintf(f, " neuron_%d", j);
    fprintf(f, "\n");

    int n_rec_steps = T_steps / 10;
    for (int s = 0; s < n_rec_steps; s++) {
        fprintf(f, "%.3f", s * dt * 10);
        for (int j = 0; j < n_record; j++) {
            fprintf(f, " %.4f", h_V_trace[j * n_rec_steps + s]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("\n  Voltage traces written to hh_voltage.txt\n");

    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    cudaFree(d_V); cudaFree(d_m); cudaFree(d_h); cudaFree(d_n); cudaFree(d_I);
    free(h_V); free(h_V_trace);
    return 0;
}
