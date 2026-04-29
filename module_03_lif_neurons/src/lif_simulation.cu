/*
 * lif_simulation.cu
 * Module 03 — Leaky Integrate-and-Fire Neuron Simulation
 *
 * Simulates N LIF neurons in parallel on the GPU.
 * Each thread owns one neuron and updates it at each timestep.
 *
 * Model:  τm * dV/dt = -(V - EL) + Rm * I(t)
 * Spike:  if V >= V_th  →  spike, V = V_reset, enter refractory period
 *
 * State vectors (SoA layout for coalesced access):
 *   V[N]      — membrane voltage (mV)
 *   I[N]      — input current (pA)
 *   ref[N]    — refractory counter (steps remaining)
 *   spikes[]  — flat spike log: (neuron_id, timestep) pairs
 *
 * Usage:
 *   nvcc -O2 -o lif_simulation lif_simulation.cu -lm
 *   ./lif_simulation [N] [T_ms] [dt_ms]
 *
 * Outputs:
 *   spikes.txt  — spike log: one line per spike "neuron_id  time_ms"
 *   summary.txt — firing rate statistics
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

// ─────────────────────────────────────────────────────────────────────────────
// Simulation parameters in constant memory
// ─────────────────────────────────────────────────────────────────────────────
__constant__ float c_dt;
__constant__ float c_tau_m;
__constant__ float c_E_L;
__constant__ float c_Rm;
__constant__ float c_V_th;
__constant__ float c_V_reset;
__constant__ int   c_T_ref;    // refractory period in timesteps

// ─────────────────────────────────────────────────────────────────────────────
// Spike buffer: pre-allocated on GPU
// We use atomic operations to safely append from many threads
// ─────────────────────────────────────────────────────────────────────────────
#define MAX_SPIKES_PER_NEURON 500
// Total spike buffer = N * MAX_SPIKES_PER_NEURON (pre-allocated)

// ─────────────────────────────────────────────────────────────────────────────
// LIF kernel: one thread per neuron
// ─────────────────────────────────────────────────────────────────────────────
__global__ void lif_step(
    float* V,           // [N] membrane voltages
    const float* I,     // [N] input currents
    int*  ref_count,    // [N] remaining refractory steps
    int*  spike_count,  // [N] total spikes fired by each neuron
    int   N,
    int   timestep
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // In refractory period: skip voltage update
    if (ref_count[i] > 0) {
        ref_count[i]--;
        return;
    }

    // Euler integration of the LIF equation
    // dV/dt = (-(V - E_L) + Rm * I) / tau_m
    float dV = c_dt / c_tau_m * (-(V[i] - c_E_L) + c_Rm * I[i]);
    V[i] += dV;

    // Spike detection and reset
    if (V[i] >= c_V_th) {
        V[i] = c_V_reset;
        ref_count[i] = c_T_ref;
        spike_count[i]++;   // atomic not needed — each thread owns its own element
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spike recording kernel (optional, run after lif_step when recording)
// Saves (neuron_id, timestep) pairs to a flat spike log using atomicAdd
// ─────────────────────────────────────────────────────────────────────────────
__global__ void record_spikes(
    const float* V,         // current voltages (checked for threshold)
    const int*   ref_count, // if ref_count > 0, neuron just fired
    int*  spike_log_id,     // [MAX_TOTAL_SPIKES] neuron IDs
    float* spike_log_time,  // [MAX_TOTAL_SPIKES] spike times (ms)
    int*  n_spikes,         // global spike counter (atomicAdd)
    int   N,
    int   max_spikes,
    float time_ms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // A neuron fired this step if its refractory counter is exactly T_ref
    if (ref_count[i] == c_T_ref) {
        int idx = atomicAdd(n_spikes, 1);
        if (idx < max_spikes) {
            spike_log_id[idx]   = i;
            spike_log_time[idx] = time_ms;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Input current initialization: heterogeneous bias + Gaussian noise (simple)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void init_currents(float* I, int N, float I_mean, float I_std, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Box-Muller using a simple hash-based random
    unsigned u1 = (i * 1664525u + seed) * 1013904223u;
    unsigned u2 = (i * 22695477u + seed) * 1664525u;
    float z = sqrtf(-2.0f * logf((u1 + 1u) / 4294967296.0f)) *
              cosf(6.283185f * u2 / 4294967296.0f);
    I[i] = I_mean + I_std * z;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    // Simulation parameters
    int   N     = (argc > 1) ? atoi(argv[1]) : 10000;   // neurons
    float T_ms  = (argc > 2) ? atof(argv[2]) : 1000.0f; // total time (ms)
    float dt    = (argc > 3) ? atof(argv[3]) : 0.1f;    // timestep (ms)

    float tau_m   = 20.0f;   // ms
    float E_L     = -65.0f;  // mV
    float Rm      = 10.0f;   // MOhm
    float V_th    = -55.0f;  // mV
    float V_reset = -70.0f;  // mV
    float T_ref_ms= 2.0f;    // refractory period (ms)
    float I_mean  = 1.8f;    // pA — mean injected current (above threshold)
    float I_std   = 0.3f;    // pA — neuron-to-neuron variability

    int T_steps   = (int)(T_ms / dt);
    int T_ref_steps = (int)(T_ref_ms / dt);

    // Set constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_dt,      &dt,          sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_m,   &tau_m,       sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_E_L,     &E_L,         sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Rm,      &Rm,          sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_V_th,    &V_th,        sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_V_reset, &V_reset,     sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_T_ref,   &T_ref_steps, sizeof(int)));

    printf("LIF Simulation\n");
    printf("  Neurons: %d\n", N);
    printf("  Duration: %.0f ms (%d steps, dt=%.2f ms)\n", T_ms, T_steps, dt);
    printf("  I_mean: %.2f pA, I_std: %.2f pA\n\n", I_mean, I_std);

    size_t f_bytes = N * sizeof(float);
    size_t i_bytes = N * sizeof(int);

    // Device state (SoA layout)
    float *d_V, *d_I;
    int   *d_ref, *d_spike_count;
    CUDA_CHECK(cudaMalloc(&d_V,           f_bytes));
    CUDA_CHECK(cudaMalloc(&d_I,           f_bytes));
    CUDA_CHECK(cudaMalloc(&d_ref,         i_bytes));
    CUDA_CHECK(cudaMalloc(&d_spike_count, i_bytes));

    // Spike log
    int max_spikes = N * MAX_SPIKES_PER_NEURON;
    int   *d_spike_id;
    float *d_spike_time;
    int   *d_n_spikes;
    CUDA_CHECK(cudaMalloc(&d_spike_id,   max_spikes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_spike_time, max_spikes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_n_spikes,   sizeof(int)));
    CUDA_CHECK(cudaMemset(d_n_spikes, 0, sizeof(int)));

    // Initialize state
    CUDA_CHECK(cudaMemset(d_ref, 0, i_bytes));
    CUDA_CHECK(cudaMemset(d_spike_count, 0, i_bytes));

    // Initialize V at E_L
    float* h_V = (float*)malloc(f_bytes);
    for (int i = 0; i < N; i++) h_V[i] = E_L;
    CUDA_CHECK(cudaMemcpy(d_V, h_V, f_bytes, cudaMemcpyHostToDevice));

    // Initialize heterogeneous input currents
    int threads = 256, blocks = (N + threads - 1) / threads;
    init_currents<<<blocks, threads>>>(d_I, N, I_mean, I_std, 12345u);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Simulation loop ───────────────────────────────────────────────────────
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (int step = 0; step < T_steps; step++) {
        float t_ms = step * dt;

        // Update all neurons in parallel
        lif_step<<<blocks, threads>>>(d_V, d_I, d_ref, d_spike_count, N, step);

        // Record spikes (every step for spike log)
        record_spikes<<<blocks, threads>>>(
            d_V, d_ref,
            d_spike_id, d_spike_time, d_n_spikes,
            N, max_spikes, t_ms
        );
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float sim_ms;
    CUDA_CHECK(cudaEventElapsedTime(&sim_ms, t0, t1));

    // ── Copy results back ─────────────────────────────────────────────────────
    int* h_spike_count = (int*)malloc(i_bytes);
    CUDA_CHECK(cudaMemcpy(h_spike_count, d_spike_count, i_bytes, cudaMemcpyDeviceToHost));

    int h_n_spikes;
    CUDA_CHECK(cudaMemcpy(&h_n_spikes, d_n_spikes, sizeof(int), cudaMemcpyDeviceToHost));
    h_n_spikes = (h_n_spikes < max_spikes) ? h_n_spikes : max_spikes;

    int*   h_spike_id   = (int*)malloc(h_n_spikes * sizeof(int));
    float* h_spike_time = (float*)malloc(h_n_spikes * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_spike_id,   d_spike_id,   h_n_spikes * sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_spike_time, d_spike_time, h_n_spikes * sizeof(float), cudaMemcpyDeviceToHost));

    // ── Statistics ────────────────────────────────────────────────────────────
    long total_spikes = 0;
    float max_fr = 0, min_fr = 1e9;
    for (int i = 0; i < N; i++) {
        total_spikes += h_spike_count[i];
        float fr = h_spike_count[i] / (T_ms / 1000.0f);  // Hz
        if (fr > max_fr) max_fr = fr;
        if (fr < min_fr) min_fr = fr;
    }
    float mean_fr = total_spikes / (float)N / (T_ms / 1000.0f);

    printf("=== Results ===\n");
    printf("  Total spikes:     %ld\n", total_spikes);
    printf("  Mean firing rate: %.1f Hz\n", mean_fr);
    printf("  Min/Max rate:     %.1f / %.1f Hz\n", min_fr, max_fr);
    printf("  GPU sim time:     %.2f ms\n", sim_ms);
    printf("  Speedup factor:   %.1fx real-time\n", T_ms / sim_ms);
    printf("  Throughput:       %.1f M neuron-steps/s\n",
           (float)N * T_steps / sim_ms / 1000.0f);

    // ── Write spike log ───────────────────────────────────────────────────────
    FILE* f = fopen("spikes.txt", "w");
    fprintf(f, "# neuron_id  time_ms\n");
    for (int k = 0; k < h_n_spikes; k++) {
        fprintf(f, "%d  %.2f\n", h_spike_id[k], h_spike_time[k]);
    }
    fclose(f);
    printf("\n  Spike log written to spikes.txt (%d spikes)\n", h_n_spikes);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    cudaFree(d_V); cudaFree(d_I); cudaFree(d_ref); cudaFree(d_spike_count);
    cudaFree(d_spike_id); cudaFree(d_spike_time); cudaFree(d_n_spikes);
    free(h_V); free(h_spike_count); free(h_spike_id); free(h_spike_time);
    return 0;
}
