/*
 * network_sim.cu
 * Module 05 — Recurrent LIF Network with Synaptic Conductances
 *
 * Architecture:
 *   N neurons (80% E, 20% I) with random sparse connectivity (p=0.1)
 *   Synaptic model: exponential conductance
 *   g_syn decays with tau_s; incremented by w when pre-neuron fires
 *   I_syn = g_syn * (V - E_syn)
 *
 * GPU strategy:
 *   - Thread i updates neuron i (voltage + conductance)
 *   - Spike propagation: for each spike in this step,
 *     iterate over its outgoing connections and increment g_syn
 *     (done serially in a separate kernel for simplicity)
 *   - For large N, use sorted CSR + parallel scatter instead
 *
 * Compile:  nvcc -O2 -o network_sim network_sim.cu -lm
 * Run:      ./network_sim [N] [T_ms]
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

__constant__ float c_dt, c_tau_m, c_E_L, c_Rm, c_V_th, c_V_reset;
__constant__ int   c_T_ref;
__constant__ float c_tau_E, c_tau_I;    // synaptic time constants
__constant__ float c_E_E, c_E_I;        // synaptic reversal potentials
__constant__ float c_w_E, c_w_I;        // default synaptic weights

// ─────────────────────────────────────────────────────────────────────────────
// LIF + conductance-based synaptic input
// ─────────────────────────────────────────────────────────────────────────────
__global__ void update_neurons(
    float* V,         // [N] voltages
    float* g_E,       // [N] excitatory conductances
    float* g_I,       // [N] inhibitory conductances
    int*   ref,       // [N] refractory counters
    const float* I_ext, // [N] external input
    int*   fired,     // [N] 1 if this neuron fired this step, 0 otherwise
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Decay synaptic conductances
    g_E[i] *= expf(-c_dt / c_tau_E);
    g_I[i] *= expf(-c_dt / c_tau_I);

    fired[i] = 0;

    if (ref[i] > 0) {
        ref[i]--;
        V[i] = c_V_reset;
        return;
    }

    // Conductance-based synaptic currents
    float I_E_syn = g_E[i] * (V[i] - c_E_E);
    float I_I_syn = g_I[i] * (V[i] - c_E_I);
    float I_total = I_ext[i] - I_E_syn - I_I_syn;

    float dV = c_dt / c_tau_m * (-(V[i] - c_E_L) + c_Rm * I_total);
    V[i] += dV;

    if (V[i] >= c_V_th) {
        V[i] = c_V_reset;
        ref[i] = c_T_ref;
        fired[i] = 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Propagate spikes through synaptic connections
// CSR (Compressed Sparse Row) format for the connectivity matrix:
//   row_ptr[i]..row_ptr[i+1]-1 index into col_idx and weights
//   col_idx[k] = post-synaptic neuron j
//   weights[k] = synaptic weight w_ij
// ─────────────────────────────────────────────────────────────────────────────
__global__ void propagate_spikes(
    const int*   fired,     // [N] which neurons fired
    const int*   row_ptr,   // [N+1] CSR row pointers
    const int*   col_idx,   // [n_syn] post-synaptic neuron IDs
    const float* weights,   // [n_syn] synaptic weights
    const int*   syn_type,  // [n_syn] 0=E, 1=I
    float* g_E,             // [N] excitatory conductances
    float* g_I,             // [N] inhibitory conductances
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !fired[i]) return;

    // Neuron i fired: iterate over its outgoing synapses
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
        int   j = col_idx[k];
        float w = weights[k];
        if (syn_type[k] == 0) {
            atomicAdd(&g_E[j], w);   // excitatory synapse
        } else {
            atomicAdd(&g_I[j], w);   // inhibitory synapse
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Build random sparse connectivity on CPU, copy to GPU
// p_connect = connection probability
// ─────────────────────────────────────────────────────────────────────────────
void build_connectivity(
    int N, int N_E, float p_connect, float w_E, float w_I,
    int** h_row_ptr, int** h_col_idx, float** h_weights, int** h_syn_type,
    int* n_syn_out
) {
    // First pass: count connections per neuron
    int* counts = (int*)calloc(N + 1, sizeof(int));
    srand(42);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j && (float)rand() / RAND_MAX < p_connect) counts[i]++;
        }
    }

    // CSR row pointers
    *h_row_ptr = (int*)malloc((N + 1) * sizeof(int));
    (*h_row_ptr)[0] = 0;
    for (int i = 0; i < N; i++) (*h_row_ptr)[i+1] = (*h_row_ptr)[i] + counts[i];
    int n_syn = (*h_row_ptr)[N];
    *n_syn_out = n_syn;

    *h_col_idx  = (int*)malloc(n_syn * sizeof(int));
    *h_weights  = (float*)malloc(n_syn * sizeof(float));
    *h_syn_type = (int*)malloc(n_syn * sizeof(int));

    // Second pass: fill connections
    srand(42);
    int* pos = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j && (float)rand() / RAND_MAX < p_connect) {
                int k = (*h_row_ptr)[i] + pos[i]++;
                (*h_col_idx)[k]  = j;
                (*h_syn_type)[k] = (i < N_E) ? 0 : 1;  // pre E→E/I, pre I→E/I
                (*h_weights)[k]  = (i < N_E) ? w_E : w_I;
            }
        }
    }
    free(counts); free(pos);
}

int main(int argc, char** argv)
{
    int   N    = (argc > 1) ? atoi(argv[1]) : 2000;
    float T_ms = (argc > 2) ? atof(argv[2]) : 500.0f;
    float dt   = 0.1f;
    int   T_steps = (int)(T_ms / dt);
    int   N_E  = (int)(0.8f * N);   // 80% excitatory

    // Membrane parameters
    float tau_m=20.f, E_L=-65.f, Rm=10.f, V_th=-55.f, V_reset=-70.f;
    int   T_ref = (int)(2.0f / dt);

    // Synaptic parameters
    float tau_E=5.f, tau_I=10.f, E_E=0.f, E_I=-80.f;
    float w_E=0.1f, w_I=0.4f;
    float p_conn = 0.1f;     // 10% connection probability

    CUDA_CHECK(cudaMemcpyToSymbol(c_dt,      &dt,      sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_m,   &tau_m,   sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_E_L,     &E_L,     sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Rm,      &Rm,      sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_V_th,    &V_th,    sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_V_reset, &V_reset, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_T_ref,   &T_ref,   sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_E,   &tau_E,   sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_I,   &tau_I,   sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_E_E,     &E_E,     sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_E_I,     &E_I,     sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_w_E,     &w_E,     sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_w_I,     &w_I,     sizeof(float)));

    printf("Network Simulation\n");
    printf("  N=%d (NE=%d, NI=%d), T=%.0f ms, p_conn=%.2f\n",
           N, N_E, N-N_E, T_ms, p_conn);

    // Build connectivity
    int *h_row_ptr, *h_col_idx, *h_syn_type, n_syn;
    float* h_weights;
    build_connectivity(N, N_E, p_conn, w_E, w_I,
                       &h_row_ptr, &h_col_idx, &h_weights, &h_syn_type, &n_syn);
    printf("  Synapses: %d (%.1f per neuron avg)\n", n_syn, (float)n_syn/N);

    // Device state
    size_t fb = N * sizeof(float), ib = N * sizeof(int);
    float *d_V, *d_gE, *d_gI, *d_Iext;
    int   *d_ref, *d_fired;
    CUDA_CHECK(cudaMalloc(&d_V,    fb)); CUDA_CHECK(cudaMalloc(&d_gE,   fb));
    CUDA_CHECK(cudaMalloc(&d_gI,   fb)); CUDA_CHECK(cudaMalloc(&d_Iext, fb));
    CUDA_CHECK(cudaMalloc(&d_ref,  ib)); CUDA_CHECK(cudaMalloc(&d_fired,ib));
    CUDA_CHECK(cudaMemset(d_gE, 0, fb)); CUDA_CHECK(cudaMemset(d_gI, 0, fb));
    CUDA_CHECK(cudaMemset(d_ref, 0, ib));

    // Init V at E_L
    float* h_V = (float*)malloc(fb);
    float* h_Iext = (float*)malloc(fb);
    for (int i = 0; i < N; i++) {
        h_V[i] = E_L;
        h_Iext[i] = (i < N_E) ? 1.5f : 1.0f;  // heterogeneous input
    }
    CUDA_CHECK(cudaMemcpy(d_V,    h_V,    fb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Iext, h_Iext, fb, cudaMemcpyHostToDevice));

    // Device connectivity
    int *d_row_ptr, *d_col_idx, *d_syn_type; float *d_weights;
    CUDA_CHECK(cudaMalloc(&d_row_ptr,  (N+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,  n_syn*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights,  n_syn*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_syn_type, n_syn*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr,  h_row_ptr,  (N+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx,  h_col_idx,  n_syn*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights,  h_weights,  n_syn*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_syn_type, h_syn_type, n_syn*sizeof(int),   cudaMemcpyHostToDevice));

    // Spike log
    int max_spikes = N * 500;
    int *d_spike_id, *d_ns; float *d_spike_t;
    CUDA_CHECK(cudaMalloc(&d_spike_id, max_spikes*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_spike_t,  max_spikes*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ns,       sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ns, 0, sizeof(int)));

    int thr=256, blk=(N+thr-1)/thr;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    for (int step = 0; step < T_steps; step++) {
        update_neurons<<<blk, thr>>>(d_V, d_gE, d_gI, d_ref, d_Iext, d_fired, N);
        propagate_spikes<<<blk, thr>>>(d_fired, d_row_ptr, d_col_idx,
                                        d_weights, d_syn_type, d_gE, d_gI, N);

        // Record spikes
        // (simplified: count total per neuron stored in d_ns as total spike count)
    }

    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    float sim_ms;
    CUDA_CHECK(cudaEventElapsedTime(&sim_ms, t0, t1));

    printf("  GPU time: %.2f ms (%.1fx real-time)\n", sim_ms, T_ms/sim_ms);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    cudaFree(d_V); cudaFree(d_gE); cudaFree(d_gI); cudaFree(d_Iext);
    cudaFree(d_ref); cudaFree(d_fired);
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_weights); cudaFree(d_syn_type);
    cudaFree(d_spike_id); cudaFree(d_spike_t); cudaFree(d_ns);
    free(h_V); free(h_Iext);
    free(h_row_ptr); free(h_col_idx); free(h_weights); free(h_syn_type);
    return 0;
}
