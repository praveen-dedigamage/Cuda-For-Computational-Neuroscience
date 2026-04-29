/*
 * stdp.cu
 * Module 05 — Spike-Timing-Dependent Plasticity
 *
 * Implements STDP using eligibility traces:
 *   Pre-synaptic trace:  x_pre  incremented on pre-spike, decays with tau_+
 *   Post-synaptic trace: x_post incremented on post-spike, decays with tau_-
 *
 * Weight updates:
 *   On post-spike:  dw = A_+ * x_pre[i]   (pre fired recently → LTP)
 *   On pre-spike:   dw = -A_- * x_post[j] (post fired recently → LTD)
 *
 * Soft weight bounds: A_+ scaled by (w_max - w) / w_max
 *                     A_- scaled by (w - w_min) / w_max
 *
 * Compile:  nvcc -O2 -o stdp stdp.cu -lm
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
__constant__ float c_tau_plus, c_tau_minus;   // STDP time constants
__constant__ float c_A_plus, c_A_minus;       // STDP learning rates
__constant__ float c_w_max, c_w_min;          // weight bounds

// ─────────────────────────────────────────────────────────────────────────────
// Update eligibility traces
// ─────────────────────────────────────────────────────────────────────────────
__global__ void decay_traces(float* x_pre, float* x_post, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x_pre[i]  *= expf(-c_dt / c_tau_plus);
    x_post[i] *= expf(-c_dt / c_tau_minus);
}

// ─────────────────────────────────────────────────────────────────────────────
// LIF step with STDP weight updates
//
// On pre-spike (neuron i fires):
//   - Increment x_pre[i]
//   - For all outgoing synapses i→j: apply LTD (post before pre)
//     dw = -A_- * x_post[j]  (penalise if post already fired recently)
//
// On post-spike (neuron j fires):
//   - Increment x_post[j]
//   - For all incoming synapses i→j: apply LTP (pre before post)
//     dw = A_+ * x_pre[i]  (reward if pre fired recently)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void lif_stdp_step(
    float* V, float* g_E, int* ref,
    float* x_pre, float* x_post,
    const int* row_ptr_out, const int* col_out,   // outgoing connections
    const int* row_ptr_in,  const int* col_in,    // incoming connections
    float* weights,                               // [n_syn] weights
    const int* syn_out_idx,                       // index into weights[] for each out-synapse
    int*  fired,
    const float* I_ext, int N
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    // Decay conductance
    g_E[j] *= expf(-c_dt / 5.0f);  // tau_E = 5 ms (hardcoded for simplicity)

    fired[j] = 0;

    if (ref[j] > 0) { ref[j]--; V[j] = c_V_reset; return; }

    float I_syn = g_E[j] * (V[j] - 0.0f);  // E_E = 0 mV
    float dV = c_dt / c_tau_m * (-(V[j] - c_E_L) + c_Rm * (I_ext[j] - I_syn));
    V[j] += dV;

    if (V[j] >= c_V_th) {
        V[j] = c_V_reset;
        ref[j] = c_T_ref;
        fired[j] = 1;

        // Post-spike: increment post trace
        x_post[j] += 1.0f;

        // Post-spike LTP: for each incoming synapse i→j, dw += A+ * x_pre[i]
        for (int k = row_ptr_in[j]; k < row_ptr_in[j+1]; k++) {
            int i   = col_in[k];
            int idx = syn_out_idx[k];
            float dw = c_A_plus * x_pre[i] * (c_w_max - weights[idx]) / c_w_max;
            atomicAdd(&weights[idx], dw);
        }
    }
}

__global__ void apply_pre_spike_ltd(
    const int* fired,
    const int* row_ptr_out, const int* col_out,
    float* x_pre, float* x_post,
    float* weights, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !fired[i]) return;

    // Pre-spike: increment pre trace
    x_pre[i] += 1.0f;

    // Pre-spike LTD: for each outgoing synapse i→j, dw -= A- * x_post[j]
    for (int k = row_ptr_out[i]; k < row_ptr_out[i+1]; k++) {
        int j = col_out[k];
        float dw = -c_A_minus * x_post[j] * (weights[k] - c_w_min) / c_w_max;
        atomicAdd(&weights[k], dw);
    }
}

int main() {
    printf("STDP network simulation\n");
    printf("See 03_stdp_learning.ipynb for the full interactive version.\n");
    printf("This src/ file demonstrates the kernel design.\n");
    return 0;
}
