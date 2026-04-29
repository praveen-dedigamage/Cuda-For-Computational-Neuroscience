/*
 * large_snn.cu
 * Project 01 — Large-Scale Spiking Neural Network with STDP
 *
 * Starter code: fill in the TODO sections.
 *
 * Compile:  nvcc -O2 -o large_snn large_snn.cu -lm
 * Run:      ./large_snn [N] [T_ms] [enable_stdp]
 *           e.g.: ./large_snn 10000 5000 1
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

// ─── Constant memory ─────────────────────────────────────────────────────────
__constant__ float c_dt, c_tau_m_E, c_tau_m_I, c_E_L, c_Rm;
__constant__ float c_V_th, c_V_reset;
__constant__ int   c_T_ref;
__constant__ float c_tau_E, c_tau_I, c_E_E, c_E_I;
__constant__ float c_tau_plus, c_tau_minus, c_A_plus, c_A_minus;
__constant__ float c_w_max, c_w_min;

// ─── Neuron update ────────────────────────────────────────────────────────────
__global__ void update_neurons(
    float* V, float* g_E, float* g_I, int* ref,
    const float* I_ext, const int* is_exc,
    int* fired, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    g_E[i] *= expf(-c_dt / c_tau_E);
    g_I[i] *= expf(-c_dt / c_tau_I);
    fired[i] = 0;

    if (ref[i] > 0) { ref[i]--; V[i] = c_V_reset; return; }

    // Different tau_m for E and I neurons
    float tau_m = is_exc[i] ? c_tau_m_E : c_tau_m_I;
    float I_syn = -(g_E[i]*(V[i]-c_E_E) + g_I[i]*(V[i]-c_E_I));
    V[i] += c_dt / tau_m * (-(V[i] - c_E_L) + c_Rm * (I_ext[i] + I_syn));

    if (V[i] >= c_V_th) {
        V[i] = c_V_reset; ref[i] = c_T_ref; fired[i] = 1;
    }
}

// ─── Spike propagation + STDP LTD ─────────────────────────────────────────────
__global__ void propagate_and_ltd(
    const int* fired, const int* row_out, const int* col_out,
    float* weights, const int* syn_type,
    float* g_E, float* g_I,
    float* x_pre, const float* x_post,
    int enable_stdp, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !fired[i]) return;

    if (enable_stdp) x_pre[i] += 1.0f;

    for (int k = row_out[i]; k < row_out[i+1]; k++) {
        int j = col_out[k];
        float w = weights[k];
        if (syn_type[k] == 0) atomicAdd(&g_E[j], w);
        else                  atomicAdd(&g_I[j], w);

        // LTD on excitatory synapses only
        if (enable_stdp && syn_type[k] == 0) {
            float dw = -c_A_minus * x_post[j] * (w - c_w_min) / c_w_max;
            atomicAdd(&weights[k], dw);
        }
    }
}

// ─── Post-spike LTP ───────────────────────────────────────────────────────────
__global__ void post_spike_ltp(
    const int* fired,
    const int* row_in, const int* col_in,
    float* weights, const int* syn_in_to_out,
    float* x_post, const float* x_pre,
    int enable_stdp, int N
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N || !fired[j]) return;

    if (enable_stdp) x_post[j] += 1.0f;

    if (!enable_stdp) return;

    // LTP on incoming excitatory synapses
    for (int k = row_in[j]; k < row_in[j+1]; k++) {
        int i   = col_in[k];
        int idx = syn_in_to_out[k];
        float dw = c_A_plus * x_pre[i] * (c_w_max - weights[idx]) / c_w_max;
        atomicAdd(&weights[idx], dw);
    }
}

// ─── Trace decay ──────────────────────────────────────────────────────────────
__global__ void decay_traces(float* x_pre, float* x_post, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x_pre[i]  *= expf(-c_dt / c_tau_plus);
    x_post[i] *= expf(-c_dt / c_tau_minus);
}

// ─── Build outgoing CSR ───────────────────────────────────────────────────────
void build_csr_out(int N, int N_E, float p, float w_E, float w_I,
                   int** rp, int** ci, float** wv, int** stype, int* nnz_out)
{
    printf("Building connectivity (N=%d, p=%.2f)...\n", N, p);
    srand(42);
    int* cnt = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j && (float)rand()/RAND_MAX < p) cnt[i]++;

    *rp = (int*)malloc((N+1)*sizeof(int)); (*rp)[0] = 0;
    for (int i = 0; i < N; i++) (*rp)[i+1] = (*rp)[i] + cnt[i];
    int nnz = (*rp)[N]; *nnz_out = nnz;

    *ci    = (int*)malloc(nnz*sizeof(int));
    *wv    = (float*)malloc(nnz*sizeof(float));
    *stype = (int*)malloc(nnz*sizeof(int));

    srand(42);
    int* pos = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j && (float)rand()/RAND_MAX < p) {
                int k = (*rp)[i] + pos[i]++;
                (*ci)[k] = j;
                (*stype)[k] = (i < N_E) ? 0 : 1;
                (*wv)[k] = (i < N_E) ? w_E : w_I;
            }
    free(cnt); free(pos);
    printf("  Synapses: %d (%.0f per neuron)\n", nnz, (float)nnz/N);
}

void build_csr_in(int N, const int* rp_out, const int* ci_out,
                  const int* stype_out, int nnz,
                  int** rp_in, int** ci_in, int** syn_map, int** stype_in)
{
    int* in_cnt = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++)
        for (int k = rp_out[i]; k < rp_out[i+1]; k++)
            if (stype_out[k] == 0) in_cnt[ci_out[k]]++;  // only excitatory

    *rp_in = (int*)malloc((N+1)*sizeof(int)); (*rp_in)[0] = 0;
    for (int j = 0; j < N; j++) (*rp_in)[j+1] = (*rp_in)[j] + in_cnt[j];
    int nnz_in = (*rp_in)[N];
    *ci_in   = (int*)malloc(nnz_in*sizeof(int));
    *syn_map = (int*)malloc(nnz_in*sizeof(int));
    *stype_in= (int*)malloc(nnz_in*sizeof(int));

    int* pos = (int*)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++)
        for (int k = rp_out[i]; k < rp_out[i+1]; k++)
            if (stype_out[k] == 0) {
                int j = ci_out[k];
                int slot = (*rp_in)[j] + pos[j]++;
                (*ci_in)[slot]   = i;
                (*syn_map)[slot] = k;
                (*stype_in)[slot]= 0;
            }
    free(in_cnt); free(pos);
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    int   N         = (argc > 1) ? atoi(argv[1]) : 10000;
    float T_ms      = (argc > 2) ? atof(argv[2]) : 5000.f;
    int   en_stdp   = (argc > 3) ? atoi(argv[3]) : 0;
    float dt        = 0.1f;
    int   T_steps   = (int)(T_ms / dt);
    int   N_E       = (int)(0.8f * N);

    // Neuron parameters
    float tau_m_E=20.f, tau_m_I=10.f, E_L=-65.f, Rm=10.f;
    float V_th=-55.f, V_reset=-70.f;
    int   T_ref = (int)(2.f / dt);

    // Synaptic parameters
    float tau_E=5.f, tau_I=10.f, E_E=0.f, E_I=-80.f;
    float w_E=0.1f, w_I=0.4f;
    float p_conn = 0.1f;

    // STDP
    float tau_plus=20.f, tau_minus=20.f;
    float A_plus=0.005f, A_minus=0.00525f;
    float w_max=0.5f, w_min=0.0f;

    printf("Large SNN: N=%d, NE=%d, NI=%d, T=%.0f ms, STDP=%s\n",
           N, N_E, N-N_E, T_ms, en_stdp ? "ON" : "OFF");

    // Upload constants
    CUDA_CHECK(cudaMemcpyToSymbol(c_dt,       &dt,       sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_m_E,  &tau_m_E,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_m_I,  &tau_m_I,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_E_L,      &E_L,      sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Rm,       &Rm,       sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_V_th,     &V_th,     sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_V_reset,  &V_reset,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_T_ref,    &T_ref,    sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_E,    &tau_E,    sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_I,    &tau_I,    sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_E_E,      &E_E,      sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_E_I,      &E_I,      sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_plus, &tau_plus, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tau_minus,&tau_minus,sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_A_plus,   &A_plus,   sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_A_minus,  &A_minus,  sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_w_max,    &w_max,    sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_w_min,    &w_min,    sizeof(float)));

    // Build connectivity
    int *h_rp_out, *h_ci_out, *h_stype_out, nnz_out;
    float* h_wv;
    build_csr_out(N, N_E, p_conn, w_E, w_I,
                  &h_rp_out, &h_ci_out, &h_wv, &h_stype_out, &nnz_out);

    int *h_rp_in=NULL, *h_ci_in=NULL, *h_syn_map=NULL, *h_stype_in=NULL;
    if (en_stdp)
        build_csr_in(N, h_rp_out, h_ci_out, h_stype_out, nnz_out,
                     &h_rp_in, &h_ci_in, &h_syn_map, &h_stype_in);

    // Device allocations
    float *d_V, *d_gE, *d_gI, *d_Iext, *d_xpre=NULL, *d_xpost=NULL, *d_wv;
    int   *d_ref, *d_fired, *d_is_exc;
    int   *d_rp_out, *d_ci_out, *d_stype_out;
    int   *d_rp_in=NULL, *d_ci_in=NULL, *d_syn_map=NULL;

    CUDA_CHECK(cudaMalloc(&d_V,       N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gE,      N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gI,      N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Iext,    N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref,     N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fired,   N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_exc,  N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wv,      nnz_out*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rp_out, (N+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ci_out,  nnz_out*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stype_out,nnz_out*sizeof(int)));

    if (en_stdp) {
        int nnz_in = h_rp_in ? h_rp_in[N] : 0;
        CUDA_CHECK(cudaMalloc(&d_xpre,  N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_xpost, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rp_in, (N+1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ci_in,  nnz_in*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_syn_map,nnz_in*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_rp_in, h_rp_in,(N+1)*sizeof(int),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ci_in, h_ci_in, nnz_in*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_syn_map,h_syn_map,nnz_in*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_xpre, 0, N*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_xpost,0, N*sizeof(float)));
    }

    // Init state
    float* h_V    = (float*)malloc(N*sizeof(float));
    float* h_Iext = (float*)malloc(N*sizeof(float));
    int*   h_is_exc=(int*)malloc(N*sizeof(int));
    srand(7);
    for (int i = 0; i < N; i++) {
        h_V[i]     = E_L + (V_th-E_L)*(float)rand()/RAND_MAX;
        h_Iext[i]  = (i < N_E) ? 2.0f + 0.5f*(float)rand()/RAND_MAX
                                : 1.0f + 0.3f*(float)rand()/RAND_MAX;
        h_is_exc[i]= (i < N_E) ? 1 : 0;
    }
    CUDA_CHECK(cudaMemcpy(d_V,      h_V,      N*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Iext,   h_Iext,   N*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_is_exc, h_is_exc, N*sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_gE, 0, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gI, 0, N*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ref,0, N*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_wv,      h_wv,     nnz_out*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rp_out,  h_rp_out,(N+1)*sizeof(int),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci_out,  h_ci_out, nnz_out*sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_stype_out,h_stype_out,nnz_out*sizeof(int),cudaMemcpyHostToDevice));

    int thr=256, blk=(N+thr-1)/thr;

    FILE* fspikes = fopen("project01_spikes.txt","w");
    FILE* fweights= fopen("project01_weights.txt","w");

    int* h_fired = (int*)malloc(N*sizeof(int));
    float* h_wv_snap = (float*)malloc(nnz_out*sizeof(float));

    int snap_every = (int)(500/dt);
    int log_every  = 10;
    long total_spikes = 0;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    for (int step = 0; step < T_steps; step++) {
        float t_ms = step * dt;

        if (en_stdp)
            decay_traces<<<blk,thr>>>(d_xpre, d_xpost, N);

        update_neurons<<<blk,thr>>>(d_V, d_gE, d_gI, d_ref,
                                     d_Iext, d_is_exc, d_fired, N);

        if (en_stdp)
            post_spike_ltp<<<blk,thr>>>(d_fired, d_rp_in, d_ci_in,
                                         d_wv, d_syn_map, d_xpost, d_xpre,
                                         en_stdp, N);

        propagate_and_ltd<<<blk,thr>>>(d_fired, d_rp_out, d_ci_out,
                                        d_wv, d_stype_out,
                                        d_gE, d_gI, d_xpre, d_xpost,
                                        en_stdp, N);

        if (step % log_every == 0) {
            CUDA_CHECK(cudaMemcpy(h_fired, d_fired, N*sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < N; i++)
                if (h_fired[i]) { fprintf(fspikes, "%d %.1f\n", i, t_ms); total_spikes++; }
        }

        if (step % snap_every == 0) {
            CUDA_CHECK(cudaMemcpy(h_wv_snap, d_wv, nnz_out*sizeof(float), cudaMemcpyDeviceToHost));
            fprintf(fweights, "# t=%.0f ms\n", t_ms);
            // Log summary stats, not all weights (too large for 100k neurons)
            double wsum = 0; float wmin = 1e9f, wmax = -1e9f;
            int n_exc_syn = 0;
            for (int k = 0; k < nnz_out; k++) {
                if (h_stype_out[k] == 0) {
                    wsum += h_wv_snap[k];
                    if (h_wv_snap[k] < wmin) wmin = h_wv_snap[k];
                    if (h_wv_snap[k] > wmax) wmax = h_wv_snap[k];
                    n_exc_syn++;
                }
            }
            fprintf(fweights, "%.6f %.6f %.6f\n",
                    (float)(wsum/n_exc_syn), wmin, wmax);
        }
    }

    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    float sim_ms;
    CUDA_CHECK(cudaEventElapsedTime(&sim_ms, ev0, ev1));

    printf("GPU time: %.2f ms (%.1fx real-time)\n", sim_ms, T_ms/sim_ms);
    printf("Total spikes: %ld (mean %.1f Hz)\n",
           total_spikes, (float)total_spikes / (N * T_ms / 1000.f));
    printf("Throughput: %.1f M neuron-steps/s\n",
           (float)T_steps * N / (sim_ms * 1e3f));

    fclose(fspikes); fclose(fweights);
    CUDA_CHECK(cudaEventDestroy(ev0)); CUDA_CHECK(cudaEventDestroy(ev1));

    cudaFree(d_V); cudaFree(d_gE); cudaFree(d_gI);
    cudaFree(d_Iext); cudaFree(d_ref); cudaFree(d_fired); cudaFree(d_is_exc);
    cudaFree(d_wv); cudaFree(d_rp_out); cudaFree(d_ci_out); cudaFree(d_stype_out);
    if (en_stdp) {
        cudaFree(d_xpre); cudaFree(d_xpost);
        cudaFree(d_rp_in); cudaFree(d_ci_in); cudaFree(d_syn_map);
        free(h_rp_in); free(h_ci_in); free(h_syn_map); free(h_stype_in);
    }
    free(h_V); free(h_Iext); free(h_is_exc); free(h_fired); free(h_wv_snap);
    free(h_rp_out); free(h_ci_out); free(h_wv); free(h_stype_out);
    return 0;
}
