/*
 * cufft_lfp.cu
 * Module 06 — cuFFT for LFP Spectral Analysis
 *
 * Demonstrates:
 *   1. Single-channel power spectrum: P(f) = |FFT(x)|^2
 *   2. Batch FFT: compute spectra for N channels simultaneously
 *   3. Short-time Fourier Transform (STFT) — spectrogram
 *
 * Synthetic LFP: sum of oscillations at 8 Hz (theta), 40 Hz (gamma),
 *               120 Hz (high-gamma) + white noise
 *
 * Compile: nvcc -O2 -o cufft_lfp cufft_lfp.cu -lcufft -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); } \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult e = (call); \
    if (e != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, e); \
        exit(1); } \
} while(0)

// Compute power spectrum from complex FFT output
__global__ void compute_power(const cufftComplex* fft_out, float* power,
                               int N_fft, int n_channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_freq = N_fft / 2 + 1;  // one-sided spectrum
    if (idx >= n_channels * n_freq) return;

    int ch  = idx / n_freq;
    int bin = idx % n_freq;
    cufftComplex c = fft_out[ch * N_fft + bin];  // batch layout
    float scale = 1.0f / (N_fft * N_fft);
    power[idx] = (c.x * c.x + c.y * c.y) * scale;
}

// Apply Hann window before FFT
__global__ void apply_hann(float* signal, int N_fft, int n_channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_channels * N_fft) return;
    int bin = idx % N_fft;
    float w = 0.5f * (1.0f - cosf(2.0f * M_PI * bin / (N_fft - 1)));
    signal[idx] *= w;
}

int main(int argc, char** argv)
{
    float fs   = 1000.0f;  // sampling rate (Hz)
    float T    = 4.0f;     // duration (seconds)
    int   N    = (int)(fs * T);  // 4000 samples
    int   N_ch = 64;       // number of LFP channels (e.g., electrode array)

    printf("=== cuFFT LFP Analysis ===\n");
    printf("Signal: %d samples @ %.0f Hz = %.1f s\n", N, fs, T);
    printf("Channels: %d\n", N_ch);

    // ─── 1. Generate synthetic LFP ─────────────────────────────────────────
    float* h_lfp = (float*)malloc((size_t)N_ch * N * sizeof(float));
    srand(42);
    for (int ch = 0; ch < N_ch; ch++) {
        float phase_theta = (float)rand()/RAND_MAX * 2*M_PI;
        float phase_gamma = (float)rand()/RAND_MAX * 2*M_PI;
        float phase_hg    = (float)rand()/RAND_MAX * 2*M_PI;
        for (int t = 0; t < N; t++) {
            float sec = (float)t / fs;
            float theta  = 1.5f * sinf(2*M_PI*8.f*sec  + phase_theta);
            float gamma  = 0.8f * sinf(2*M_PI*40.f*sec + phase_gamma);
            float hgamma = 0.3f * sinf(2*M_PI*120.f*sec+ phase_hg);
            float noise  = 0.2f * ((float)rand()/RAND_MAX * 2.f - 1.f);
            h_lfp[ch * N + t] = theta + gamma + hgamma + noise;
        }
    }

    // ─── 2. Batch FFT: all channels simultaneously ─────────────────────────
    float* d_lfp;
    cufftComplex* d_fft;
    float* d_power;

    int n_freq = N / 2 + 1;
    CUDA_CHECK(cudaMalloc(&d_lfp,   (size_t)N_ch * N      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fft,   (size_t)N_ch * N      * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_power, (size_t)N_ch * n_freq * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lfp, h_lfp, (size_t)N_ch*N*sizeof(float),
                          cudaMemcpyHostToDevice));

    // Apply Hann window
    int thr = 256;
    int blk = ((size_t)N_ch * N + thr - 1) / thr;
    apply_hann<<<blk, thr>>>(d_lfp, N, N_ch);

    // Create batched FFT plan: N_ch transforms of length N
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, N_ch));

    cudaEvent_t t0, t1; float ms;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    int REPS = 50;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < REPS; r++) {
        CUFFT_CHECK(cufftExecR2C(plan, d_lfp, d_fft));
    }
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("Batch FFT (%d channels × N=%d): %.3f ms/call\n",
           N_ch, N, ms/REPS);

    // Compute power spectra
    blk = ((size_t)N_ch * n_freq + thr - 1) / thr;
    compute_power<<<blk, thr>>>(d_fft, d_power, N, N_ch);

    // Copy back power spectrum of channel 0 for verification
    float* h_power = (float*)malloc(n_freq * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_power, d_power, n_freq * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Find peaks
    printf("\nPower spectrum peaks (channel 0):\n");
    float df = fs / N;
    float peak_val = 0.f; int peak_bin = 0;
    for (int b = 1; b < n_freq - 1; b++) {
        if (h_power[b] > h_power[b-1] && h_power[b] > h_power[b+1]) {
            if (h_power[b] > 0.01f * h_power[1]) {  // threshold
                printf("  f = %.1f Hz, P = %.4f\n", b * df, h_power[b]);
            }
            if (h_power[b] > peak_val) { peak_val = h_power[b]; peak_bin = b; }
        }
    }
    printf("Dominant frequency: %.1f Hz\n", peak_bin * df);

    // Save full power spectrum for Python plotting
    FILE* f = fopen("lfp_power_spectrum.txt", "w");
    fprintf(f, "# freq_Hz  power\n");
    for (int b = 0; b < n_freq; b++)
        fprintf(f, "%.3f %.8f\n", b * df, h_power[b]);
    fclose(f);
    printf("\nPower spectrum saved to lfp_power_spectrum.txt\n");

    // ─── 3. STFT (spectrogram) ──────────────────────────────────────────────
    printf("\n=== STFT (spectrogram) ===\n");
    int win = 256;        // window length (samples)
    int hop = 64;         // hop size (samples)
    int n_frames = (N - win) / hop + 1;
    int n_freq_stft = win / 2 + 1;
    printf("Window=%d samples (%.0f ms), hop=%d (%.0f ms), frames=%d\n",
           win, 1000.f*win/fs, hop, 1000.f*hop/fs, n_frames);

    // Prepare windowed frames for channel 0
    float* h_frames = (float*)malloc((size_t)n_frames * win * sizeof(float));
    for (int fr = 0; fr < n_frames; fr++) {
        int start = fr * hop;
        for (int s = 0; s < win; s++) {
            float w = 0.5f * (1.0f - cosf(2.0f * M_PI * s / (win - 1)));
            h_frames[fr * win + s] = (start+s < N) ? h_lfp[start+s] * w : 0.f;
        }
    }

    float* d_frames; cufftComplex* d_stft_out; float* d_stft_power;
    CUDA_CHECK(cudaMalloc(&d_frames,    (size_t)n_frames*win*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stft_out,  (size_t)n_frames*win*sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_stft_power,(size_t)n_frames*n_freq_stft*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_frames, h_frames,
                          (size_t)n_frames*win*sizeof(float), cudaMemcpyHostToDevice));

    cufftHandle stft_plan;
    CUFFT_CHECK(cufftPlan1d(&stft_plan, win, CUFFT_R2C, n_frames));
    CUFFT_CHECK(cufftExecR2C(stft_plan, d_frames, d_stft_out));

    blk = ((size_t)n_frames * n_freq_stft + thr - 1) / thr;
    compute_power<<<blk, thr>>>(d_stft_out, d_stft_power, win, n_frames);

    float* h_stft_power = (float*)malloc((size_t)n_frames*n_freq_stft*sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_stft_power, d_stft_power,
                          (size_t)n_frames*n_freq_stft*sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Save spectrogram
    FILE* fsg = fopen("lfp_spectrogram.txt", "w");
    fprintf(fsg, "# rows=freq bins (%d), cols=time frames (%d)\n",
            n_freq_stft, n_frames);
    fprintf(fsg, "# freq_resolution=%.3f Hz, time_resolution=%.3f ms\n",
            fs/win, 1000.f*hop/fs);
    for (int b = 0; b < n_freq_stft; b++) {
        for (int fr = 0; fr < n_frames; fr++)
            fprintf(fsg, "%.6e ", h_stft_power[fr * n_freq_stft + b]);
        fprintf(fsg, "\n");
    }
    fclose(fsg);
    printf("Spectrogram saved to lfp_spectrogram.txt\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUFFT_CHECK(cufftDestroy(plan)); CUFFT_CHECK(cufftDestroy(stft_plan));
    cudaFree(d_lfp); cudaFree(d_fft); cudaFree(d_power);
    cudaFree(d_frames); cudaFree(d_stft_out); cudaFree(d_stft_power);
    free(h_lfp); free(h_power); free(h_frames); free(h_stft_power);
    return 0;
}
