# Project 02 — Neural Field Theory on GPU

## Overview

Simulate the **Wilson-Cowan equations** on a 2D cortical grid (256×256 pixels). The connectivity is a Mexican-hat kernel (local excitation, distant inhibition) applied via **cuFFT-based convolution**. Observe self-organised spatial patterns: travelling waves, standing oscillations, and Turing instabilities.

## Scientific Background

Wilson-Cowan (1972) equations model the mean firing rate of excitatory (E) and inhibitory (I) populations:

$$\tau_E \frac{du}{dt} = -u + S(w_{EE} * u - w_{EI} * v + I_E)$$
$$\tau_I \frac{dv}{dt} = -v + S(w_{IE} * u - w_{II} * v + I_I)$$

where:
- $u, v$ are excitatory and inhibitory population activities (0–1)
- $S(x) = 1/(1+e^{-x})$ is the sigmoid activation function  
- $*$ denotes spatial convolution with a connectivity kernel
- $w_{XY}$ kernels have the form: Gaussian (excitation) - wider Gaussian (inhibition) = Mexican hat

## Tasks

### Task 1: cuFFT Convolution

Complete `neural_field.cu`:
- [ ] Implement `convolve_fft`: multiply spectra and IFFT
- [ ] Apply Mexican-hat connectivity kernel
- [ ] Euler integration of Wilson-Cowan equations

### Task 2: Pattern Exploration

In the analysis notebook, vary parameters to observe:
- [ ] Homogeneous steady state (weak connectivity)
- [ ] Travelling waves (intermediate connectivity + noise)
- [ ] Turing patterns / spatial oscillations (strong Mexican-hat kernel)
- [ ] Plot: 2D activation map at t=0, 250, 500, 1000 ms

### Task 3: Dispersion Relation

For a linearised version around the uniform state, compute the dispersion relation analytically:
$$\sigma(k) = -1/\tau_E + w_{EE} \hat{K}(k) S'(u^*)$$
where $\hat{K}(k)$ is the Fourier transform of the kernel and $k$ is spatial frequency.

- [ ] Plot $\sigma(k)$ vs $k$ — find the unstable spatial frequency
- [ ] Compare to the dominant pattern size observed in simulation

### Task 4: Performance

- [ ] cuFFT convolution vs direct (loop-based) convolution
- [ ] Benchmark grid sizes: 64², 128², 256², 512²
- [ ] Plot speedup

## Parameters

```
Grid:      256 × 256 pixels (cortical surface, 1 mm² per pixel)
dt:        0.1 ms
T:         1000 ms
tau_E:     10 ms
tau_I:     20 ms

Connectivity kernels (Mexican hat):
  w_EE(r) = A_EE * exp(-r²/2σ_EE²) - B_EE * exp(-r²/2σ_IE²)
  A_EE=4,  σ_EE=2 mm,  B_EE=2, σ_IE=8 mm

External input:
  I_E = 0.5 + 0.1 * noise
  I_I = 0.5
```

## Files

| File | Description |
|------|-------------|
| `neural_field.cu` | Main simulation (complete the TODOs) |
| `analysis.ipynb` | Analysis + visualization notebook |
| `run.sh` | Compile and run |
