# Capstone Projects

These two projects integrate all course material into self-contained GPU neuroscience simulations. Each is a realistic research-scale problem with starter code and clear deliverables.

## Project 01 — Large-Scale Spiking Neural Network with STDP

**Goal:** Simulate 100,000 LIF neurons with sparse excitatory/inhibitory connectivity and STDP. Analyse the emergent dynamics and weight distribution.

**Skills covered:** Modules 01–05 (CUDA, memory, LIF, STDP, sparse connectivity)

**Deliverable:** Working GPU simulation + analysis notebook showing:
- Raster plot and population firing rate
- Weight distribution before/after STDP
- Scaling benchmark: neurons/second

[→ Go to Project 01](project_01_large_scale_snn/)

---

## Project 02 — Neural Field Theory on GPU

**Goal:** Simulate Wilson-Cowan rate equations on a 2D cortical grid (512×512 pixels). Use cuFFT for the convolution with the cortical connectivity kernel. Observe travelling waves and Turing patterns.

**Skills covered:** Modules 01–02, 06 (cuFFT, memory, advanced kernels)

**Deliverable:** GPU simulation + visualization notebook showing:
- 2D activation maps at multiple time points
- Dispersion relation (frequency vs spatial frequency)
- cuFFT vs direct convolution performance comparison

[→ Go to Project 02](project_02_neural_field_theory/)
