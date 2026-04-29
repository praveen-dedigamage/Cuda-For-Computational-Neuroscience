# Project 01 — Large-Scale Spiking Neural Network with STDP

## Overview

Simulate a network of **100,000 LIF neurons** (80% excitatory, 20% inhibitory) with sparse random connectivity (p=0.1) and STDP on excitatory synapses. Run for 5 seconds of simulated time and analyse the emergent dynamics.

## Scientific Background

Large-scale cortical simulations (Brunel 2000, Vogels & Abbott 2005) have shown that balanced excitation-inhibition produces **asynchronous irregular (AI)** firing — a hallmark of cortical dynamics in awake animals. STDP acting on this background can spontaneously form **cell assemblies** — groups of neurons with strong mutual connections that fire together.

## Tasks

### Task 1: Set up the simulation (starter code provided)

Complete the simulation in `large_snn.cu`:
- [ ] Build CSR connectivity for N=100,000 neurons
- [ ] Run for 5,000 ms with dt=0.1 ms
- [ ] Record spikes to file every 10 steps
- [ ] Log mean weights every 500 ms

### Task 2: Baseline analysis (no STDP)

Run with STDP disabled. In the analysis notebook:
- [ ] Raster plot (sample 500 neurons)
- [ ] Population firing rate (20 ms bins)
- [ ] Inter-spike interval (ISI) distribution and CV
- [ ] Verify AI state: mean ISI >> tau_m, CV ≈ 1

### Task 3: STDP analysis

Enable STDP and compare:
- [ ] Weight distribution: initial vs 2000 ms vs 5000 ms
- [ ] Do bimodal weights emerge?
- [ ] Do high-weight neurons form clusters (check weight matrix subblock)
- [ ] How does mean firing rate change with STDP?

### Task 4: Scaling benchmark

Vary N from 10,000 to 100,000. For each N:
- [ ] GPU time per 100 ms of simulation
- [ ] Plot throughput: million neuron-steps per second
- [ ] Estimate time for N=1,000,000 neurons

## Parameters

```
N          = 100,000 (NE = 80,000, NI = 20,000)
p_connect  = 0.1
dt         = 0.1 ms
tau_m      = 20 ms (E), 10 ms (I)
E_L        = -65 mV
V_th       = -55 mV
V_reset    = -70 mV
tau_ref    = 2 ms
tau_E      = 5 ms (AMPA)
tau_I      = 10 ms (GABA-A)
w_E        = 0.1 nS (initial excitatory weight)
w_I        = 0.4 nS (inhibitory weight, fixed)
I_ext_E    = N(2.0, 0.5) pA
I_ext_I    = N(1.0, 0.3) pA

STDP (excitatory only):
A_plus     = 0.005
A_minus    = 0.00525
tau_plus   = 20 ms
tau_minus  = 20 ms
w_max      = 0.5
```

## Files

| File | Description |
|------|-------------|
| `large_snn.cu` | Main simulation (complete to run) |
| `analysis.ipynb` | Analysis notebook (with plotting templates) |
| `run.sh` | Compile and run script |
