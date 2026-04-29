# Module 03 — Parallel Neuron Simulation: Leaky Integrate-and-Fire

## Overview

This is where GPU programming meets computational neuroscience. We implement the **Leaky Integrate-and-Fire (LIF)** model — the workhorse of computational neuroscience — for 10,000 neurons in parallel on the GPU.

## The LIF Model

$$\tau_m \frac{dV}{dt} = -(V - E_L) + R_m I(t)$$

**Spiking rule:** If $V \geq V_{th}$, fire a spike and reset $V \leftarrow V_{reset}$

Parameters:
- $\tau_m = 20$ ms — membrane time constant  
- $E_L = -65$ mV — leak reversal potential
- $R_m = 10$ MΩ — membrane resistance  
- $V_{th} = -55$ mV — spike threshold  
- $V_{reset} = -70$ mV — post-spike reset

## Learning Objectives

1. Implement a complete LIF simulation with spike detection on GPU
2. Apply Structure of Arrays (SoA) layout for coalesced access
3. Record spike times into a GPU-resident spike array
4. Visualize raster plots and firing rate histograms from GPU output
5. Benchmark simulation throughput vs CPU

## Notebooks

<table>
<thead><tr><th>Notebook</th><th>Topic</th></tr></thead>
<tbody>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_03_lif_neurons/01_lif_theory.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="01_lif_theory.ipynb">01_lif_theory.ipynb</a></td><td valign="middle">LIF model derivation and analysis</td></tr>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_03_lif_neurons/02_parallel_lif_gpu.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="02_parallel_lif_gpu.ipynb">02_parallel_lif_gpu.ipynb</a></td><td valign="middle">Full GPU implementation</td></tr>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_03_lif_neurons/03_spike_analysis.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="03_spike_analysis.ipynb">03_spike_analysis.ipynb</a></td><td valign="middle">Raster plots, ISI, firing rate analysis</td></tr>
</tbody>
</table>

## Exercises

<table>
<thead><tr><th>Notebook</th><th>Description</th></tr></thead>
<tbody>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_03_lif_neurons/exercises/ex03_stub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="exercises/ex03_stub.ipynb">exercises/ex03_stub.ipynb</a></td><td valign="middle">Add refractory period and heterogeneous inputs</td></tr>
<tr><td valign="middle"><a href="https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_03_lif_neurons/exercises/ex03_solution.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <a href="exercises/ex03_solution.ipynb">exercises/ex03_solution.ipynb</a></td><td valign="middle">Complete solution</td></tr>
</tbody>
</table>

## Standalone Source

| File | Description |
|------|-------------|
| [src/lif_simulation.cu](src/lif_simulation.cu) | Complete LIF simulation, 10,000 neurons |
| [src/run_local.sh](src/run_local.sh) | Compile and run |
