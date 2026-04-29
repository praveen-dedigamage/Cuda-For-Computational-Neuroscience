# Module 05 — Synaptic Networks & Spike-Timing-Dependent Plasticity

## Overview

We now connect neurons into networks. This module covers synaptic models, efficient sparse connectivity representations on GPU, and STDP — the biologically inspired learning rule where synapse strength changes based on the relative timing of pre- and post-synaptic spikes.

## Key Models

### Exponential Synapse
$$\tau_s \frac{dg}{dt} = -g + w \cdot \delta(t - t_{pre})$$
$$I_{syn}(t) = g(t) \cdot (V - E_{syn})$$

### STDP Rule
$$\Delta w = \begin{cases} A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \text{ (pre before post)} \\ -A_- e^{\Delta t / \tau_-} & \text{if } \Delta t < 0 \text{ (post before pre)} \end{cases}$$

where $\Delta t = t_{post} - t_{pre}$.

## Learning Objectives

1. Implement sparse synaptic connectivity (CSR format) on GPU
2. Model exponential and alpha-function synaptic conductances
3. Implement the STDP learning rule with eligibility traces
4. Simulate a recurrent LIF network with plastic synapses
5. Observe Hebbian learning emerge from STDP

## Notebooks

| Notebook | Topic |
|----------|-------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_05_networks_plasticity/01_synaptic_models.ipynb) [01_synaptic_models.ipynb](01_synaptic_models.ipynb) | Conductance-based synapses on GPU |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_05_networks_plasticity/02_sparse_connectivity.ipynb) [02_sparse_connectivity.ipynb](02_sparse_connectivity.ipynb) | CSR format, sparse matrix-vector multiply |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_05_networks_plasticity/03_stdp_learning.ipynb) [03_stdp_learning.ipynb](03_stdp_learning.ipynb) | STDP rule, weight dynamics, learning |

## Exercises

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_05_networks_plasticity/exercises/ex05_stub.ipynb) [exercises/ex05_stub.ipynb](exercises/ex05_stub.ipynb) | Implement BCM (Bienenstock-Cooper-Munro) plasticity |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_05_networks_plasticity/exercises/ex05_solution.ipynb) [exercises/ex05_solution.ipynb](exercises/ex05_solution.ipynb) | Complete solution |
