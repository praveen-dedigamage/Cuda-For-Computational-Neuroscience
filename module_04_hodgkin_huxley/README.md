# Module 04 — The Hodgkin-Huxley Model on GPU

## Overview

The Hodgkin-Huxley (HH) model is the gold standard of biophysically realistic neuron simulation. It explicitly models the dynamics of sodium, potassium, and leak ion channels using coupled ODEs. Solving HH for thousands of neurons requires GPU parallelism — and enables parameter sweeps that are impossible on CPU.

## The Hodgkin-Huxley Model

$$C_m \frac{dV}{dt} = -g_{Na} m^3 h (V - E_{Na}) - g_K n^4 (V - E_K) - g_L (V - E_L) + I_{ext}$$

Gating variable dynamics (example for m):
$$\frac{dm}{dt} = \alpha_m(V)(1-m) - \beta_m(V) m$$

where $\alpha_m$ and $\beta_m$ are voltage-dependent rate functions.

## Learning Objectives

1. Implement the full HH model as a CUDA kernel with 4 coupled ODEs per neuron
2. Use `__device__` functions for voltage-dependent rate constants
3. Implement 4th-order Runge-Kutta on the GPU
4. Perform parameter sweeps: simulate identical neurons with different I_ext in parallel
5. Visualise action potential shape, ion channel kinetics, and f-I curves

## Notebooks

| Notebook | Topic | |
|----------|-------|-|
| [01_hh_theory.ipynb](01_hh_theory.ipynb) | HH model derivation, gating variables, action potential | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_04_hodgkin_huxley/01_hh_theory.ipynb) |
| [02_hh_gpu_solver.ipynb](02_hh_gpu_solver.ipynb) | Euler and RK4 solvers on GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_04_hodgkin_huxley/02_hh_gpu_solver.ipynb) |
| [03_parameter_sweep.ipynb](03_parameter_sweep.ipynb) | Parallel parameter exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_04_hodgkin_huxley/03_parameter_sweep.ipynb) |

## Exercises

| File | Description | |
|------|-------------|-|
| [exercises/ex04_stub.ipynb](exercises/ex04_stub.ipynb) | Add a potassium A-current (I_A) to the model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_04_hodgkin_huxley/exercises/ex04_stub.ipynb) |
| [exercises/ex04_solution.ipynb](exercises/ex04_solution.ipynb) | Complete solution | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/praveen-dedigamage/Cuda-For-Computational-Neuroscience/blob/main/module_04_hodgkin_huxley/exercises/ex04_solution.ipynb) |

## Estimated Time

- Lectures: 4–5 hours
- Exercises: 2–3 hours
