# GPU Programming with CUDA C++ for Computational Neuroscience

A hands-on, lecture-style course teaching GPU programming (NVIDIA CUDA C++) through the lens of computational neuroscience. Every concept is grounded in a real neuroscience simulation — from single neurons to large spiking networks.

---

## Who This Course Is For

Researchers with basic programming experience (Python or C/C++) who want to:
- Understand how GPUs accelerate scientific computing
- Write and optimize CUDA C++ kernels from scratch
- Simulate biologically realistic neural models at scale

No prior GPU experience required.

---

## Course Structure

| Module | Topic | Neuroscience Application |
|--------|-------|--------------------------|
| [01](module_01_cuda_fundamentals/) | GPU Architecture & CUDA Fundamentals | Parallel neuron indexing |
| [02](module_02_memory_optimization/) | Memory Hierarchy & Optimization | State variable caching |
| [03](module_03_lif_neurons/) | Parallel Neuron Simulation — LIF | 10,000-neuron LIF network |
| [04](module_04_hodgkin_huxley/) | ODE Solvers on GPU — Hodgkin-Huxley | Ion channel parameter sweeps |
| [05](module_05_networks_plasticity/) | Synaptic Networks & STDP | Spike-timing-dependent plasticity |
| [06](module_06_advanced/) | cuBLAS, cuFFT & Multi-GPU | LFP spectral analysis |

Each module contains:
- **Lecture notebooks** — full explanations with runnable code (Google Colab-ready)
- **Exercise stubs** — guided fill-in-the-blank tasks
- **Solutions** — complete worked solutions
- **`src/`** — standalone `.cu` source files with local/remote run scripts

---

## How to Use This Course

### Option A — Google Colab (Recommended for learning)

Each notebook has a "Open in Colab" badge at the top. Colab provides free NVIDIA T4 GPU access.

1. Click the badge in any notebook, or go to [colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → GitHub → paste this repo URL
3. Runtime → Change runtime type → **GPU** (T4)
4. Run cells top to bottom

All CUDA code is compiled inside the notebook using `!nvcc`. No local installation needed.

### Option B — Local Machine (Linux / WSL2 / macOS with NVIDIA GPU)

```bash
# Clone the repo
git clone https://github.com/praveen-dedigamage/Cuda-For-Computational-Neuroscience.git
cd GPU-Programming-For-Computational-Neuroscience

# Run the setup script
chmod +x setup/local_setup.sh
./setup/local_setup.sh

# Launch Jupyter
jupyter notebook
```

Requirements: NVIDIA GPU, CUDA Toolkit 11.x or 12.x, Python 3.8+

### Option C — Remote HPC / Cloud VM

```bash
chmod +x setup/remote_setup.sh
./setup/remote_setup.sh
```

See `setup/remote_setup.sh` for instructions on AWS, Google Cloud, and university HPC clusters.

---

## Prerequisites

- Basic Python (loops, functions, NumPy arrays)
- Familiarity with C-style syntax is helpful but not required
- High school / undergraduate calculus (derivatives, ODEs) for neuroscience models

---

## Suggested Learning Path

```
Week 1-2:  Modules 01 + 02  (GPU fundamentals & memory)
Week 3-4:  Module 03        (LIF neurons — first real simulation)
Week 5-6:  Module 04        (Hodgkin-Huxley — ODE solvers)
Week 7-8:  Module 05        (Networks & plasticity)
Week 9-10: Module 06        (Advanced tools)
Week 11+:  Capstone projects
```

---

## Projects

Two open-ended capstone projects in [`projects/`](projects/):

1. **Large-Scale Spiking Neural Network** — simulate 100,000 LIF neurons with sparse STDP synapses
2. **Neural Field Theory on GPU** — solve Wilson-Cowan equations on a 2D cortical grid

---

## Environment Tested On

| Environment | CUDA | GPU |
|-------------|------|-----|
| Google Colab | 12.x | T4 |
| Ubuntu 22.04 | 12.1 | RTX 3090 |
| AWS p3.2xlarge | 11.8 | V100 |

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

If you use this material in your research or teaching, please cite:

```
GPU Programming for Computational Neuroscience
https://github.com/praveen-dedigamage/Cuda-For-Computational-Neuroscience
```
