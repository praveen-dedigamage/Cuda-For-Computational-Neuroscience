#!/usr/bin/env bash
# Local setup for GPU Programming for Computational Neuroscience
# Tested on Ubuntu 22.04 + CUDA 12.x

set -e

echo "=== GPU Programming for Computational Neuroscience — Local Setup ==="

# --- 1. Check for NVIDIA GPU ---
echo ""
echo "[1/5] Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install the NVIDIA driver first."
    echo "  Ubuntu: sudo apt install nvidia-driver-535"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "GPU check passed."

# --- 2. Check for CUDA Toolkit ---
echo ""
echo "[2/5] Checking for CUDA Toolkit (nvcc)..."
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found. Installing CUDA Toolkit..."
    echo "Please follow: https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "Quick install (Ubuntu 22.04, CUDA 12.1):"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "  sudo dpkg -i cuda-keyring_1.1-1_all.deb"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y cuda-toolkit-12-1"
    echo "  echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc"
    echo "  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    exit 1
fi
nvcc --version
echo "CUDA Toolkit check passed."

# --- 3. Check for Python ---
echo ""
echo "[3/5] Setting up Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "python3 not found. Installing..."
    sudo apt-get install -y python3 python3-pip python3-venv
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# --- 4. Install Python dependencies ---
echo ""
echo "[4/5] Installing Python dependencies..."
pip install -r setup/requirements.txt
echo "Python dependencies installed."

# --- 5. Test compilation ---
echo ""
echo "[5/5] Testing CUDA compilation..."
cat > /tmp/test_cuda.cu << 'EOF'
#include <stdio.h>
__global__ void hello() { printf("CUDA OK: thread %d\n", threadIdx.x); }
int main() { hello<<<1,4>>>(); cudaDeviceSynchronize(); return 0; }
EOF
nvcc -o /tmp/test_cuda /tmp/test_cuda.cu
/tmp/test_cuda
echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start:"
echo "  source .venv/bin/activate"
echo "  jupyter notebook"
