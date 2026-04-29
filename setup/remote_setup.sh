#!/usr/bin/env bash
# Remote setup — AWS / GCP / HPC clusters
# Run this after SSH-ing into a GPU instance

set -e

echo "=== Remote GPU Setup for Computational Neuroscience Course ==="

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: AWS EC2 (p3.2xlarge = V100, p3.8xlarge = 4x V100)
# ─────────────────────────────────────────────────────────────────────────────
# Launch with Deep Learning AMI (Ubuntu) — CUDA pre-installed
#
#   aws ec2 run-instances \
#     --image-id ami-0abcdef1234567890 \   # Deep Learning AMI ID for your region
#     --instance-type p3.2xlarge \
#     --key-name YOUR_KEY \
#     --security-group-ids sg-XXXXXXXX \
#     --count 1
#
#   ssh -i YOUR_KEY.pem ubuntu@<EC2_PUBLIC_IP>
#
# On the instance:
#   git clone https://github.com/YOUR_USERNAME/GPU-Programming-For-Computational-Neuroscience.git
#   cd GPU-Programming-For-Computational-Neuroscience
#   bash setup/remote_setup.sh

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Google Cloud (n1-standard-4 + T4)
# ─────────────────────────────────────────────────────────────────────────────
#
#   gcloud compute instances create gpu-neuro \
#     --zone=us-central1-a \
#     --machine-type=n1-standard-4 \
#     --accelerator=type=nvidia-tesla-t4,count=1 \
#     --image-family=common-cu121 \
#     --image-project=deeplearning-platform-release \
#     --maintenance-policy=TERMINATE
#
#   gcloud compute ssh gpu-neuro

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: HPC Cluster (SLURM — adapt to your system)
# ─────────────────────────────────────────────────────────────────────────────
# Example SLURM batch script is written to: setup/slurm_job.sh (see below)

# ─────────────────────────────────────────────────────────────────────────────
# ACTUAL SETUP COMMANDS (runs on the remote machine)
# ─────────────────────────────────────────────────────────────────────────────

echo "[1/4] Updating system packages..."
sudo apt-get update -qq

echo "[2/4] Installing Python and Jupyter..."
sudo apt-get install -y python3 python3-pip python3-venv

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r setup/requirements.txt

echo "[3/4] Verifying CUDA..."
nvcc --version
nvidia-smi

echo "[4/4] Starting Jupyter on port 8888 (no browser)..."
echo ""
echo "  Run this command on your LOCAL machine to create an SSH tunnel:"
echo "  ssh -N -L 8888:localhost:8888 user@REMOTE_IP"
echo ""
echo "  Then open: http://localhost:8888"
echo ""
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# ─────────────────────────────────────────────────────────────────────────────
# Write SLURM example script
# ─────────────────────────────────────────────────────────────────────────────
cat > setup/slurm_job.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=neuro_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out

module load cuda/12.1
module load python/3.10

source .venv/bin/activate

# Example: compile and run the LIF simulation from Module 3
nvcc -O2 -o bin/lif module_03_lif_neurons/src/lif_simulation.cu
./bin/lif --neurons 10000 --steps 10000 --dt 0.1 > results/lif_output.txt

echo "Job complete."
SLURM

echo "SLURM example written to setup/slurm_job.sh"
