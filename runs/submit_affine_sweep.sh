#!/bin/bash
#
#SBATCH --job-name=affine_sweep
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/affine_sweep_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/affine_sweep_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

# Ensure uv is on PATH (on SLURM nodes, HOME=/tmp/home/$USER)
export PATH="$HOME/.local/bin:$PATH"
# Also set TMPDIR and HOME-related caches to scratch to avoid AFS quota issues
export TMPDIR="/tmp"

# Override INTERCEPTS to only run 0.0, 0.5, 1.0
export INTERCEPTS="0.0 0.5 1.0"

# Run the sweep script (it picks up INTERCEPTS from env)
bash runs/affine_sweep.sh

echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
