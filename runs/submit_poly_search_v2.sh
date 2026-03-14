#!/bin/bash
#
#SBATCH --job-name=poly_v2
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=04:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/poly_v2_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/poly_v2_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#
# EXPERIMENT: Polynomial schedule search v2 — sequential greedy optimization.
# 4 experiments + GPU benchmark on 2 H100s.

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: Poly Search v2 — Sequential Greedy"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=8
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Force unbuffered Python output so we can monitor progress
python -u poly_search_v2.py

echo "========================================"
echo "Poly search v2 completed at: $(date)"
echo "========================================"
