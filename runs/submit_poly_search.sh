#!/bin/bash
#
#SBATCH --job-name=poly_search
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=04:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/poly_search_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/poly_search_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#
# EXPERIMENT: Polynomial schedule search for Newton-Schulz orthogonalization.
# Find iteration schedules that achieve bf16 accuracy in fewer matmuls than Muon's 5×quintic.
# Uses 2 H100 GPUs (benchmarking on cuda:0, second GPU available for parallel work).

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: Polynomial Schedule Search"
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

# Run the polynomial search script
python poly_search.py

echo "========================================"
echo "Polynomial search completed at: $(date)"
echo "========================================"
