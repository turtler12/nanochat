#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/abl_%x_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/abl_%x_%j.err

# Usage: sbatch --job-name=<mode> runs/submit_ablation.sh <mode>
# e.g.:  sbatch --job-name=muon runs/submit_ablation.sh muon

MODE=${1:?"Usage: sbatch runs/submit_ablation.sh <update-mode>"}

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Mode: $MODE"
echo "Start: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export OMP_NUM_THREADS=1
export TMPDIR="/tmp"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=1 \
    -m scripts.ablation_train \
    --update-mode "$MODE" \
    --depth 12 \
    --num-iterations 2205 \
    --matrix-lr 0.02 \
    --run "ablation_${MODE}_d12"

echo "========================================"
echo "Done: $(date)"
echo "========================================"
