#!/bin/bash
#
#SBATCH --job-name=pi_baseline
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=06:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/pi_baseline_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/pi_baseline_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#
# BASELINE: Standard Muon (Polar Express) at depth 20.
# This is the control experiment for the Polar Interpolation comparison.

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: BASELINE (standard Muon, depth=20)"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=4 \
    -m scripts.base_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --model-tag pi_baseline_d20 \
    --run dummy

echo "========================================"
echo "Baseline completed at: $(date)"
echo "========================================"
