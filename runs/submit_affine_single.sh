#!/bin/bash
#
#SBATCH --job-name=affine_sweep
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=06:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/affine_sweep_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/affine_sweep_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#
# Single affine sweep run. Expects env vars:
#   AFFINE_INTERCEPT  - e.g. "0.0"
#   AFFINE_SN_MODE    - "yes" or "no"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: intercept=$AFFINE_INTERCEPT, SN=$AFFINE_SN_MODE"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

# Ensure uv is on PATH
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"

# Setup venv
command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Run single training configuration
RESULTS_DIR="$(pwd)/affine_sweep_results"
mkdir -p "$RESULTS_DIR"

SN_TAG=$([ "$AFFINE_SN_MODE" = "yes" ] && echo "YES_SN" || echo "NO_SN")
RUN_TAG="affine_${AFFINE_INTERCEPT}_${SN_TAG}"

# Skip if already completed
if [ -f "$RESULTS_DIR/$RUN_TAG/val_loss.json" ]; then
    echo "[$RUN_TAG] Already completed, skipping."
    exit 0
fi

echo "Starting: $RUN_TAG"
torchrun --standalone --nproc_per_node=4 \
    -m scripts.affine_sweep_train -- \
    --affine-intercept "$AFFINE_INTERCEPT" \
    --sn-mode "$AFFINE_SN_MODE" \
    --results-dir "$RESULTS_DIR" \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --run dummy

echo "========================================"
echo "[$RUN_TAG] Job completed at: $(date)"
echo "========================================"
