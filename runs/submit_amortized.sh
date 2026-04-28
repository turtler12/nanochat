#!/bin/bash
#
#SBATCH --job-name=amortized
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=03:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/amort_%x_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/amort_%x_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=ALL
#
# Single amortized NS run. Expects env var:
#   AMORT_MODE  - e.g. "amort_4_e01"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Mode: $AMORT_MODE"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/data/scratch/medhaven/nanochat_cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"

git pull --ff-only

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found at $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
    exit 1
fi

MATRIX_LR="${MATRIX_LR:-0.02}"

echo "Starting amortized mode: $AMORT_MODE (matrix-lr=$MATRIX_LR)"
torchrun --standalone --nproc_per_node=1 \
    -m scripts.amortized_train -- \
    --update-mode "$AMORT_MODE" \
    --depth 12 \
    --num-iterations 2205 \
    --matrix-lr "$MATRIX_LR" \
    --run dummy

echo "========================================"
echo "[$AMORT_MODE] Job completed at: $(date)"
echo "========================================"
