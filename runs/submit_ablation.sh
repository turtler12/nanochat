#!/bin/bash
#
#SBATCH --job-name=ablation
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=03:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/abl_%x_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/abl_%x_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=ALL
#
# Single ablation run. Expects env var:
#   ABLATION_MODE  - e.g. "muon", "ns1", "ns0_normalized"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Mode: $ABLATION_MODE"
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

# Fail fast if tokenizer or data is missing rather than training on empty data
if [ ! -f "cache/tokenizer/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found at cache/tokenizer/tokenizer.pkl"
    echo "Run setup first: python -m nanochat.dataset -n 370 && python -m scripts.tok_train"
    exit 1
fi

echo "Starting ablation mode: $ABLATION_MODE"
torchrun --standalone --nproc_per_node=1 \
    -m scripts.ablation_train -- \
    --update-mode "$ABLATION_MODE" \
    --depth 12 \
    --num-iterations 2205 \
    --matrix-lr 0.02 \
    --run dummy

echo "========================================"
echo "[$ABLATION_MODE] Job completed at: $(date)"
echo "========================================"
