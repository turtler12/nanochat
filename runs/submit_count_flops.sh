#!/bin/bash
#SBATCH --job-name=count_flops
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/count_flops_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/count_flops_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/data/scratch/medhaven/nanochat_cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_count_flops_$$"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found at $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
    exit 1
fi

DEPTH="${DEPTH:-12}"
FTRL_ETA="${FTRL_ETA:-0p3}"

echo "Counting optimizer FLOPs: Muon vs NS3+FTRL"
echo "  depth=$DEPTH  ftrl_eta=$FTRL_ETA"

python -m scripts.count_optimizer_flops \
    --depth "$DEPTH" \
    --ftrl-eta "$FTRL_ETA"

echo "========================================"
echo "Done at: $(date)"
echo "========================================"
