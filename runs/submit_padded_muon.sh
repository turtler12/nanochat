#!/bin/bash
#
#SBATCH --job-name=padded_muon
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=06:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/padded_muon_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/padded_muon_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#
# Runs baseline + padded Muon comparison.
# Environment variables (set before sbatch, or passed via --export):
#   MUON_PAD        - "0" for baseline, "1" for padded (default: 0)
#   MUON_PAD_ALPHA  - padding strength (default: 0.03)
#   MUON_SV_LOG     - "1" to enable SV logging (default: 1)
#   MUON_SV_LOG_EVERY - log every N steps (default: 500)

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: MUON_PAD=${MUON_PAD:-0}, MUON_PAD_ALPHA=${MUON_PAD_ALPHA:-0.03}"
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

# Default env vars
export MUON_PAD="${MUON_PAD:-0}"
export MUON_PAD_ALPHA="${MUON_PAD_ALPHA:-0.03}"
export MUON_SV_LOG="${MUON_SV_LOG:-1}"
export MUON_SV_LOG_EVERY="${MUON_SV_LOG_EVERY:-500}"

# Setup venv
command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

mkdir -p padded_muon_results slurm_logs

TAG=$([ "$MUON_PAD" = "1" ] && echo "padded_alpha${MUON_PAD_ALPHA}" || echo "baseline")

# Skip if already completed
if [ -f "padded_muon_results/$TAG/val_loss.json" ]; then
    echo "[$TAG] Already completed, skipping."
    exit 0
fi

echo "Starting: $TAG"
torchrun --standalone --nproc_per_node=4 \
    -m scripts.padded_muon_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --seed 42 \
    --results-dir padded_muon_results \
    --run dummy

echo "========================================"
echo "[$TAG] Job completed at: $(date)"
echo "========================================"
