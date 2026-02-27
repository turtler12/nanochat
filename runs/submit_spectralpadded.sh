#!/bin/bash
#
#SBATCH --job-name=spectralpad_muon
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=06:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/spectralpad_muon_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/spectralpad_muon_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#
# Spectral-padded Muon experiment.
# Environment variables (set before sbatch, or passed via --export):
#   PAD_RATIO       - blend ratio gamma (0.0 = baseline, 0.1 = spectral padding)
#   POWER_ITERS     - power iterations for sigma_max (default: 1)
#   POWER_BATCH     - probe vectors per matrix (default: 4)
#   SV_LOG          - "1" to enable SV logging (default: 1)
#   SV_LOG_EVERY    - log every N steps (default: 500)

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: PAD_RATIO=${PAD_RATIO:-0.0}, POWER_ITERS=${POWER_ITERS:-1}, POWER_BATCH=${POWER_BATCH:-4}, SV_LOG=${SV_LOG:-1}"
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
export PAD_RATIO="${PAD_RATIO:-0.0}"
export POWER_ITERS="${POWER_ITERS:-1}"
export POWER_BATCH="${POWER_BATCH:-4}"
export SV_LOG="${SV_LOG:-1}"
export SV_LOG_EVERY="${SV_LOG_EVERY:-500}"

# Setup venv
command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

mkdir -p spectralpadded_results slurm_logs

if [ "$PAD_RATIO" = "0.0" ] || [ "$PAD_RATIO" = "0" ]; then
    TAG="baseline"
else
    TAG="spectralpad_gamma${PAD_RATIO}_pi${POWER_ITERS}_pb${POWER_BATCH}"
fi

# Skip if already completed
if [ -f "spectralpadded_results/$TAG/val_loss.json" ]; then
    echo "[$TAG] Already completed, skipping."
    exit 0
fi

echo "Starting: $TAG"
torchrun --standalone --nproc_per_node=4 \
    -m scripts.spectralpadded_muon_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --seed 42 \
    --results-dir spectralpadded_results \
    --run dummy

echo "========================================"
echo "[$TAG] Job completed at: $(date)"
echo "========================================"
