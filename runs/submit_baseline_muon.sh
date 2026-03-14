#!/bin/bash
#
#SBATCH --job-name=baseline_muon
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/baseline_muon_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/baseline_muon_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#
# Baseline Muon training with SVD spectrum logging
# Logs singular value spectra of weight matrices, gradients, and updates

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: Baseline Muon + SVD spectrum logging"
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
    -m scripts.baseline_muon_svd_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --ns-mode baseline \
    --svd-every 250 \
    --svd-top-k 64 \
    --model-tag baseline_muon \
    --run dummy

echo "========================================"
echo "Baseline Muon SVD training completed at: $(date)"
echo "========================================"
