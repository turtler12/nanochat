#!/bin/bash
#
#SBATCH --job-name=diag_muon
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/diag_muon_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/diag_muon_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#
# Spectral diagnostic: Muon optimizer baseline

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: Spectral Diagnostic - Muon"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_diag_muon_$$"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
    -m scripts.spectral_diagnostic_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --optimizer muon \
    --diag-every 50 \
    --diag-layer 5 \
    --model-tag diag_muon_d20 \
    --run dummy

echo "========================================"
echo "Spectral Diagnostic (Muon) completed at: $(date)"
echo "========================================"
