#!/bin/bash
#
#SBATCH --job-name=profile_d12
#SBATCH --account=lingo
#SBATCH --partition=vision-shared-h100
#SBATCH --qos=shared-if-available
#SBATCH --time=00:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/profile_d12_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/profile_d12_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#
# Profile training step breakdown for d12 model (4 GPU)

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: profile d12, 4 GPU"
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
    -m scripts.profile_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --total-batch-size 524288 \
    --num-profile-steps 50 \
    --warmup-steps 15

echo "========================================"
echo "Profile completed at: $(date)"
echo "========================================"
