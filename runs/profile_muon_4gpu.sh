#!/bin/bash
#SBATCH --job-name=prof_muon4
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/profile_muon_4gpu_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/profile_muon_4gpu_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

cd /data/scratch/medhaven/nanochat
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_prof_muon4_$$"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=4 --master_port=$((29500 + RANDOM % 1000)) \
    -m scripts.profile_step_timing -- \
    --method muon \
    --depth 12 \
    --device-batch-size 16 \
    --fp8
