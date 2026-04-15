#!/bin/bash
#SBATCH --job-name=tay1.0_d12
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=02:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/verify_taylor_eta1.0_d12_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/verify_taylor_eta1.0_d12_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

cd /data/scratch/medhaven/nanochat
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_tay1_0_d12_$$"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
    -m scripts.softmax_taylor_muon_train -- \
    --depth 12 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --eta 1.0 \
    --spectral-log-every -1 \
    --model-tag verify_taylor_eta1.0_d12 \
    --run dummy
