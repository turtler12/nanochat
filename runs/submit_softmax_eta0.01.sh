#!/bin/bash
#
#SBATCH --job-name=sm_eta0.01
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/sm_eta0.01_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/sm_eta0.01_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#
# SoftmaxMuon eta=0.01 (near-uniform, Muon-like)

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: SoftmaxMuon eta=0.01"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_sm_eta001_$$"
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
    -m scripts.softmax_muon_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --eta 0.01 \
    --spectral-log-every 100 \
    --model-tag softmax_eta0.01_d20 \
    --run dummy

echo "========================================"
echo "SoftmaxMuon eta=0.01 completed at: $(date)"
echo "========================================"
