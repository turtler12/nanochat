#!/bin/bash
#
#SBATCH --job-name=medrank
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/medrank_k64_eta0.1_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/medrank_k64_eta0.1_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: MedRankSoftmaxMuon k=64 eta=0.1 (Path 2: medium-rank SVD)"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi
cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_medrank_$$"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
    -m scripts.medrank_softmax_muon_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --eta 0.1 \
    --rank-k 64 \
    --spectral-log-every 100 \
    --model-tag medrank_k64_eta0.1_d20 \
    --run dummy

echo "========================================"
echo "MedRankSoftmaxMuon completed at: $(date)"
echo "========================================"
