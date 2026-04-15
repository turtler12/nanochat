#!/bin/bash
#
#SBATCH --job-name=csvd_f1
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=01:50:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/csvd_freq1_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/csvd_freq1_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: Cached-SVD SoftmaxMuon svd_freq=1 (no caching)"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi
cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_csvd_f1_$$"
export NCCL_P2P_LEVEL=NVL

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

for attempt in 1 2 3; do
    echo "--- Attempt $attempt ($(date)) ---"
    torchrun --standalone --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
        -m scripts.cached_softmax_muon_train -- \
        --depth 20 \
        --device-batch-size 16 \
        --eval-every 250 \
        --target-param-data-ratio 10.5 \
        --fp8 \
        --eta 0.5 \
        --svd-freq 1 \
        --spectral-log-every 100 \
        --model-tag cached_svd_freq1_d20 \
        --run dummy
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Completed successfully on attempt $attempt"
        break
    fi
    echo "Attempt $attempt failed (exit $exit_code), retrying in 10s..."
    sleep 10
done

echo "========================================"
echo "svd_freq=1 completed at: $(date)"
echo "========================================"
