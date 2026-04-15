#!/bin/bash
#
#SBATCH --job-name=wsm_0.05
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=18:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/wsoftmax_eta0.05_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/wsoftmax_eta0.05_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: WeightSoftmaxMuon eta=0.05"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi
cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_wsm__$$"
export NCCL_P2P_LEVEL=NVL

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

for attempt in 1 2 3; do
    echo "--- Attempt $attempt ($(date)) ---"
    torchrun --standalone --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
        -m scripts.weight_softmax_muon_train -- \
        --depth 20 \
        --device-batch-size 16 \
        --eval-every 250 \
        --target-param-data-ratio 10.5 \
        --fp8 \
        --eta 0.05 \
        --spectral-log-every 100 \
        --model-tag wsoftmax_eta0.05_d20 \
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
echo "WeightSoftmaxMuon eta=0.05 completed at: $(date)"
echo "========================================"
