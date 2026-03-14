#!/bin/bash
#
#SBATCH --job-name=tn_grow_fro
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/tn_grow_fro_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/tn_grow_fro_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#
# Option 3: Growing Frobenius target (rate=0.001, max=3.5x)

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: Trace Norm - Option 3 (Growing Frobenius)"
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

torchrun --standalone --nproc_per_node=2 \
    -m scripts.trace_norm_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --ns-mode baseline \
    --norm-mode growing-fro \
    --growth-rate 0.001 \
    --growth-max-ratio 3.5 \
    --spectral-log-every 250 \
    --model-tag tn_growing-fro_d20 \
    --run dummy

echo "========================================"
echo "Option 3 (Growing Frobenius) completed at: $(date)"
echo "========================================"
