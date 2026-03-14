#!/bin/bash
#
#SBATCH --job-name=tn_spec_pen
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/tn_spec_pen_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/tn_spec_pen_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#
# Option 4: Soft spectral penalty (lambda=0.01)

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: Trace Norm - Option 4 (Spectral Penalty)"
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
    --norm-mode spectral-penalty \
    --penalty-lambda 0.01 \
    --spectral-log-every 250 \
    --model-tag tn_spectral-penalty_d20 \
    --run dummy

echo "========================================"
echo "Option 4 (Spectral Penalty) completed at: $(date)"
echo "========================================"
