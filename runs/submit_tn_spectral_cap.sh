#!/bin/bash
#
#SBATCH --job-name=tn_spec_cap
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/tn_spec_cap_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/tn_spec_cap_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#
# Option 2: Spectral normalization only (cap sigma_max at 1.5x init)

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: Trace Norm - Option 2 (Spectral Cap)"
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
    --norm-mode spectral-cap \
    --spectral-cap-ratio 1.5 \
    --spectral-log-every 250 \
    --model-tag tn_spectral-cap_d20 \
    --run dummy

echo "========================================"
echo "Option 2 (Spectral Cap) completed at: $(date)"
echo "========================================"
