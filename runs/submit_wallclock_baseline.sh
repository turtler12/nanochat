#!/bin/bash
#
#SBATCH --job-name=wc_base
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=06:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/wallclock_baseline_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/wallclock_baseline_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#
# Fair wall-clock baseline: same 2 GPUs, exclusive node, matching Taylor run.

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: Baseline Muon (wallclock comparison, 2 GPUs)"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi
cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_wc_baseline_$$"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=2 --master_port=29500 \
    -m scripts.base_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --model-tag wallclock_baseline_d20 \
    --run dummy

echo "========================================"
echo "Baseline Muon completed at: $(date)"
echo "========================================"
