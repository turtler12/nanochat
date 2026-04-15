#!/bin/bash
#
#SBATCH --job-name=wallclock
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=12:00:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/wallclock_comparison_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/wallclock_comparison_%j.err
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#
# Fair wall-clock comparison: baseline Muon vs TaylorMuon eta=0.5
# Both run sequentially on the SAME exclusive node with identical GPU count.

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: Wall-clock comparison (baseline then Taylor eta=0.5)"
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

# =========================================================================
# Run 1: Baseline Muon
# =========================================================================
echo ""
echo "========================================"
echo "Starting BASELINE MUON at: $(date)"
echo "========================================"

export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_wc_baseline_$$"

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

# =========================================================================
# Run 2: TaylorMuon eta=0.5
# =========================================================================
echo ""
echo "========================================"
echo "Starting TAYLOR MUON eta=0.5 at: $(date)"
echo "========================================"

export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_wc_taylor_$$"

torchrun --standalone --nproc_per_node=2 --master_port=29501 \
    -m scripts.softmax_taylor_muon_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --eta 0.5 \
    --spectral-log-every 100 \
    --model-tag wallclock_taylor_eta0.5_d20 \
    --run dummy

echo "========================================"
echo "TaylorMuon eta=0.5 completed at: $(date)"
echo "========================================"
echo ""
echo "Both runs complete. Results in cache/base_checkpoints/wallclock_*"
