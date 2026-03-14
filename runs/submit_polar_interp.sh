#!/bin/bash
#
#SBATCH --job-name=pi_v2_b03
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=06:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/pi_v2_b03_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/pi_v2_b03_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#
# EXPERIMENT: Polar Interpolation v2 (β=0.3, 5 NS steps) at depth 20.
# v2: interpolation happens AFTER NorMuon, not before.
# g_final = (1-β)·g_muon + β·g_raw_scaled, renormalized to ||g_muon||

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Config: Polar Interp v2 β=0.3, NS_STEPS=5, depth=20"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
export UV_CACHE_DIR="$(pwd)/.uv_cache"

# Polar Interpolation v2 config
export BETA_INTERP=0.3
export NS_STEPS=5

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=4 \
    -m scripts.polar_interp_train -- \
    --depth 20 \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --model-tag pi_v2_b03_d20 \
    --run dummy

echo "========================================"
echo "Polar Interp v2 β=0.3 completed at: $(date)"
echo "========================================"
