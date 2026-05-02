#!/bin/bash
#
#SBATCH --job-name=clamp_sv_d12
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=06:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/clamp_sv_d12_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/clamp_sv_d12_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Experiment: clamp_sv d12 (clamp SVs below 0.1, random in [0.1, 1.0] above)"
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

RESULTS_DIR="$(pwd)/clamp_sv_results"
mkdir -p "$RESULTS_DIR"

torchrun --standalone --nproc_per_node=4 \
    -m scripts.clamp_sv_train -- \
    --depth 12 \
    --rand-lo 0.1 \
    --rand-hi 1.0 \
    --clamp-below \
    --results-dir "$RESULTS_DIR" \
    --device-batch-size 16 \
    --eval-every 250 \
    --target-param-data-ratio 10.5 \
    --fp8 \
    --run dummy

echo "========================================"
echo "clamp_sv_d12 completed at: $(date)"
echo "========================================"
