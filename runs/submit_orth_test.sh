#!/bin/bash
#
#SBATCH --job-name=orth_space_test
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:30:00
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/orth_test_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/orth_test_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "========================================"

nvidia-smi

cd /data/scratch/medhaven/nanochat

export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="/tmp"
export OMP_NUM_THREADS=1

source .venv/bin/activate

python orthogonal_space_test.py --beta 0.3

echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
