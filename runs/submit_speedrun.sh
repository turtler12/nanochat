#!/bin/bash
#
#SBATCH --job-name=nanochat_gpt2
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=08:00:00 # 8 hours (training takes ~6 hours)
#SBATCH --output=/data/scratch/medhaven/nanochat/slurm_logs/job_%j.log
#SBATCH --error=/data/scratch/medhaven/nanochat/slurm_logs/job_%j.err
#SBATCH --gpus=4  # Request 4 H100 GPUs
#SBATCH --cpus-per-task=32  # More CPUs for data loading
#SBATCH --mem=256G  # Enough memory for 4 GPUs

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "========================================"

# Check GPUs
nvidia-smi

# Change to the project directory
cd /data/scratch/medhaven/nanochat

# Run the speedrun script
bash runs/speedrun_myrun.sh

echo "========================================"
echo "Job completed at: $(date)"
echo "========================================"
