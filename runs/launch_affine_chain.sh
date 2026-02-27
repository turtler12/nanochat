#!/bin/bash
# =============================================================================
# Launch 6 sequential SLURM jobs for the affine sweep experiment.
# Each job runs one (intercept, SN) configuration on 4 H100s for up to 6.5h.
# Jobs are chained with --dependency=afterany so they run one after another.
# =============================================================================

set -e

cd /data/scratch/medhaven/nanochat

CONFIGS=(
    "0.0 yes"
    "0.0 no"
    "0.5 yes"
    "0.5 no"
    "1.0 yes"
    "1.0 no"
)

PREV_JOB=""

echo "Submitting affine sweep chain (${#CONFIGS[@]} jobs)..."
echo ""

for config in "${CONFIGS[@]}"; do
    read -r intercept sn_mode <<< "$config"
    SN_TAG=$([ "$sn_mode" = "yes" ] && echo "YES_SN" || echo "NO_SN")
    JOB_NAME="affine_${intercept}_${SN_TAG}"

    if [ -z "$PREV_JOB" ]; then
        JOB_ID=$(AFFINE_INTERCEPT="$intercept" AFFINE_SN_MODE="$sn_mode" \
            sbatch --parsable --job-name="$JOB_NAME" \
            runs/submit_affine_single.sh)
    else
        JOB_ID=$(AFFINE_INTERCEPT="$intercept" AFFINE_SN_MODE="$sn_mode" \
            sbatch --parsable --job-name="$JOB_NAME" \
            --dependency=afterany:$PREV_JOB \
            runs/submit_affine_single.sh)
    fi

    echo "  [$JOB_NAME] Job $JOB_ID submitted$([ -n "$PREV_JOB" ] && echo " (after $PREV_JOB)")"
    PREV_JOB=$JOB_ID
done

# Final plotting job after all training completes
PLOT_JOB=$(sbatch --parsable \
    --dependency=afterany:$PREV_JOB \
    --job-name="affine_plot" \
    --account=lingo \
    --partition=lingo-h100 \
    --qos=lingo-main \
    --time=00:10:00 \
    --gpus=1 \
    --cpus-per-task=4 \
    --mem=16G \
    --output=/data/scratch/medhaven/nanochat/slurm_logs/affine_plot_%j.log \
    --error=/data/scratch/medhaven/nanochat/slurm_logs/affine_plot_%j.err \
    --wrap="cd /data/scratch/medhaven/nanochat && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=\$(pwd)/.uv_cache && export TMPDIR=/tmp && source .venv/bin/activate && python -m scripts.affine_sweep_plot --results-dir affine_sweep_results")

echo ""
echo "  [affine_plot] Job $PLOT_JOB submitted (after $PREV_JOB)"
echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will be in: affine_sweep_results/"
