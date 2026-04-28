#!/bin/bash
# =============================================================================
# Launch amortized NS ablation jobs in batches of 3 (1 GPU each).
#
# 7 configs total:
#   amort_2_e0   interval=2, eta=0.0  (plain cached Q)
#   amort_2_e01  interval=2, eta=0.1  (FTRL-corrected)
#   amort_3_e0   interval=3, eta=0.0
#   amort_3_e01  interval=3, eta=0.1
#   amort_4_e0   interval=4, eta=0.0
#   amort_4_e01  interval=4, eta=0.1
#   amort_8_e01  interval=8, eta=0.1
#
# Usage: bash runs/launch_amortized.sh [--batch <1|2|3>]
#   --batch 1  → submit first 3 jobs  (default)
#   --batch 2  → submit next 3 jobs
#   --batch 3  → submit final job
#   (no arg)   → submit all 7 at once (override if you have enough GPUs)
#
# Run from: /data/scratch/medhaven/nanochat
# =============================================================================

set -e

cd /data/scratch/medhaven/nanochat

BATCH="${1:-}"

BATCH1=(amort_2_e0 amort_2_e01 amort_3_e0)
BATCH2=(amort_3_e01 amort_4_e0 amort_4_e01)
BATCH3=(amort_8_e01)

submit_mode() {
    local MODE="$1"
    local JOB_ID
    JOB_ID=$(AMORT_MODE="$MODE" \
        sbatch --parsable --job-name="amort_${MODE}" \
        runs/submit_amortized.sh)
    echo "  [amort_${MODE}] Job $JOB_ID submitted"
}

if [ "$BATCH" = "--batch" ] && [ -n "$2" ]; then
    BATCH_NUM="$2"
elif [ -z "$BATCH" ]; then
    BATCH_NUM="all"
else
    echo "Usage: $0 [--batch <1|2|3>]"
    exit 1
fi

echo "Submitting amortized NS ablation jobs..."
echo ""

case "$BATCH_NUM" in
    1)
        echo "Batch 1/3 (${#BATCH1[@]} jobs):"
        for MODE in "${BATCH1[@]}"; do submit_mode "$MODE"; done
        echo ""
        echo "When done, run:  bash runs/launch_amortized.sh --batch 2"
        ;;
    2)
        echo "Batch 2/3 (${#BATCH2[@]} jobs):"
        for MODE in "${BATCH2[@]}"; do submit_mode "$MODE"; done
        echo ""
        echo "When done, run:  bash runs/launch_amortized.sh --batch 3"
        ;;
    3)
        echo "Batch 3/3 (${#BATCH3[@]} jobs):"
        for MODE in "${BATCH3[@]}"; do submit_mode "$MODE"; done
        ;;
    all)
        echo "All 7 jobs at once:"
        for MODE in "${BATCH1[@]}" "${BATCH2[@]}" "${BATCH3[@]}"; do submit_mode "$MODE"; done
        ;;
    *)
        echo "Unknown batch '$BATCH_NUM'. Use 1, 2, 3, or omit for all."
        exit 1
        ;;
esac

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results: /data/scratch/medhaven/nanochat_cache/base_checkpoints/amortized_<mode>_d12/"
