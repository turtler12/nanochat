#!/bin/bash
# =============================================================================
# Launch FloorMuon epsilon sweep in batches of 3 (1 GPU each).
#
# 5 configs total (Phase 1 sweep):
#   floor_e0p5   eps=0.5
#   floor_e1p0   eps=1.0
#   floor_e2p0   eps=2.0
#   floor_e5p0   eps=5.0
#   floor_e10p0  eps=10.0
#
# Usage: bash runs/launch_floor.sh [--batch <1|2>]
#   --batch 1  → submit eps=1,5,10  (default)
#   --batch 2  → submit eps=0.5,2.0
#   (no arg)   → submit all 5 at once
#
# Run from: /data/scratch/medhaven/nanochat
# =============================================================================

set -e

cd /data/scratch/medhaven/nanochat

BATCH="${1:-}"

BATCH1=(floor_e1p0 floor_e5p0 floor_e10p0)
BATCH2=(floor_e0p5 floor_e2p0)

submit_mode() {
    local MODE="$1"
    local JOB_ID
    JOB_ID=$(FLOOR_MODE="$MODE" \
        sbatch --parsable --job-name="floor_${MODE}" \
        runs/submit_floor.sh)
    echo "  [floor_${MODE}] Job $JOB_ID submitted"
}

if [ "$BATCH" = "--batch" ] && [ -n "$2" ]; then
    BATCH_NUM="$2"
elif [ -z "$BATCH" ]; then
    BATCH_NUM="all"
else
    echo "Usage: $0 [--batch <1|2>]"
    exit 1
fi

echo "Submitting FloorMuon epsilon sweep..."
echo ""

case "$BATCH_NUM" in
    1)
        echo "Batch 1/2 (${#BATCH1[@]} jobs):"
        for MODE in "${BATCH1[@]}"; do submit_mode "$MODE"; done
        echo ""
        echo "When done, run:  bash runs/launch_floor.sh --batch 2"
        ;;
    2)
        echo "Batch 2/2 (${#BATCH2[@]} jobs):"
        for MODE in "${BATCH2[@]}"; do submit_mode "$MODE"; done
        ;;
    all)
        echo "All 5 jobs at once:"
        for MODE in "${BATCH1[@]}" "${BATCH2[@]}"; do submit_mode "$MODE"; done
        ;;
    *)
        echo "Unknown batch '$BATCH_NUM'. Use 1, 2, or omit for all."
        exit 1
        ;;
esac

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results: /data/scratch/medhaven/nanochat_cache/base_checkpoints/floor_<mode>_d12/"
