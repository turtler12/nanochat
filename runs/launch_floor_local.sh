#!/bin/bash
# =============================================================================
# Launch FloorMuon epsilon sweep directly on a single-node multi-GPU machine.
# Runs 3 jobs in parallel, one per GPU, no SLURM needed.
#
# Usage: bash runs/launch_floor_local.sh [--batch <1|2>]
#   --batch 1  → eps=1.0, 5.0, 10.0  on GPUs 0,1,2  (default)
#   --batch 2  → eps=0.5, 2.0        on GPUs 0,1
#   (no arg)   → batch 1
#
# Run from: /workspace/nanochat (or any directory containing the repo)
# =============================================================================

set -e

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/nanochat_cache}"
MATRIX_LR="${MATRIX_LR:-0.02}"

BATCH1=(floor_e1p0 floor_e5p0 floor_e10p0)
BATCH2=(floor_e0p5 floor_e2p0)

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found at $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
    exit 1
fi

mkdir -p slurm_logs

run_mode() {
    local MODE="$1"
    local GPU="$2"
    local LOG="slurm_logs/floor_${MODE}_$(date +%s).log"
    echo "  [$MODE] starting on GPU $GPU → $LOG"
    CUDA_VISIBLE_DEVICES="$GPU" \
    FLOOR_MODE="$MODE" \
        torchrun --standalone --nproc_per_node=1 \
            -m scripts.floor_train -- \
            --update-mode "$MODE" \
            --depth 12 \
            --num-iterations 2205 \
            --matrix-lr "$MATRIX_LR" \
            --run dummy \
        > "$LOG" 2>&1 &
    echo $!
}

BATCH_NUM="${2:-1}"
if [ "$1" = "--batch" ] && [ -n "$2" ]; then
    BATCH_NUM="$2"
elif [ -z "$1" ]; then
    BATCH_NUM="1"
fi

case "$BATCH_NUM" in
    1) MODES=("${BATCH1[@]}") ;;
    2) MODES=("${BATCH2[@]}") ;;
    *) echo "Usage: $0 [--batch <1|2>]"; exit 1 ;;
esac

echo "Launching FloorMuon batch $BATCH_NUM (${#MODES[@]} runs)..."
echo ""

PIDS=()
for i in "${!MODES[@]}"; do
    PID=$(run_mode "${MODES[$i]}" "$i")
    PIDS+=("$PID")
done

echo ""
echo "All runs started. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  [${MODES[$i]}] done"
    else
        echo "  [${MODES[$i]}] FAILED (exit $?)"
        FAILED=1
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "All runs completed successfully."
else
    echo ""
    echo "One or more runs failed — check slurm_logs/floor_*.log"
    exit 1
fi
