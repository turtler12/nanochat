#!/bin/bash
# =============================================================================
# Affine SVD Mapping Sweep for Muon Optimizer
# =============================================================================
# Tests the relative importance of spectral normalization (SN) vs the
# eigenvalue mapping step in the Muon optimizer.
#
# For each affine intercept i, runs two variants:
#   YES_SN: spectral normalization enabled (normalize by sigma_max)
#   NO_SN:  spectral normalization disabled (raw singular values)
#
# Designed for 4 H100 GPUs. Each run trains a model from scratch.
#
# Usage:
#   bash runs/affine_sweep.sh
#   # Or with wandb:
#   WANDB_RUN=affine_sweep bash runs/affine_sweep.sh
#   # Or in a screen session:
#   screen -L -Logfile runs/affine_sweep.log -S affine_sweep bash runs/affine_sweep.sh
# =============================================================================

set -e  # Exit on error

# Configuration
NUM_GPUS=4
DEPTH=20                      # Model depth (d20 is a good size for experiments)
DEVICE_BATCH_SIZE=16          # Per-GPU batch size (reduce if OOM)
EVAL_EVERY=250                # Evaluate val loss every N steps
TARGET_PARAM_DATA_RATIO=10.5  # Compute-optimal training horizon

# Affine intercept values to sweep
# i=0 is identity baseline, i=1 is full step function
INTERCEPTS="${INTERCEPTS:-0.0 0.1 0.2 0.398 0.5 0.734 0.84 0.9 1.0}"

# SN modes: "yes" = normalize by sigma_max, "no" = skip normalization
SN_MODES="${SN_MODES:-yes no}"

# Results directory (absolute path relative to project root)
RESULTS_DIR="$(pwd)/affine_sweep_results"
mkdir -p "$RESULTS_DIR"

# Wandb run name (default: dummy = no wandb logging)
WANDB_RUN="${WANDB_RUN:-dummy}"

# Environment setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$(pwd)/cache"
mkdir -p "$NANOCHAT_BASE_DIR"

# =============================================================================
# Python venv setup
# =============================================================================
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export TMPDIR="/tmp"

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# =============================================================================
# Ensure data is available (download if needed)
# =============================================================================
echo "Ensuring dataset is available..."
python -m nanochat.dataset -n 370 &
DATASET_PID=$!

# Train tokenizer if not already done
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer.model" ]; then
    echo "Training tokenizer..."
    # Need a small amount of data for tokenizer
    wait $DATASET_PID  # Wait for data first
    python -m scripts.tok_train
    python -m scripts.tok_eval
    # Re-launch data download (it completed above, but kept for safety)
else
    echo "Tokenizer already exists, skipping training."
fi

# Wait for full dataset
echo "Waiting for dataset download to complete..."
wait $DATASET_PID 2>/dev/null || true

# =============================================================================
# Run sweep
# =============================================================================

TOTAL_RUNS=0
COMPLETED_RUNS=0
FAILED_RUNS=0

# Count total runs
for intercept in $INTERCEPTS; do
    for sn_mode in $SN_MODES; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
done

echo "============================================================"
echo "AFFINE SVD MAPPING SWEEP"
echo "============================================================"
echo "Depth: $DEPTH"
echo "GPUs: $NUM_GPUS"
echo "Intercepts: $INTERCEPTS"
echo "SN modes: $SN_MODES"
echo "Total runs: $TOTAL_RUNS"
echo "Results dir: $RESULTS_DIR"
echo "============================================================"

# Log file for the entire sweep
SWEEP_LOG="$RESULTS_DIR/sweep_log.txt"
echo "Sweep started at $(date)" > "$SWEEP_LOG"
echo "Config: depth=$DEPTH, gpus=$NUM_GPUS" >> "$SWEEP_LOG"
echo "" >> "$SWEEP_LOG"

for sn_mode in $SN_MODES; do
    SN_TAG=$([ "$sn_mode" = "yes" ] && echo "YES_SN" || echo "NO_SN")
    echo ""
    echo "============================================================"
    echo "Running all intercepts with SN=$SN_TAG"
    echo "============================================================"

    for intercept in $INTERCEPTS; do
        RUN_TAG="affine_${intercept}_${SN_TAG}"
        RUN_RESULTS="$RESULTS_DIR/$RUN_TAG"

        # Skip if this run already completed successfully
        if [ -f "$RUN_RESULTS/val_loss.json" ]; then
            echo "[$RUN_TAG] Already completed, skipping."
            COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
            echo "$RUN_TAG: SKIPPED (already done)" >> "$SWEEP_LOG"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "[$RUN_TAG] Starting: intercept=$intercept, SN=$sn_mode"
        echo "------------------------------------------------------------"

        RUN_START=$(date +%s)

        # Run the training
        if torchrun --standalone --nproc_per_node=$NUM_GPUS \
            -m scripts.affine_sweep_train -- \
            --affine-intercept "$intercept" \
            --sn-mode "$sn_mode" \
            --results-dir "$RESULTS_DIR" \
            --depth "$DEPTH" \
            --device-batch-size "$DEVICE_BATCH_SIZE" \
            --eval-every "$EVAL_EVERY" \
            --target-param-data-ratio "$TARGET_PARAM_DATA_RATIO" \
            --fp8 \
            --run "$WANDB_RUN" \
            2>&1 | tee "$RESULTS_DIR/${RUN_TAG}.log"; then
            RUN_END=$(date +%s)
            RUN_DURATION=$(( (RUN_END - RUN_START) / 60 ))
            echo "[$RUN_TAG] Completed in ${RUN_DURATION}m"
            COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
            echo "$RUN_TAG: COMPLETED in ${RUN_DURATION}m" >> "$SWEEP_LOG"
        else
            RUN_END=$(date +%s)
            RUN_DURATION=$(( (RUN_END - RUN_START) / 60 ))
            echo "[$RUN_TAG] FAILED after ${RUN_DURATION}m"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            echo "$RUN_TAG: FAILED after ${RUN_DURATION}m" >> "$SWEEP_LOG"
            # Continue to next run instead of aborting the whole sweep
        fi
    done
done

# =============================================================================
# Summary and plotting
# =============================================================================

echo ""
echo "============================================================"
echo "SWEEP COMPLETE"
echo "============================================================"
echo "Total: $TOTAL_RUNS | Completed: $COMPLETED_RUNS | Failed: $FAILED_RUNS"
echo "Results saved in: $RESULTS_DIR"
echo "============================================================"

echo "" >> "$SWEEP_LOG"
echo "Sweep finished at $(date)" >> "$SWEEP_LOG"
echo "Total: $TOTAL_RUNS | Completed: $COMPLETED_RUNS | Failed: $FAILED_RUNS" >> "$SWEEP_LOG"

# Generate plots
echo "Generating plots..."
python -m scripts.affine_sweep_plot --results-dir "$RESULTS_DIR"

echo "Done! Check $RESULTS_DIR for results and plots."
