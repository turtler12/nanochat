#!/bin/bash
# =============================================================================
# Run the optimizer FLOPs comparison: Muon vs NS3+FTRL (single GPU, local).
#
# Usage:
#   bash runs/count_flops_local.sh               # depth 12, eta=0.3
#   bash runs/count_flops_local.sh --depth 20    # depth 20
#   CUDA_VISIBLE_DEVICES=1 bash runs/count_flops_local.sh
#
# Run from: repo root (nanochat/)
# =============================================================================

set -e
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/nanochat_cache}"

DEPTH="${DEPTH:-12}"
FTRL_ETA="${FTRL_ETA:-0p3}"

# Allow passing --depth N as an argument
while [[ $# -gt 0 ]]; do
    case "$1" in
        --depth) DEPTH="$2"; shift 2 ;;
        --ftrl-eta) FTRL_ETA="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LOG="slurm_logs/count_flops_d${DEPTH}_$(date +%s).log"
mkdir -p slurm_logs

echo "Running FLOPs comparison: Muon vs NS3+FTRL"
echo "  depth=${DEPTH}, ftrl_eta=${FTRL_ETA}"
echo "  log -> ${LOG}"
echo ""

python -m scripts.count_optimizer_flops \
    --depth "$DEPTH" \
    --ftrl-eta "$FTRL_ETA" \
    2>&1 | tee "$LOG"

echo ""
echo "Done. Full log saved to: $LOG"
