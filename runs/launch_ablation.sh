#!/bin/bash
# =============================================================================
# Launch 7 ablation jobs (one per update mode) to measure NS iteration quality.
# Mirrors launch_affine_chain.sh pattern exactly.
# Run from: /data/scratch/medhaven/nanochat
# =============================================================================

set -e

cd /data/scratch/medhaven/nanochat

MODES=(muon fast4 ns3 ns2 ns1 ns0_normalized ns0_sign)

echo "Submitting ablation sweep (${#MODES[@]} jobs)..."
echo ""

for MODE in "${MODES[@]}"; do
    JOB_ID=$(ABLATION_MODE="$MODE" \
        sbatch --parsable --job-name="abl_${MODE}" \
        runs/submit_ablation.sh)
    echo "  [abl_${MODE}] Job $JOB_ID submitted"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will be in: cache/base_checkpoints/ablation_<mode>_d12/"
