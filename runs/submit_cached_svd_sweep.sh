#!/bin/bash
# Submit all cached-SVD sweep jobs with dependencies so they run sequentially.
# This prevents GPU memory corruption from overlapping jobs on the same node.
# Usage: bash runs/submit_cached_svd_sweep.sh

echo "Submitting cached-SVD sweep jobs (sequential via dependencies)..."

JID1=$(sbatch --parsable runs/submit_csvd_baseline.sh)
echo "  Baseline: job $JID1"

JID2=$(sbatch --parsable --dependency=afterany:$JID1 runs/submit_csvd_freq1.sh)
echo "  svd_freq=1: job $JID2 (after $JID1)"

JID3=$(sbatch --parsable --dependency=afterany:$JID2 runs/submit_csvd_freq5.sh)
echo "  svd_freq=5: job $JID3 (after $JID2)"

JID4=$(sbatch --parsable --dependency=afterany:$JID3 runs/submit_csvd_freq10.sh)
echo "  svd_freq=10: job $JID4 (after $JID3)"

JID5=$(sbatch --parsable --dependency=afterany:$JID4 runs/submit_csvd_freq20.sh)
echo "  svd_freq=20: job $JID5 (after $JID4)"

echo "All jobs submitted. They will run sequentially."
