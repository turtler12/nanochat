#!/bin/bash
# Submit randomized SVD sweep: k=16 and k=32 in parallel, k=64 after k=16 finishes.
# Usage: bash runs/submit_rsvd_sweep.sh

echo "Submitting randomized SVD sweep jobs (2 parallel, then 1)..."

JID1=$(sbatch --parsable runs/submit_rsvd_k16.sh)
echo "  k=16: job $JID1"

JID2=$(sbatch --parsable runs/submit_rsvd_k32.sh)
echo "  k=32: job $JID2"

JID3=$(sbatch --parsable --dependency=afterany:$JID1 runs/submit_rsvd_k64.sh)
echo "  k=64: job $JID3 (after $JID1)"

echo "All jobs submitted. k=16 and k=32 run in parallel; k=64 runs after k=16."
