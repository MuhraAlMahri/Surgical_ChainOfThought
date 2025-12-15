#!/bin/bash
# ============================================================================
# Submit Exp4 Stage 3 (after Stage 2 completes)
# ============================================================================

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"
SLURM_DIR="${BASE_DIR}/slurm"

# Get Stage 2 job ID from user or find it
if [ -z "$1" ]; then
    echo "Usage: $0 <STAGE2_JOB_ID>"
    echo ""
    echo "Or find Stage 2 job ID:"
    echo "  squeue -u \$USER | grep exp4_s2"
    echo ""
    echo "Then run:"
    echo "  $0 <STAGE2_JOB_ID>"
    exit 1
fi

STAGE2_JOB=$1

echo "Submitting Stage 3 (depends on Stage 2: ${STAGE2_JOB})..."
JOB3=$(sbatch --dependency=afterok:${STAGE2_JOB} ${SLURM_DIR}/train_exp4_stage3.slurm | grep -o '[0-9]*')

if [ -z "$JOB3" ]; then
    echo "❌ Failed to submit Stage 3"
    exit 1
fi

echo "✓ Stage 3: Job ${JOB3} (waits for ${STAGE2_JOB})"
echo ""
echo "Monitor: squeue -u \$USER"
echo "View log: tail -f ${SLURM_DIR}/logs/exp4_s3_${JOB3}.out"

