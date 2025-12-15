#!/bin/bash
# ============================================================================
# Simple Exp4 Submission - Separate Jobs with Dependencies
# ============================================================================
# This submits 3 separate jobs with proper dependencies
# Each job has its own time limit (avoids QOS limit issues)
# ============================================================================

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"
SLURM_DIR="${BASE_DIR}/slurm"

echo "=========================================="
echo "SUBMITTING EXP4 CURRICULUM LEARNING"
echo "=========================================="
echo ""

# Stage 1
echo "Submitting Stage 1..."
JOB1=$(sbatch ${SLURM_DIR}/train_exp4_stage1.slurm | grep -o '[0-9]*')
if [ -z "$JOB1" ]; then
    echo "❌ Failed to submit Stage 1"
    exit 1
fi
echo "✓ Stage 1: Job ${JOB1}"
echo ""

# Stage 2 (depends on Stage 1)
echo "Submitting Stage 2 (depends on Stage 1: ${JOB1})..."
JOB2=$(sbatch --dependency=afterok:${JOB1} ${SLURM_DIR}/train_exp4_stage2.slurm | grep -o '[0-9]*')
if [ -z "$JOB2" ]; then
    echo "❌ Failed to submit Stage 2"
    exit 1
fi
echo "✓ Stage 2: Job ${JOB2} (waits for ${JOB1})"
echo ""

# Stage 3 (depends on Stage 2)
echo "Submitting Stage 3 (depends on Stage 2: ${JOB2})..."
JOB3=$(sbatch --dependency=afterok:${JOB2} ${SLURM_DIR}/train_exp4_stage3.slurm | grep -o '[0-9]*')
if [ -z "$JOB3" ]; then
    echo "❌ Failed to submit Stage 3"
    exit 1
fi
echo "✓ Stage 3: Job ${JOB3} (waits for ${JOB2})"
echo ""

echo "=========================================="
echo "ALL JOBS SUBMITTED SUCCESSFULLY"
echo "=========================================="
echo "Stage 1: Job ${JOB1} (~7 hours)"
echo "Stage 2: Job ${JOB2} (~12 hours, waits for ${JOB1})"
echo "Stage 3: Job ${JOB3} (~30 min, waits for ${JOB2})"
echo ""
echo "Total estimated time: ~20 hours"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f ${SLURM_DIR}/logs/exp4_s1_${JOB1}.out"
echo "  tail -f ${SLURM_DIR}/logs/exp4_s2_${JOB2}.out"
echo "  tail -f ${SLURM_DIR}/logs/exp4_s3_${JOB3}.out"
echo ""
echo "Cancel all:"
echo "  scancel ${JOB1} ${JOB2} ${JOB3}"
echo "=========================================="

