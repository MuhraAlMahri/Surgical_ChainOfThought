#!/bin/bash
# ============================================================================
# Submit Exp5 Evaluation Job
# ============================================================================
# This script submits the Exp5 evaluation job with dependency on training job 157094
# ============================================================================

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"
SLURM_SCRIPT="${BASE_DIR}/slurm/eval_exp5.slurm"
TRAINING_JOB_ID=157094

echo "=========================================="
echo "Submitting Exp5 Evaluation Job"
echo "=========================================="
echo "Training Job ID (dependency): ${TRAINING_JOB_ID}"
echo "Slurm Script: ${SLURM_SCRIPT}"
echo ""

# Check if training job exists
if ! squeue -j ${TRAINING_JOB_ID} &>/dev/null; then
    echo "⚠️  Warning: Training job ${TRAINING_JOB_ID} not found in queue."
    echo "   It may have already completed or been cancelled."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."
        exit 1
    fi
fi

# Submit evaluation job with dependency
echo "Submitting evaluation job..."
EVAL_JOB_ID=$(sbatch --dependency=afterok:${TRAINING_JOB_ID} ${SLURM_SCRIPT} | grep -oP '\d+')

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation job submitted successfully!"
    echo "  Job ID: ${EVAL_JOB_ID}"
    echo "  Dependency: afterok:${TRAINING_JOB_ID} (Exp5 training)"
    echo ""
    echo "The evaluation will start automatically after training completes."
    echo ""
    echo "Monitor with:"
    echo "  squeue -u muhra.almahri"
    echo "  tail -f ${BASE_DIR}/slurm/logs/eval_exp5_${EVAL_JOB_ID}.out"
else
    echo "✗ Failed to submit evaluation job"
    exit 1
fi



