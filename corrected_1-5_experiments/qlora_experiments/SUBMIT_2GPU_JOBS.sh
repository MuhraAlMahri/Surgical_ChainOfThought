#!/bin/bash
# ============================================================================
# Submit Stage 3 Training and Evaluations as Separate 2-GPU Jobs
# ============================================================================
# This splits the 4-GPU mega job into two 2-GPU jobs for better queue availability
# ============================================================================

cd "$(dirname "$0")"

echo "=========================================="
echo "Submitting 2-GPU Jobs (Split from 4-GPU Mega Job)"
echo "=========================================="
echo ""

# Check if job 156712 is still running (prerequisite)
PREV_JOB="156712"
if squeue -j ${PREV_JOB} 2>/dev/null | grep -q ${PREV_JOB}; then
    echo "⚠️  Job ${PREV_JOB} is still running"
    echo "   Stage 3 training depends on it completing"
    echo ""
    DEPENDENCY="--dependency=afterok:${PREV_JOB}"
else
    echo "✓ Job ${PREV_JOB} has completed (or not found)"
    echo ""
    DEPENDENCY=""
fi

# Submit Job 1: Stage 3 Training (2 GPUs)
echo "Submitting Job 1: Stage 3 Training (2 GPUs)..."
if [ -n "${DEPENDENCY}" ]; then
    TRAIN_JOB=$(sbatch --parsable ${DEPENDENCY} slurm/stage3_training_2gpu.slurm)
else
    TRAIN_JOB=$(sbatch --parsable slurm/stage3_training_2gpu.slurm)
fi
echo "✓ Submitted: ${TRAIN_JOB}"
echo "  - Exp3-S3 Training (GPU 0)"
echo "  - Exp4-S3 Training (GPU 1)"
echo ""

# Submit Job 2: Exp2 Evaluation (1 GPU, independent - no dependency!)
echo "Submitting Job 2: Exp2 Evaluation (1 GPU, independent)..."
EXP2_EVAL_JOB=$(sbatch --parsable slurm/exp2_eval_standalone_2gpu.slurm)
echo "✓ Submitted: ${EXP2_EVAL_JOB}"
echo "  - Exp2 Evaluation (independent, can run immediately)"
echo ""

# Submit Job 3: Exp3 and Exp4 Evaluations (2 GPUs) - depends on training
echo "Submitting Job 3: Exp3 and Exp4 Evaluations (2 GPUs)..."
EXP3_EXP4_EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} slurm/exp3_exp4_eval_2gpu.slurm)
echo "✓ Submitted: ${EXP3_EXP4_EVAL_JOB}"
echo "  - Exp3 Evaluation (GPU 0, after Exp3-S3 training)"
echo "  - Exp4 Evaluation (GPU 1, after Exp4-S3 training)"
echo ""

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Training Job: ${TRAIN_JOB} (2 GPUs)"
echo "  - Will start after job ${PREV_JOB} completes"
echo ""
echo "Exp2 Evaluation Job: ${EXP2_EVAL_JOB} (1 GPU)"
echo "  - Independent, can start immediately (no dependency!)"
echo ""
echo "Exp3/Exp4 Evaluation Job: ${EXP3_EXP4_EVAL_JOB} (2 GPUs)"
echo "  - Will start after training job ${TRAIN_JOB} completes"
echo ""
echo "Monitor jobs:"
echo "  squeue -j ${TRAIN_JOB},${EXP2_EVAL_JOB},${EXP3_EXP4_EVAL_JOB}"
echo ""
echo "View logs:"
echo "  tail -f slurm/logs/s3_train_2gpu_${TRAIN_JOB}.out"
echo "  tail -f slurm/logs/exp2_eval_standalone_${EXP2_EVAL_JOB}.out"
echo "  tail -f slurm/logs/exp3_exp4_eval_2gpu_${EXP3_EXP4_EVAL_JOB}.out"
echo "=========================================="

