#!/bin/bash
# Script to submit evaluation jobs after training jobs complete
# This script uses SLURM dependencies to automatically start evaluation
# after the corresponding training job finishes

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
SLURM_DIR="${EXPERIMENTS_DIR}/slurm"

cd "${EXPERIMENTS_DIR}"

echo "=========================================="
echo "Submitting Evaluation Jobs"
echo "=========================================="
echo "This script will submit evaluation jobs that depend on training jobs"
echo ""

# Check current running jobs
echo "Current running jobs:"
squeue -u muhra.almahri --format="%.10i %.20j %.8T %.10M" | head -10
echo ""

# Get job IDs from user or from running jobs
echo "Please provide the training job IDs:"
echo "  - Main experiments training job ID (default: will auto-detect)"
echo "  - Instruction fine-tuning job ID (default: will auto-detect)"
echo ""

# Auto-detect job IDs from running jobs
MAIN_TRAIN_JOB=$(squeue -u muhra.almahri --format="%.10i %.20j" --noheader | grep -E "endovis_all|mega_job.*retrain" | head -1 | awk '{print $1}' || echo "")
INSTR_TRAIN_JOB=$(squeue -u muhra.almahri --format="%.10i %.20j" --noheader | grep -E "instr_ft|instruction_finetuning" | head -1 | awk '{print $1}' || echo "")

if [ -z "$MAIN_TRAIN_JOB" ]; then
    read -p "Main experiments training job ID: " MAIN_TRAIN_JOB
fi

if [ -z "$INSTR_TRAIN_JOB" ]; then
    read -p "Instruction fine-tuning job ID: " INSTR_TRAIN_JOB
fi

if [ -z "$MAIN_TRAIN_JOB" ] && [ -z "$INSTR_TRAIN_JOB" ]; then
    echo "Error: No job IDs provided"
    exit 1
fi

echo ""
echo "=========================================="
echo "Submitting Evaluation Jobs"
echo "=========================================="

# Submit main experiments evaluation (depends on main training)
if [ -n "$MAIN_TRAIN_JOB" ]; then
    echo ""
    echo "Submitting main experiments evaluation (depends on job ${MAIN_TRAIN_JOB})..."
    MAIN_EVAL_JOB=$(sbatch --dependency=afterok:${MAIN_TRAIN_JOB} \
        --job-name=eval_main \
        "${SLURM_DIR}/mega_job_evaluate_all.slurm" | awk '{print $4}')
    
    if [ -n "$MAIN_EVAL_JOB" ]; then
        echo "✓ Main experiments evaluation submitted: Job ${MAIN_EVAL_JOB}"
        echo "  Will start after training job ${MAIN_TRAIN_JOB} completes"
    else
        echo "✗ Failed to submit main experiments evaluation"
    fi
fi

# Submit instruction fine-tuning evaluation (depends on instruction fine-tuning training)
if [ -n "$INSTR_TRAIN_JOB" ]; then
    echo ""
    echo "Submitting instruction fine-tuning evaluation (depends on job ${INSTR_TRAIN_JOB})..."
    INSTR_EVAL_JOB=$(sbatch --dependency=afterok:${INSTR_TRAIN_JOB} \
        --job-name=eval_instr_ft \
        "${SLURM_DIR}/mega_job_evaluate_instruction_finetuning.slurm" | awk '{print $4}')
    
    if [ -n "$INSTR_EVAL_JOB" ]; then
        echo "✓ Instruction fine-tuning evaluation submitted: Job ${INSTR_EVAL_JOB}"
        echo "  Will start after training job ${INSTR_TRAIN_JOB} completes"
    else
        echo "✗ Failed to submit instruction fine-tuning evaluation"
    fi
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
if [ -n "$MAIN_EVAL_JOB" ]; then
    echo "Main experiments evaluation: Job ${MAIN_EVAL_JOB} (depends on ${MAIN_TRAIN_JOB})"
fi
if [ -n "$INSTR_EVAL_JOB" ]; then
    echo "Instruction fine-tuning evaluation: Job ${INSTR_EVAL_JOB} (depends on ${INSTR_TRAIN_JOB})"
fi
echo ""
echo "You can monitor jobs with: squeue -u muhra.almahri"
echo "=========================================="

exit 0











