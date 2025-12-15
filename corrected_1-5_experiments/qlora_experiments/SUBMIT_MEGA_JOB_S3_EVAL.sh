#!/bin/bash
# Submit the mega job that runs Stage 3 training + evaluations on 4 GPUs
# Usage: bash SUBMIT_MEGA_JOB_S3_EVAL.sh

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"

echo "========================================================================"
echo "SUBMITTING MEGA JOB: Stage 3 Training + Evaluations (4 GPUs)"
echo "========================================================================"
echo ""
echo "This job will run:"
echo "  GPU 0: Exp3-S3 - Training (CXRTrek Stage 3)"
echo "  GPU 1: Exp4-S3 - Training (Curriculum Stage 3)"
echo "  GPU 2: Exp2 - Evaluation (re-run with fixed script, independent)"
echo "  GPU 2/3: Exp3 - Evaluation (after Exp3-S3 completes)"
echo "  GPU 3: Exp4 - Evaluation (after Exp4-S3 completes)"
echo ""
echo "NOTE: This job will wait for Exp3-S2 and Exp4-S2 (from job 156712) to complete"
echo ""

# Check if job 156712 (Exp3-S2 and Exp4-S2) is still running
PREV_JOB="156712"

PREV_JOB_STATUS=$(squeue -j ${PREV_JOB} --format="%T" --noheader 2>/dev/null || echo "NOT_FOUND")

if [ "${PREV_JOB_STATUS}" != "NOT_FOUND" ]; then
    echo "Current status of prerequisite job:"
    echo "  Job ${PREV_JOB} (Exp3-S2 & Exp4-S2): ${PREV_JOB_STATUS}"
    echo ""
    echo "The mega job will be submitted with dependency: afterok:${PREV_JOB}"
    echo "It will start automatically after the job completes."
    echo ""
    DEPENDENCY="--dependency=afterok:${PREV_JOB}"
else
    echo "Prerequisite job ${PREV_JOB} appears to be completed or not found."
    echo "Checking if required checkpoints exist..."
    echo ""
    
    EXP3_S2_CHECKPOINT="${BASE_DIR}/models/exp3_cxrtrek/stage2"
    EXP4_S2_CHECKPOINT="${BASE_DIR}/models/exp4_curriculum/stage2"
    
    if [ ! -d "${EXP3_S2_CHECKPOINT}" ]; then
        echo "WARNING: Exp3-S2 checkpoint not found at ${EXP3_S2_CHECKPOINT}"
    else
        echo "✓ Exp3-S2 checkpoint exists"
    fi
    
    if [ ! -d "${EXP4_S2_CHECKPOINT}" ]; then
        echo "ERROR: Exp4-S2 checkpoint not found at ${EXP4_S2_CHECKPOINT}"
        echo "Cannot proceed without Exp4-S2 checkpoint!"
        exit 1
    else
        echo "✓ Exp4-S2 checkpoint exists"
    fi
    
    echo ""
    echo "Submitting mega job without dependency..."
    echo ""
    DEPENDENCY=""
fi

cd ${BASE_DIR}

echo ">>> Submitting Mega Job (Stage 3 + Evaluations)..."
if [ -n "${DEPENDENCY}" ]; then
    MEGA_JOB=$(sbatch --parsable ${DEPENDENCY} slurm/mega_job_stage3_and_eval.slurm)
else
    MEGA_JOB=$(sbatch --parsable slurm/mega_job_stage3_and_eval.slurm)
fi
echo "✓ Submitted Mega Job: ${MEGA_JOB}"
echo ""

echo "========================================================================"
echo "✓ MEGA JOB SUBMITTED!"
echo "========================================================================"
echo ""
echo "Job ID: ${MEGA_JOB}"
echo "  - 4 GPUs running in parallel"
echo "  - 2 Training jobs (Exp3-S3, Exp4-S3)"
echo "  - 3 Evaluation jobs (Exp2 re-run, Exp3, Exp4)"
if [ -n "${DEPENDENCY}" ]; then
    echo "  - Dependency: afterok:${PREV_JOB}"
fi
echo ""
echo "Monitor job:"
echo "  squeue -j ${MEGA_JOB}"
echo "  tail -f slurm/logs/mega_s3_eval_${MEGA_JOB}.out"
echo "  tail -f slurm/logs/mega_s3_gpu*_${MEGA_JOB}.out  # Individual GPU logs"
echo ""
echo "Results will be saved to:"
echo "  results/exp2_evaluation.json (re-run with fixed script)"
echo "  results/exp3_evaluation.json"
echo "  results/exp4_evaluation.json"
echo "========================================================================"

