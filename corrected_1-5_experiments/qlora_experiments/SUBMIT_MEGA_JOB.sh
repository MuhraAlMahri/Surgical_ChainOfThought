#!/bin/bash
# Submit the mega job that runs 4 experiments in parallel on 4 GPUs
# Usage: bash SUBMIT_MEGA_JOB.sh

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"

echo "========================================================================"
echo "SUBMITTING MEGA JOB (4 GPUs: 2 Evaluations + 2 Training)"
echo "========================================================================"
echo ""
echo "This job will run:"
echo "  GPU 0: Exp1 - Evaluation (Random Baseline)"
echo "  GPU 1: Exp2 - Evaluation (Qwen Reordered)"
echo "  GPU 2: Exp3-S2 - Training (CXRTrek Stage 2)"
echo "  GPU 3: Exp4-S2 - Training (Curriculum Stage 2)"
echo ""
echo "NOTE: This job will wait for Exp3-S1 (156143_2) and Exp4-S1 (156143_3) to complete"
echo ""

# Check if Exp3-S1 or Exp4-S1 are still running
EXP3_S1_JOB="156143_2"
EXP4_S1_JOB="156143_3"

EXP3_S1_STATUS=$(squeue -j ${EXP3_S1_JOB} --format="%T" --noheader 2>/dev/null || echo "NOT_FOUND")
EXP4_S1_STATUS=$(squeue -j ${EXP4_S1_JOB} --format="%T" --noheader 2>/dev/null || echo "NOT_FOUND")

if [ "${EXP3_S1_STATUS}" != "NOT_FOUND" ] || [ "${EXP4_S1_STATUS}" != "NOT_FOUND" ]; then
    echo "Current status of prerequisite jobs:"
    echo "  Exp3-S1 (${EXP3_S1_JOB}): ${EXP3_S1_STATUS}"
    echo "  Exp4-S1 (${EXP4_S1_JOB}): ${EXP4_S1_STATUS}"
    echo ""
    echo "The mega job will be submitted with dependency: afterok:${EXP3_S1_JOB}:${EXP4_S1_JOB}"
    echo "It will start automatically after both jobs complete."
    echo ""
    DEPENDENCY="--dependency=afterok:${EXP3_S1_JOB}:${EXP4_S1_JOB}"
else
    echo "Prerequisite jobs appear to be completed or not found."
    echo "Submitting mega job without dependency..."
    echo ""
    DEPENDENCY=""
fi

cd ${BASE_DIR}

echo ">>> Submitting Mega Job..."
if [ -n "${DEPENDENCY}" ]; then
    MEGA_JOB=$(sbatch --parsable ${DEPENDENCY} slurm/mega_job_eval_and_train.slurm)
else
    MEGA_JOB=$(sbatch --parsable slurm/mega_job_eval_and_train.slurm)
fi
echo "✓ Submitted Mega Job: ${MEGA_JOB}"
echo ""

echo "========================================================================"
echo "✓ MEGA JOB SUBMITTED!"
echo "========================================================================"
echo ""
echo "Job ID: ${MEGA_JOB}"
echo "  - 4 GPUs running in parallel"
echo "  - 2 Evaluations (Exp1, Exp2)"
echo "  - 2 Training jobs (Exp3-S2, Exp4-S2)"
if [ -n "${DEPENDENCY}" ]; then
    echo "  - Dependency: afterok:${EXP3_S1_JOB}:${EXP4_S1_JOB}"
fi
echo ""
echo "Monitor job:"
echo "  squeue -j ${MEGA_JOB}"
echo "  tail -f slurm/logs/mega_job_${MEGA_JOB}.out"
echo "  tail -f slurm/logs/mega_gpu*_${MEGA_JOB}.out  # Individual GPU logs"
echo ""
echo "Results will be saved to:"
echo "  results/exp1_evaluation.json"
echo "  results/exp2_evaluation.json"
echo "========================================================================"

