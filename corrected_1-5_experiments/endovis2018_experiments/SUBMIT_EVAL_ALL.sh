#!/bin/bash
# Submit script for evaluating all EndoVis2018 experiments

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
SLURM_SCRIPT="${EXPERIMENTS_DIR}/slurm/mega_job_evaluate_all.slurm"

echo "=========================================="
echo "Submitting Mega Job: Evaluate All Experiments"
echo "=========================================="
echo ""
echo "This will evaluate ALL 5 experiments:"
echo "  - Exp1: Random Baseline"
echo "  - Exp2: Qwen Reordered"
echo "  - Exp3: Sequential (Stage 1 → Stage 2)"
echo "  - Exp4: Curriculum (Stage 2 model)"
echo "  - Exp5: Sequential CoT"
echo ""
echo "⚠️  All evaluations use the CORRECTED test dataset"
echo ""
echo "GPU Distribution:"
echo "  GPU 0: Exp1"
echo "  GPU 1: Exp2"
echo "  GPU 2: Exp3 (Stage 1 & Stage 2 models)"
echo "  GPU 3: Exp4 → Exp5"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Submitting job..."
JOB_ID=$(sbatch "${SLURM_SCRIPT}" | grep -oP '\d+')

if [ -n "${JOB_ID}" ]; then
    echo ""
    echo "=========================================="
    echo "✅ Job Submitted Successfully!"
    echo "=========================================="
    echo "Job ID: ${JOB_ID}"
    echo "Job Name: endovis_eval_all"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j ${JOB_ID}"
    echo "  tail -f ${EXPERIMENTS_DIR}/slurm/logs/mega_job_eval_all_${JOB_ID}.out"
    echo ""
    echo "Expected runtime: ~4-8 hours (depending on test set size)"
    echo "=========================================="
else
    echo ""
    echo "❌ Failed to submit job"
    exit 1
fi
















