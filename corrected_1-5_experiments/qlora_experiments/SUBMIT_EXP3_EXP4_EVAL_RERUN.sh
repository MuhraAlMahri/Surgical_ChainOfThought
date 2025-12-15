#!/bin/bash
# ============================================================================
# Re-run Exp3 and Exp4 Evaluations with Updated Scripts
# ============================================================================
# This script re-runs the evaluations to include instruction and question_type
# fields in the results, matching Exp1 and Exp2 evaluation format.
# ============================================================================

echo "=========================================="
echo "Re-running Exp3 and Exp4 Evaluations"
echo "=========================================="
echo ""

cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments

# Submit the 2-GPU evaluation job
echo "Submitting Exp3 and Exp4 evaluation job..."
JOB_ID=$(sbatch --parsable slurm/exp3_exp4_eval_2gpu.slurm)

echo ""
echo "=========================================="
echo "Job Submitted Successfully!"
echo "=========================================="
echo "Job ID: ${JOB_ID}"
echo "Job Name: exp3_exp4_eval_2gpu"
echo ""
echo "This job will:"
echo "  - GPU 0: Re-evaluate Exp3 (CXRTrek Sequential)"
echo "  - GPU 1: Re-evaluate Exp4 (Curriculum Learning)"
echo ""
echo "Updated results will be saved to:"
echo "  - results/exp3_evaluation.json"
echo "  - results/exp4_evaluation.json"
echo ""
echo "Check status with:"
echo "  squeue -u muhra.almahri -j ${JOB_ID}"
echo ""
echo "View logs with:"
echo "  tail -f slurm/logs/exp3_exp4_eval_2gpu_${JOB_ID}.out"
echo "=========================================="




