#!/bin/bash
# Submit mega job for evaluation and retraining

cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/endovis2018_experiments

echo "=========================================="
echo "Submitting Mega Job: Evaluation + Retraining"
echo "=========================================="
echo ""
echo "This job will:"
echo "  GPU 0: Evaluate Exp1 (Random Baseline)"
echo "  GPU 1: Evaluate Exp2 (Qwen Reordered)"
echo "  GPU 2: Evaluate Exp4 Stage2 + Exp5 (Sequential)"
echo "  GPU 3: Retrain Exp4 Stage1 (fixed script)"
echo ""
echo "Note: Exp3 evaluation will run separately after Exp3 training completes"
echo ""

JOB_ID=$(sbatch slurm/mega_job_eval_and_retrain.slurm | grep -oP '\d+')

echo "✅ Job submitted: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f slurm/logs/mega_job_eval_retrain_${JOB_ID}.out"
echo ""
echo "Results will be saved to: results/"
echo "  - exp1_evaluation.json"
echo "  - exp2_evaluation.json"
echo "  - exp4_stage2_evaluation.json"
echo "  - exp5_evaluation.json"
echo "=========================================="
















