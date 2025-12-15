#!/bin/bash
# Submit both zero-shot evaluation jobs

echo "=========================================="
echo "Submitting Zero-Shot Evaluation Jobs"
echo "=========================================="
echo ""

cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments

# Submit Batch 1 (Exp1-4)
echo "Submitting Batch 1: Exp1, Exp2, Exp3, Exp4 (4 GPUs)..."
JOB1=$(sbatch slurm/zeroshot_batch1.slurm | awk '{print $4}')
echo "  Batch 1 Job ID: ${JOB1}"
echo ""

# Submit Batch 2 (Exp5)
echo "Submitting Batch 2: Exp5 (4 GPUs)..."
JOB2=$(sbatch slurm/zeroshot_batch2.slurm | awk '{print $4}')
echo "  Batch 2 Job ID: ${JOB2}"
echo ""

echo "=========================================="
echo "Both jobs submitted successfully!"
echo "=========================================="
echo ""
echo "Job IDs:"
echo "  Batch 1 (Exp1-4): ${JOB1}"
echo "  Batch 2 (Exp5):   ${JOB2}"
echo ""
echo "Monitor jobs:"
echo "  squeue -j ${JOB1},${JOB2}"
echo ""
echo "View logs:"
echo "  tail -f slurm/logs/zeroshot_batch1_${JOB1}.out"
echo "  tail -f slurm/logs/zeroshot_batch2_${JOB2}.out"
echo "=========================================="

