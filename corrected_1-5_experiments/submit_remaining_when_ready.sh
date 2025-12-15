#!/bin/bash
# Auto-submission script for remaining jobs
# Run this after QOS is fixed

cd "/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║          🚀 SUBMITTING REMAINING JOBS (POST-QOS FIX) 🚀           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Exp3 Stage 2 is done
EXP3_S2_STATUS=$(squeue -u muhra.almahri -j 152678 -h -o "%T" 2>/dev/null)

if [ -z "$EXP3_S2_STATUS" ]; then
    echo "✅ Exp3 Stage 2 completed - submitting Stage 3..."
    JOB_EXP3_S3=$(sbatch experiments/exp3_cxrtrek_sequential/train_stage3.slurm | grep -o '[0-9]*')
    echo "   → Job $JOB_EXP3_S3 submitted (Exp3 Stage 3)"
else
    echo "⏳ Exp3 Stage 2 still running - Stage 3 will be submitted later"
    JOB_EXP3_S3="PENDING"
fi

echo ""

# Submit Exp4 stages (sequential with dependencies)
echo "📊 Submitting Exp4 (Curriculum Learning) - Sequential Stages"
JOB_EXP4_S1=$(sbatch experiments/exp4_curriculum_learning/train_stage1.slurm | grep -o '[0-9]*')
echo "   → Job $JOB_EXP4_S1 (Exp4 Stage 1)"

JOB_EXP4_S2=$(sbatch --dependency=afterok:$JOB_EXP4_S1 experiments/exp4_curriculum_learning/train_stage2.slurm | grep -o '[0-9]*')
echo "   → Job $JOB_EXP4_S2 (Exp4 Stage 2, waits for $JOB_EXP4_S1)"

JOB_EXP4_S3=$(sbatch --dependency=afterok:$JOB_EXP4_S2 experiments/exp4_curriculum_learning/train_stage3.slurm | grep -o '[0-9]*')
echo "   → Job $JOB_EXP4_S3 (Exp4 Stage 3, waits for $JOB_EXP4_S2)"

echo ""
echo "=" * 72
echo ""

echo "✅ SUBMITTED JOBS:"
echo "  Training:"
echo "    • Exp3 Stage 3: ${JOB_EXP3_S3}"
echo "    • Exp4 Stage 1: ${JOB_EXP4_S1}"
echo "    • Exp4 Stage 2: ${JOB_EXP4_S2} (depends on ${JOB_EXP4_S1})"
echo "    • Exp4 Stage 3: ${JOB_EXP4_S3} (depends on ${JOB_EXP4_S2})"
echo ""

echo "📋 REMAINING TO SUBMIT MANUALLY (after training completes):"
echo "  • Exp3 Evaluation (after all Exp3 stages done)"
echo "  • Exp4 Evaluation (after all Exp4 stages done)"
echo "  • Exp5 Evaluation (after Exp5 training done - Job 152813)"
echo ""

echo "Monitor with: squeue -u muhra.almahri"
echo ""

squeue -u muhra.almahri





