#!/bin/bash
# ============================================================================
# Submit Exp4 Curriculum Learning Retraining Jobs
# ============================================================================
# This script submits Exp4 training jobs with proper dependencies
# ============================================================================

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"
SLURM_DIR="${BASE_DIR}/slurm"

echo "=========================================="
echo "SUBMITTING EXP4 CURRICULUM LEARNING JOBS"
echo "=========================================="
echo ""

# Option 1: Submit all stages in one job (sequential, recommended)
echo "Option 1: Submit all stages in one sequential job"
echo "This runs Stage 1 → Stage 2 → Stage 3 in sequence"
echo ""
read -p "Submit all-in-one job? (y/n): " submit_all

if [ "$submit_all" = "y" ] || [ "$submit_all" = "Y" ]; then
    echo ""
    echo "Submitting all-in-one job..."
    JOB_ID=$(sbatch ${SLURM_DIR}/train_exp4_all_stages.slurm | grep -o '[0-9]*')
    echo "✓ Submitted job ID: ${JOB_ID}"
    echo ""
    echo "This job will run all 3 stages sequentially:"
    echo "  - Stage 1: ~7 hours"
    echo "  - Stage 2: ~12 hours"
    echo "  - Stage 3: ~30 minutes"
    echo "  - Total: ~20 hours"
    echo ""
    echo "Monitor with: squeue -j ${JOB_ID}"
    echo "View logs: tail -f ${SLURM_DIR}/logs/exp4_all_${JOB_ID}.out"
    exit 0
fi

# Option 2: Submit with dependencies (separate jobs)
echo ""
echo "Option 2: Submit with dependencies (separate jobs)"
echo "This submits 3 separate jobs with dependencies"
echo ""
read -p "Submit with dependencies? (y/n): " submit_deps

if [ "$submit_deps" = "y" ] || [ "$submit_deps" = "Y" ]; then
    echo ""
    echo "Submitting Stage 1..."
    JOB1=$(sbatch ${SLURM_DIR}/train_exp4_stage1.slurm | grep -o '[0-9]*')
    echo "✓ Stage 1 job ID: ${JOB1}"
    
    echo ""
    echo "Submitting Stage 2 (depends on Stage 1)..."
    JOB2=$(sbatch --dependency=afterok:${JOB1} ${SLURM_DIR}/train_exp4_stage2.slurm | grep -o '[0-9]*')
    echo "✓ Stage 2 job ID: ${JOB2} (depends on ${JOB1})"
    
    echo ""
    echo "Submitting Stage 3 (depends on Stage 2)..."
    JOB3=$(sbatch --dependency=afterok:${JOB2} ${SLURM_DIR}/train_exp4_stage3.slurm | grep -o '[0-9]*')
    echo "✓ Stage 3 job ID: ${JOB3} (depends on ${JOB2})"
    
    echo ""
    echo "=========================================="
    echo "ALL JOBS SUBMITTED"
    echo "=========================================="
    echo "Stage 1: Job ${JOB1}"
    echo "Stage 2: Job ${JOB2} (waits for ${JOB1})"
    echo "Stage 3: Job ${JOB3} (waits for ${JOB2})"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "View logs:"
    echo "  Stage 1: tail -f ${SLURM_DIR}/logs/exp4_s1_${JOB1}.out"
    echo "  Stage 2: tail -f ${SLURM_DIR}/logs/exp4_s2_${JOB2}.out"
    echo "  Stage 3: tail -f ${SLURM_DIR}/logs/exp4_s3_${JOB3}.out"
    exit 0
fi

echo ""
echo "No jobs submitted. Exiting."
exit 0

