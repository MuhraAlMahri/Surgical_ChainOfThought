#!/bin/bash

echo "=========================================="
echo "🚀 SUBMITTING REMAINING JOBS"
echo "=========================================="
echo ""

# Check how many jobs are currently submitted
SUBMITTED=$(squeue -u muhra.almahri -h | wc -l)
echo "Current jobs in queue: $SUBMITTED"
echo "Max allowed: 4"
echo ""

if [ $SUBMITTED -ge 4 ]; then
    echo "⚠️  Already at submission limit (4 jobs)"
    echo "Wait for some jobs to complete, then run this script again."
    echo ""
    squeue -u muhra.almahri
    exit 1
fi

SLOTS=$((4 - SUBMITTED))
echo "Available submission slots: $SLOTS"
echo ""

# Submit in order of priority
echo "Submitting in order:"

# Exp 3 Stage 3 (if not already submitted)
if ! squeue -u muhra.almahri -n exp3_kvasir_s3 -h | grep -q .; then
    echo "  → Exp 3 Stage 3..."
    sbatch experiments/exp3_cxrtrek_sequential/train_stage3.slurm
    SUBMITTED=$((SUBMITTED + 1))
    
    if [ $SUBMITTED -ge 4 ]; then
        echo ""
        echo "Reached submission limit. Run script again after more jobs complete."
        exit 0
    fi
fi

# Exp 4 Stage 1 (if not already submitted)
if ! squeue -u muhra.almahri -n exp4_kvasir_s1 -h | grep -q .; then
    echo "  → Exp 4 Stage 1..."
    JOB_E4S1=$(sbatch experiments/exp4_curriculum_learning/train_stage1.slurm 2>&1 | grep -o '[0-9]*' | head -1)
    
    if [ -n "$JOB_E4S1" ]; then
        echo "    Job $JOB_E4S1 submitted"
        SUBMITTED=$((SUBMITTED + 1))
        
        # Try to submit Stage 2 with dependency
        if [ $SUBMITTED -lt 4 ]; then
            echo "  → Exp 4 Stage 2 (depends on $JOB_E4S1)..."
            JOB_E4S2=$(sbatch --dependency=afterok:$JOB_E4S1 experiments/exp4_curriculum_learning/train_stage2.slurm 2>&1 | grep -o '[0-9]*' | head -1)
            
            if [ -n "$JOB_E4S2" ]; then
                echo "    Job $JOB_E4S2 submitted"
                SUBMITTED=$((SUBMITTED + 1))
                
                # Try to submit Stage 3 with dependency
                if [ $SUBMITTED -lt 4 ]; then
                    echo "  → Exp 4 Stage 3 (depends on $JOB_E4S2)..."
                    sbatch --dependency=afterok:$JOB_E4S2 experiments/exp4_curriculum_learning/train_stage3.slurm
                    SUBMITTED=$((SUBMITTED + 1))
                fi
            fi
        fi
    fi
fi

echo ""
echo "=========================================="
echo "✅ Submission Complete"
echo "=========================================="
echo ""
squeue -u muhra.almahri
