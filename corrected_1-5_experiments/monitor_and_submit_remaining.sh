#!/bin/bash

echo ""
echo "=========================================="
echo "📊 TRAINING STATUS CHECK"
echo "=========================================="
echo ""
date
echo ""

# Check queue
echo "--- QUEUE STATUS ---"
squeue -u muhra.almahri -o "%.10i %.30j %.8T %.10M %.6D %R"
echo ""

# Count running and pending jobs
RUNNING=$(squeue -u muhra.almahri -t RUNNING -h | wc -l)
PENDING=$(squeue -u muhra.almahri -t PENDING -h | wc -l)

echo "Running: $RUNNING, Pending: $PENDING"
echo ""

# If we have less than 2 jobs in queue, submit remaining ones
TOTAL_IN_QUEUE=$((RUNNING + PENDING))

if [ $TOTAL_IN_QUEUE -lt 2 ]; then
    echo "🚀 Space available! Submitting remaining jobs..."
    echo ""
    
    # Try to submit Exp 3 Stage 3
    if [ ! -f "logs/train_exp3_stage3_*.out" ]; then
        echo "Submitting Exp 3 Stage 3..."
        sbatch experiments/exp3_cxrtrek_sequential/train_stage3.slurm
    fi
    
    # Try to submit Exp 4 Stage 1 (if not already submitted)
    if [ ! -f "logs/train_exp4_stage1_*.out" ]; then
        echo "Submitting Exp 4 Stage 1..."
        JOB_E4S1=$(sbatch experiments/exp4_curriculum_learning/train_stage1.slurm 2>&1 | grep -o '[0-9]*' | head -1)
        
        if [ -n "$JOB_E4S1" ]; then
            echo "   → Job $JOB_E4S1 submitted"
            
            # Submit Stage 2 with dependency
            echo "Submitting Exp 4 Stage 2 (depends on $JOB_E4S1)..."
            JOB_E4S2=$(sbatch --dependency=afterok:$JOB_E4S1 experiments/exp4_curriculum_learning/train_stage2.slurm 2>&1 | grep -o '[0-9]*' | head -1)
            
            if [ -n "$JOB_E4S2" ]; then
                echo "   → Job $JOB_E4S2 submitted"
                
                # Submit Stage 3 with dependency
                echo "Submitting Exp 4 Stage 3 (depends on $JOB_E4S2)..."
                sbatch --dependency=afterok:$JOB_E4S2 experiments/exp4_curriculum_learning/train_stage3.slurm
            fi
        fi
    fi
fi

# Check each experiment's latest loss
echo ""
echo "--- TRAINING PROGRESS ---"
for exp in exp1_random exp2_qwen_reordered exp3_stage1 exp3_stage2 exp3_stage3 exp4_stage1 exp4_stage2 exp4_stage3; do
    LOG=$(ls -t logs/train_${exp}*.out 2>/dev/null | head -1)
    if [ -f "$LOG" ]; then
        echo "--- $exp ---"
        
        # Get first and last loss values
        FIRST=$(grep "{'loss':" "$LOG" | head -1)
        LAST=$(grep "{'loss':" "$LOG" | tail -1)
        
        if [ -n "$FIRST" ]; then
            echo "  First: $FIRST"
        fi
        if [ -n "$LAST" ]; then
            echo "  Latest: $LAST"
        fi
        
        # Check if loss is decreasing
        FIRST_LOSS=$(echo "$FIRST" | grep -oP "'loss':\s*\K[0-9.]+")
        LAST_LOSS=$(echo "$LAST" | grep -oP "'loss':\s*\K[0-9.]+")
        
        if [ -n "$FIRST_LOSS" ] && [ -n "$LAST_LOSS" ]; then
            if (( $(echo "$LAST_LOSS < $FIRST_LOSS" | bc -l) )); then
                echo "  ✅ Loss decreasing: $FIRST_LOSS → $LAST_LOSS (training working!)"
            elif (( $(echo "$LAST_LOSS == 0" | bc -l) )); then
                echo "  ❌ Loss is 0.0 - TRAINING BUG!"
            else
                echo "  ⚠️ Loss: $LAST_LOSS (monitoring...)"
            fi
        fi
        echo ""
    fi
done

echo "=========================================="
