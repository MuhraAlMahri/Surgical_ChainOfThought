#!/bin/bash
# Cleanup script to remove old checkpoints trained on incorrect data

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
MODELS_DIR="${EXPERIMENTS_DIR}/models"

echo "=========================================="
echo "Cleaning Up Old Checkpoints"
echo "=========================================="
echo "Removing checkpoints trained on INCORRECT data"
echo "These will be retrained with the corrected dataset"
echo "=========================================="
echo ""

# List of experiments to clean
experiments=(
    "exp1_random"
    "exp2_qwen_reordered"
    "exp3_sequential"
    "exp4_curriculum"
    "exp5_sequential_cot"
)

total_deleted=0

for exp in "${experiments[@]}"; do
    exp_path="${MODELS_DIR}/${exp}"
    
    if [ -d "${exp_path}" ]; then
        # Find all checkpoints
        checkpoints=$(find "${exp_path}" -type d -name "checkpoint-*" 2>/dev/null)
        
        if [ -n "${checkpoints}" ]; then
            count=$(echo "${checkpoints}" | wc -l)
            echo "⚠️  ${exp}: Found ${count} checkpoint(s)"
            
            # Delete checkpoints
            find "${exp_path}" -type d -name "checkpoint-*" -exec rm -rf {} + 2>/dev/null
            total_deleted=$((total_deleted + count))
            echo "   ✓ Deleted ${count} checkpoint(s)"
        else
            echo "✓ ${exp}: No checkpoints found"
        fi
    else
        echo "✓ ${exp}: Directory doesn't exist"
    fi
done

echo ""
echo "=========================================="
echo "Cleanup Complete"
echo "=========================================="
echo "Total checkpoints deleted: ${total_deleted}"
echo ""
echo "✅ All experiments are now ready for retraining"
echo "   with the corrected dataset!"
echo "=========================================="
















