#!/bin/bash
# Submit categorization and training jobs together with proper dependency
# Usage: ./slurm/submit_categorization_and_training.sh [dataset] [model]

DATASET="${1:-kvasir}"
MODEL="${2:-Qwen/Qwen3-VL-8B-Instruct}"
INPUT_FILE="datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json"

echo "=========================================="
echo "Submitting Categorization + Training Jobs"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo ""

# Step 1: Submit categorization
echo "Step 1: Submitting categorization job..."
CATEG_JOB=$(sbatch --parsable slurm/01_categorize_questions_v2.slurm "$DATASET" "$INPUT_FILE")

if [ $? -ne 0 ] || [ -z "$CATEG_JOB" ]; then
    echo "❌ Failed to submit categorization job"
    exit 1
fi

echo "✅ Categorization job submitted: $CATEG_JOB"
echo ""

# Step 2: Submit training with dependency
echo "Step 2: Submitting training job (depends on $CATEG_JOB)..."
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:"$CATEG_JOB" slurm/03_train_unified.slurm "$DATASET" "$MODEL")

if [ $? -ne 0 ] || [ -z "$TRAIN_JOB" ]; then
    echo "❌ Failed to submit training job"
    exit 1
fi

echo "✅ Training job submitted: $TRAIN_JOB"
echo ""

echo "=========================================="
echo "Jobs Submitted Successfully!"
echo "=========================================="
echo "Categorization Job: $CATEG_JOB"
echo "Training Job: $TRAIN_JOB (waits for $CATEG_JOB)"
echo ""
echo "Monitor jobs:"
echo "  squeue -j $CATEG_JOB,$TRAIN_JOB"
echo ""
echo "View logs:"
echo "  tail -f slurm/logs/categorize_questions_v2_${CATEG_JOB}.out"
echo "  tail -f slurm/logs/train_unified_${TRAIN_JOB}.out"
echo "=========================================="














