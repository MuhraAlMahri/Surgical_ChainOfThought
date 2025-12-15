#!/bin/bash
# Helper script to submit training job with dependency on categorization
# Usage: ./slurm/submit_with_dependency.sh CATEG_JOB_ID [dataset] [model]

if [ -z "$1" ]; then
    echo "Usage: $0 CATEG_JOB_ID [dataset] [model]"
    echo ""
    echo "Example:"
    echo "  $0 166091 kvasir Qwen/Qwen3-VL-8B-Instruct"
    exit 1
fi

CATEG_JOB_ID="$1"
DATASET="${2:-kvasir}"
MODEL="${3:-Qwen/Qwen3-VL-8B-Instruct}"

echo "Submitting training job with dependency on categorization job $CATEG_JOB_ID..."
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo ""

# Check if categorization job exists
if ! squeue -j "$CATEG_JOB_ID" &>/dev/null && ! sacct -j "$CATEG_JOB_ID" &>/dev/null; then
    echo "⚠️  WARNING: Job $CATEG_JOB_ID not found in queue or history"
    echo "   Proceeding anyway (job may have completed)"
    echo ""
fi

JOB_ID=$(sbatch --parsable --dependency=afterok:"$CATEG_JOB_ID" slurm/03_train_unified.slurm "$DATASET" "$MODEL")

if [ $? -eq 0 ]; then
    echo "✅ Training job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo "Dependency: afterok:$CATEG_JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f slurm/logs/train_unified_${JOB_ID}.out"
else
    echo "❌ Job submission failed!"
    exit 1
fi














