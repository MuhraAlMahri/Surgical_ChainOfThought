#!/bin/bash
# Helper script to submit categorization job and optionally training job
# Usage: ./slurm/submit_categorization.sh [dataset] [input_file] [submit_training]

DATASET="${1:-kvasir}"
INPUT_FILE="${2:-datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json}"
SUBMIT_TRAINING="${3:-no}"

echo "Submitting categorization job..."
echo "Dataset: $DATASET"
echo "Input file: $INPUT_FILE"
echo ""

JOB_ID=$(sbatch --parsable slurm/01_categorize_questions_v2.slurm "$DATASET" "$INPUT_FILE")

if [ $? -eq 0 ]; then
    echo "✅ Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    
    # Save job ID to file for later use
    echo "$JOB_ID" > /tmp/categorization_job_id_$$.txt
    echo "Job ID saved to: /tmp/categorization_job_id_$$.txt"
    echo ""
    
    if [ "$SUBMIT_TRAINING" == "yes" ] || [ "$SUBMIT_TRAINING" == "y" ]; then
        echo "Submitting training job with dependency..."
        TRAIN_JOB=$(sbatch --parsable --dependency=afterok:"$JOB_ID" slurm/03_train_unified.slurm "$DATASET")
        if [ $? -eq 0 ]; then
            echo "✅ Training job submitted: $TRAIN_JOB"
        else
            echo "❌ Training job submission failed"
        fi
    else
        echo "To submit training after categorization completes, run:"
        echo "  ./slurm/submit_with_dependency.sh $JOB_ID $DATASET"
        echo ""
        echo "Or manually:"
        echo "  sbatch --dependency=afterok:$JOB_ID slurm/03_train_unified.slurm $DATASET"
    fi
else
    echo "❌ Job submission failed!"
    exit 1
fi

