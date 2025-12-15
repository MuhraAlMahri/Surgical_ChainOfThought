#!/bin/bash
#SBATCH --job-name=multihead_cot_pipeline
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/pipeline_%j.out
#SBATCH --error=slurm/logs/pipeline_%j.err

# Pipeline script to submit all jobs in sequence
# This script submits jobs with dependencies so they run in order

echo "=========================================="
echo "Multi-Head Temporal CoT Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="
echo ""

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create logs directory
mkdir -p slurm/logs

# Configuration
DATASET="${1:-kvasir}"  # kvasir or endovis
BASE_MODEL="${2:-Qwen/Qwen2-VL-2B-Instruct}"

echo "Pipeline configuration:"
echo "  Dataset: $DATASET"
echo "  Base model: $BASE_MODEL"
echo ""

# Step 1: Categorize questions
echo "Step 1: Submitting question categorization job..."
JOB1=$(sbatch --parsable slurm/01_categorize_questions.slurm "$DATASET")
echo "  Job ID: $JOB1"
echo ""

# Step 2: Create temporal structure (only for EndoVis)
if [ "$DATASET" == "endovis" ]; then
    echo "Step 2: Submitting temporal structure creation job..."
    JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/02_create_temporal_structure.slurm)
    echo "  Job ID: $JOB2"
    TEMP_DEPENDENCY="--dependency=afterok:$JOB2"
else
    TEMP_DEPENDENCY="--dependency=afterok:$JOB1"
fi
echo ""

# Step 3: Train unified model
echo "Step 3: Submitting unified training job..."
JOB3=$(sbatch --parsable $TEMP_DEPENDENCY slurm/03_train_unified.slurm "$DATASET" "$BASE_MODEL")
echo "  Job ID: $JOB3"
echo ""

# Step 4: Train sequential model
echo "Step 4: Submitting sequential training job..."
JOB4=$(sbatch --parsable $TEMP_DEPENDENCY slurm/04_train_sequential.slurm "$DATASET" "$BASE_MODEL")
echo "  Job ID: $JOB4"
echo ""

# Step 5: Evaluate unified model
echo "Step 5: Submitting evaluation job for unified model..."
JOB5=$(sbatch --parsable --dependency=afterok:$JOB3 slurm/05_evaluate.slurm)
echo "  Job ID: $JOB5"
echo ""

# Step 6: Evaluate sequential model
echo "Step 6: Submitting evaluation job for sequential model..."
JOB6=$(sbatch --parsable --dependency=afterok:$JOB4 slurm/05_evaluate.slurm)
echo "  Job ID: $JOB6"
echo ""

echo "=========================================="
echo "Pipeline submitted!"
echo "=========================================="
echo "Job dependencies:"
echo "  1. Categorization: $JOB1"
if [ "$DATASET" == "endovis" ]; then
    echo "  2. Temporal structure: $JOB2"
fi
echo "  3. Unified training: $JOB3"
echo "  4. Sequential training: $JOB4"
echo "  5. Unified evaluation: $JOB5"
echo "  6. Sequential evaluation: $JOB6"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo "  tail -f slurm/logs/*.out"
echo "=========================================="














