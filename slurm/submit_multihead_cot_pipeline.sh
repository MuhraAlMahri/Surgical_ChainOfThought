#!/bin/bash
# Submit all multi-head CoT experiments through SLURM
# This script submits jobs with proper dependencies

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

echo "=========================================="
echo "Submitting Multi-Head CoT Pipeline Jobs"
echo "=========================================="
echo ""

# Step 1: Question Categorization
echo "Step 1: Submitting question categorization..."
CATEG_JOB=$(sbatch --parsable slurm/06_categorize_questions_new.slurm \
    "datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json" \
    "datasets/EndoVis2018/raw/metadata/vqa_pairs.json" \
    "results/multihead_cot/question_categories.json")

echo "  ✅ Categorization job submitted: $CATEG_JOB"
echo ""

# Configuration for checkpoints (adjust these paths to your actual checkpoint locations)
declare -A CHECKPOINTS=(
    ["qwen3vl_kvasir"]="checkpoints/qwen3vl_kvasir_finetuned"
    ["qwen3vl_endovis"]="checkpoints/qwen3vl_endovis_finetuned"
    ["medgemma_kvasir"]="checkpoints/medgemma_kvasir_finetuned"
    ["medgemma_endovis"]="checkpoints/medgemma_endovis_finetuned"
    ["llava_med_kvasir"]="checkpoints/llava_med_kvasir_finetuned"
    ["llava_med_endovis"]="checkpoints/llava_med_endovis_finetuned"
)

# Step 2: Training jobs (depend on categorization)
echo "Step 2: Submitting training jobs..."
TRAIN_JOBS=()

for MODEL_TYPE in qwen3vl medgemma llava_med; do
    for DATASET in kvasir endovis; do
        KEY="${MODEL_TYPE}_${DATASET}"
        CHECKPOINT="${CHECKPOINTS[$KEY]}"
        
        if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ] && [ ! -d "$CHECKPOINT" ]; then
            echo "  ⚠️  Skipping ${KEY}: checkpoint not found at ${CHECKPOINT}"
            continue
        fi
        
        echo "  Submitting training: ${MODEL_TYPE} on ${DATASET}..."
        TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$CATEG_JOB \
            slurm/07_train_multihead_cot.slurm \
            "$MODEL_TYPE" \
            "$DATASET" \
            "$CHECKPOINT")
        
        TRAIN_JOBS+=($TRAIN_JOB)
        echo "    ✅ Training job submitted: $TRAIN_JOB"
    done
done

echo ""
echo "Step 3: Submitting evaluation jobs (depend on training)..."
EVAL_JOBS=()

# Wait for all training jobs to complete
if [ ${#TRAIN_JOBS[@]} -gt 0 ]; then
    TRAIN_DEPENDENCY="afterok:$(IFS=:; echo "${TRAIN_JOBS[*]}")"
    
    for MODEL_TYPE in qwen3vl medgemma llava_med; do
        for DATASET in kvasir endovis; do
            KEY="${MODEL_TYPE}_${DATASET}"
            CHECKPOINT="${CHECKPOINTS[$KEY]}"
            
            if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ] && [ ! -d "$CHECKPOINT" ]; then
                continue
            fi
            
            # Find the corresponding training job output directory
            # This is a simplified version - in practice, you'd track the output dirs
            OUTPUT_DIR="results/multihead_cot/${MODEL_TYPE}_${DATASET}_cot_*"
            CHECKPOINT_FILE=$(find results/multihead_cot -name "checkpoint_epoch_*.pt" -path "*/${MODEL_TYPE}_${DATASET}_cot_*" | sort -V | tail -1)
            
            if [ -z "$CHECKPOINT_FILE" ]; then
                echo "  ⚠️  Skipping evaluation for ${KEY}: checkpoint not found"
                continue
            fi
            
            echo "  Submitting evaluation: ${MODEL_TYPE} on ${DATASET}..."
            EVAL_JOB=$(sbatch --parsable --dependency=$TRAIN_DEPENDENCY \
                slurm/08_evaluate_multihead_cot.slurm \
                "$CHECKPOINT_FILE" \
                "$MODEL_TYPE" \
                "$DATASET")
            
            EVAL_JOBS+=($EVAL_JOB)
            echo "    ✅ Evaluation job submitted: $EVAL_JOB"
        done
    done
fi

echo ""
echo "=========================================="
echo "Job Submission Summary"
echo "=========================================="
echo "Categorization job: $CATEG_JOB"
echo "Training jobs: ${#TRAIN_JOBS[@]} jobs"
echo "Evaluation jobs: ${#EVAL_JOBS[@]} jobs"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  scontrol show job <job_id>"
echo ""
echo "Check logs in: slurm/logs/"
echo "=========================================="













