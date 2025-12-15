#!/bin/bash
# Script to fill the complete results table
# This runs all evaluations needed to fill the table

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Results directory
RESULTS_DIR="results/evaluation_all_configs"
mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     FILLING COMPLETE RESULTS TABLE                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "This script will submit jobs to evaluate all configurations:"
echo "  1. Zero-shot (no CoT)"
echo "  2. Instruction fine-tuning (no CoT)"
echo "  3. CoT zero-shot"
echo "  4. CoT with instruction fine-tuning"
echo ""
echo "For all model-dataset combinations."
echo ""

# Model-dataset combinations
declare -a COMBINATIONS=(
    "qwen3vl:kvasir"
    "qwen3vl:endovis"
    "medgemma:kvasir"
    "medgemma:endovis"
    "llava_med:kvasir"
    "llava_med:endovis"
)

# Dataset paths
declare -A TEST_DATA=(
    ["kvasir"]="datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json"
    ["endovis"]="datasets/EndoVis2018/raw/metadata/test_vqa_pairs.json"
)

declare -A IMAGE_PATHS=(
    ["kvasir"]="datasets/Kvasir-VQA/raw/images"
    ["endovis"]="datasets/EndoVis2018/raw/images"
)

# Checkpoint paths - UPDATE THESE WITH YOUR ACTUAL CHECKPOINTS
declare -A FINETUNED_CHECKPOINTS=(
    ["qwen3vl_kvasir"]="checkpoints/qwen3vl_kvasir_finetuned"
    ["qwen3vl_endovis"]="checkpoints/qwen3vl_endovis_finetuned"
    ["medgemma_kvasir"]="checkpoints/medgemma_kvasir_finetuned"
    ["medgemma_endovis"]="checkpoints/medgemma_endovis_finetuned"
    ["llava_med_kvasir"]="checkpoints/llava_med_kvasir_finetuned"
    ["llava_med_endovis"]="checkpoints/llava_med_endovis_finetuned"
)

declare -A COT_CHECKPOINTS=(
    ["qwen3vl_kvasir"]="results/multihead_cot/qwen3vl_kvasir_cot_20251208_233609/checkpoint_epoch_3.pt"
    ["qwen3vl_endovis"]="results/multihead_cot/qwen3vl_endovis_cot_*/checkpoint_epoch_*.pt"
    # Add more as they become available
)

echo "📋 Submitting evaluation jobs..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

JOB_IDS=()

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r model_type dataset <<< "$combo"
    
    test_data="${TEST_DATA[$dataset]}"
    image_path="${IMAGE_PATHS[$dataset]}"
    finetuned_ckpt="${FINETUNED_CHECKPOINTS[${model_type}_${dataset}]}"
    cot_ckpt="${COT_CHECKPOINTS[${model_type}_${dataset}]}"
    
    echo "Submitting: $model_type on $dataset"
    echo "  Test data: $test_data"
    echo "  Image path: $image_path"
    echo "  Fine-tuned checkpoint: ${finetuned_ckpt:-None}"
    echo "  CoT checkpoint: ${cot_ckpt:-None}"
    
    JOB_OUTPUT=$(sbatch slurm/10_evaluate_all_configs.slurm \
        "$model_type" \
        "$dataset" \
        "$test_data" \
        "$image_path" \
        "${finetuned_ckpt:-}" \
        "${cot_ckpt:-}" 2>&1)
    
    if [ $? -eq 0 ]; then
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+')
        JOB_IDS+=($JOB_ID)
        echo "  ✅ Job submitted: $JOB_ID"
    else
        echo "  ❌ Failed to submit job"
        echo "  Error: $JOB_OUTPUT"
    fi
    
    echo ""
    sleep 2
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All jobs submitted!"
echo ""
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "📊 Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "📈 After all jobs complete, generate final table:"
echo "  python3 generate_results_table.py \\"
echo "    --results-dir $RESULTS_DIR \\"
echo "    --output results/FINAL_RESULTS_TABLE.md"
echo ""







