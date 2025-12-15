#!/bin/bash
# Script to evaluate CoT configurations only (combines with existing baseline numbers)
# Usage: ./scripts/evaluate_cot_all.sh

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

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
    # Try multiple possible EndoVis paths
    ["endovis"]="corrected_1-5_experiments/datasets/endovis2018_vqa/test.jsonl"
)

declare -A IMAGE_PATHS=(
    ["kvasir"]="datasets/Kvasir-VQA/raw/images"
    ["endovis"]="datasets/EndoVis2018/raw/images"
)

# CoT checkpoint paths - UPDATE THESE WITH YOUR ACTUAL CHECKPOINTS
# Note: Only provide paths for checkpoints that exist
# Jobs without checkpoints will only evaluate CoT zero-shot
declare -A COT_CHECKPOINTS=(
    ["qwen3vl_kvasir"]="results/multihead_cot/qwen3vl_kvasir_cot_20251208_233609/checkpoint_epoch_3.pt"
    # ["qwen3vl_endovis"]=""  # Update when checkpoint is available
    # ["medgemma_kvasir"]=""  # Update when checkpoint is available
    # ["medgemma_endovis"]=""  # Update when checkpoint is available
    # ["llava_med_kvasir"]=""  # Update when checkpoint is available
    # ["llava_med_endovis"]=""  # Update when checkpoint is available
)

# Baseline results file (optional - if you have existing baseline numbers)
BASELINE_RESULTS="${1:-results/baseline_results.json}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     EVALUATING CoT CONFIGURATIONS ONLY                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "This will evaluate:"
echo "  1. CoT zero-shot (CoT prompts, no fine-tuning)"
echo "  2. CoT with instruction fine-tuning"
echo ""
echo "Baseline results (no CoT) will be loaded from:"
echo "  ${BASELINE_RESULTS:-(not provided)}"
echo ""

JOB_IDS=()

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r model_type dataset <<< "$combo"
    
    test_data="${TEST_DATA[$dataset]}"
    image_path="${IMAGE_PATHS[$dataset]}"
    cot_ckpt="${COT_CHECKPOINTS[${model_type}_${dataset}]}"
    
    echo "Submitting: $model_type on $dataset"
    echo "  CoT checkpoint: ${cot_ckpt:-None}"
    
    JOB_OUTPUT=$(sbatch slurm/11_evaluate_cot_only.slurm \
        "$model_type" \
        "$dataset" \
        "$test_data" \
        "$image_path" \
        "${cot_ckpt:-}" \
        "${BASELINE_RESULTS:-}" 2>&1)
    
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
echo "✅ All CoT evaluation jobs submitted!"
echo ""
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "📊 Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "📈 After all jobs complete, generate final table:"
echo "  python3 generate_results_table.py \\"
echo "    --results-dir results/cot_evaluation \\"
echo "    --output results/FINAL_RESULTS_TABLE.md"
echo ""

