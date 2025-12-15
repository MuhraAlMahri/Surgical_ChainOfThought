#!/bin/bash
# Script to run all evaluation configurations
# Usage: ./scripts/run_all_evaluations.sh

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
    ["endovis"]="datasets/EndoVis2018/raw/metadata/test_vqa_pairs.json"
)

declare -A IMAGE_PATHS=(
    ["kvasir"]="datasets/Kvasir-VQA/raw/images"
    ["endovis"]="datasets/EndoVis2018/raw/images"
)

# Checkpoint paths (update these based on your actual checkpoints)
declare -A FINETUNED_CHECKPOINTS=(
    ["qwen3vl_kvasir"]="checkpoints/qwen3vl_kvasir_finetuned"
    ["qwen3vl_endovis"]="checkpoints/qwen3vl_endovis_finetuned"
    # Add more as available
)

declare -A COT_CHECKPOINTS=(
    ["qwen3vl_kvasir"]="results/multihead_cot/qwen3vl_kvasir_cot_20251208_233609/checkpoint_epoch_3.pt"
    # Add more as available
)

echo "Submitting evaluation jobs for all configurations..."
echo ""

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r model_type dataset <<< "$combo"
    
    test_data="${TEST_DATA[$dataset]}"
    image_path="${IMAGE_PATHS[$dataset]}"
    finetuned_ckpt="${FINETUNED_CHECKPOINTS[${model_type}_${dataset}]}"
    cot_ckpt="${COT_CHECKPOINTS[${model_type}_${dataset}]}"
    
    echo "Submitting: $model_type on $dataset"
    
    sbatch slurm/10_evaluate_all_configs.slurm \
        "$model_type" \
        "$dataset" \
        "$test_data" \
        "$image_path" \
        "${finetuned_ckpt:-}" \
        "${cot_ckpt:-}"
    
    sleep 2
done

echo ""
echo "All evaluation jobs submitted!"








