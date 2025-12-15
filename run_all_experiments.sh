#!/bin/bash

# Complete pipeline for multi-head temporal CoT experiments

set -e  # Exit on error

# Configuration
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
KVASIR_DATA="${BASE_DIR}/datasets/Kvasir-VQA"
ENDOVIS_DATA="${BASE_DIR}/datasets/EndoVis2018"
OUTPUT_DIR="${BASE_DIR}/results/multihead_cot"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Multi-Head Temporal CoT Experiments"
echo "=========================================="

# Step 1: Question Categorization
echo ""
echo "=== Step 1: Question Categorization ==="
python categorize_questions.py \
    --kvasir_path "${KVASIR_DATA}/raw/metadata/raw_complete_metadata.json" \
    --endovis_path "${ENDOVIS_DATA}/raw/metadata/vqa_pairs.json" \
    --output "${OUTPUT_DIR}/question_categories.json" \
    --model "Qwen/Qwen2.5-7B-Instruct"

if [ $? -ne 0 ]; then
    echo "ERROR: Question categorization failed"
    exit 1
fi

echo "✓ Question categorization complete"

# Step 2: Train Qwen3-VL on Kvasir
echo ""
echo "=== Step 2: Train Qwen3-VL on Kvasir ==="
python train_multihead_cot.py \
    --model_type qwen3vl \
    --dataset kvasir \
    --base_checkpoint "${BASE_DIR}/checkpoints/qwen3vl_kvasir_finetuned" \
    --question_categories "${OUTPUT_DIR}/question_categories.json" \
    --data_path "${KVASIR_DATA}/raw/metadata/raw_complete_metadata.json" \
    --image_base_path "${KVASIR_DATA}/raw/images" \
    --output_dir "${OUTPUT_DIR}/qwen3vl_kvasir_cot" \
    --learning_rate 2e-5 \
    --epochs 3 \
    --batch_size 1

if [ $? -ne 0 ]; then
    echo "ERROR: Qwen3-VL Kvasir training failed"
    exit 1
fi

echo "✓ Qwen3-VL Kvasir training complete"

# Step 3: Evaluate Qwen3-VL on Kvasir
echo ""
echo "=== Step 3: Evaluate Qwen3-VL on Kvasir ==="
python evaluate_multihead.py \
    --checkpoint "${OUTPUT_DIR}/qwen3vl_kvasir_cot/checkpoint_epoch_3.pt" \
    --model-type qwen3vl \
    --test-data "${KVASIR_DATA}/raw/metadata/test_metadata.json" \
    --image-base-path "${KVASIR_DATA}/raw/images" \
    --question-categories "${OUTPUT_DIR}/question_categories.json" \
    --dataset kvasir \
    --output "${OUTPUT_DIR}/evaluation" \
    --baseline-results "${BASE_DIR}/baseline_results.json"

if [ $? -ne 0 ]; then
    echo "WARNING: Qwen3-VL Kvasir evaluation failed (continuing)"
fi

echo "✓ Qwen3-VL Kvasir evaluation complete"

# Step 4: Train Qwen3-VL on EndoVis (with temporal)
echo ""
echo "=== Step 4: Train Qwen3-VL on EndoVis (with temporal) ==="
python train_multihead_cot.py \
    --model_type qwen3vl \
    --dataset endovis \
    --base_checkpoint "${BASE_DIR}/checkpoints/qwen3vl_endovis_finetuned" \
    --question_categories "${OUTPUT_DIR}/question_categories.json" \
    --data_path "${ENDOVIS_DATA}/raw/metadata/train_vqa_pairs.json" \
    --image_base_path "${ENDOVIS_DATA}/raw/images" \
    --output_dir "${OUTPUT_DIR}/qwen3vl_endovis_cot" \
    --learning_rate 2e-5 \
    --epochs 3 \
    --batch_size 1

if [ $? -ne 0 ]; then
    echo "ERROR: Qwen3-VL EndoVis training failed"
    exit 1
fi

echo "✓ Qwen3-VL EndoVis training complete"

# Step 5: Evaluate Qwen3-VL on EndoVis
echo ""
echo "=== Step 5: Evaluate Qwen3-VL on EndoVis ==="
python evaluate_multihead.py \
    --checkpoint "${OUTPUT_DIR}/qwen3vl_endovis_cot/checkpoint_epoch_3.pt" \
    --model-type qwen3vl \
    --test-data "${ENDOVIS_DATA}/raw/metadata/test_vqa_pairs.json" \
    --image-base-path "${ENDOVIS_DATA}/raw/images" \
    --question-categories "${OUTPUT_DIR}/question_categories.json" \
    --dataset endovis \
    --output "${OUTPUT_DIR}/evaluation" \
    --baseline-results "${BASE_DIR}/baseline_results.json"

if [ $? -ne 0 ]; then
    echo "WARNING: Qwen3-VL EndoVis evaluation failed (continuing)"
fi

echo "✓ Qwen3-VL EndoVis evaluation complete"

# Step 6: Train MedGemma on Kvasir
echo ""
echo "=== Step 6: Train MedGemma on Kvasir ==="
python train_multihead_cot.py \
    --model_type medgemma \
    --dataset kvasir \
    --base_checkpoint "${BASE_DIR}/checkpoints/medgemma_kvasir_finetuned" \
    --question_categories "${OUTPUT_DIR}/question_categories.json" \
    --data_path "${KVASIR_DATA}/raw/metadata/raw_complete_metadata.json" \
    --image_base_path "${KVASIR_DATA}/raw/images" \
    --output_dir "${OUTPUT_DIR}/medgemma_kvasir_cot" \
    --learning_rate 3e-5 \
    --epochs 5 \
    --batch_size 2

if [ $? -ne 0 ]; then
    echo "ERROR: MedGemma Kvasir training failed"
    exit 1
fi

echo "✓ MedGemma Kvasir training complete"

# Step 7: Train MedGemma on EndoVis
echo ""
echo "=== Step 7: Train MedGemma on EndoVis ==="
python train_multihead_cot.py \
    --model_type medgemma \
    --dataset endovis \
    --base_checkpoint "${BASE_DIR}/checkpoints/medgemma_endovis_finetuned" \
    --question_categories "${OUTPUT_DIR}/question_categories.json" \
    --data_path "${ENDOVIS_DATA}/raw/metadata/train_vqa_pairs.json" \
    --image_base_path "${ENDOVIS_DATA}/raw/images" \
    --output_dir "${OUTPUT_DIR}/medgemma_endovis_cot" \
    --learning_rate 3e-5 \
    --epochs 5 \
    --batch_size 2

if [ $? -ne 0 ]; then
    echo "ERROR: MedGemma EndoVis training failed"
    exit 1
fi

echo "✓ MedGemma EndoVis training complete"

# Step 8: Train LLaVA-Med on Kvasir
echo ""
echo "=== Step 8: Train LLaVA-Med on Kvasir ==="
python train_multihead_cot.py \
    --model_type llava_med \
    --dataset kvasir \
    --base_checkpoint "${BASE_DIR}/checkpoints/llava_med_kvasir_finetuned" \
    --question_categories "${OUTPUT_DIR}/question_categories.json" \
    --data_path "${KVASIR_DATA}/raw/metadata/raw_complete_metadata.json" \
    --image_base_path "${KVASIR_DATA}/raw/images" \
    --output_dir "${OUTPUT_DIR}/llava_med_kvasir_cot" \
    --learning_rate 2e-5 \
    --epochs 3 \
    --batch_size 1

if [ $? -ne 0 ]; then
    echo "ERROR: LLaVA-Med Kvasir training failed"
    exit 1
fi

echo "✓ LLaVA-Med Kvasir training complete"

# Step 9: Train LLaVA-Med on EndoVis
echo ""
echo "=== Step 9: Train LLaVA-Med on EndoVis ==="
python train_multihead_cot.py \
    --model_type llava_med \
    --dataset endovis \
    --base_checkpoint "${BASE_DIR}/checkpoints/llava_med_endovis_finetuned" \
    --question_categories "${OUTPUT_DIR}/question_categories.json" \
    --data_path "${ENDOVIS_DATA}/raw/metadata/train_vqa_pairs.json" \
    --image_base_path "${ENDOVIS_DATA}/raw/images" \
    --output_dir "${OUTPUT_DIR}/llava_med_endovis_cot" \
    --learning_rate 2e-5 \
    --epochs 3 \
    --batch_size 1

if [ $? -ne 0 ]; then
    echo "ERROR: LLaVA-Med EndoVis training failed"
    exit 1
fi

echo "✓ LLaVA-Med EndoVis training complete"

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""













