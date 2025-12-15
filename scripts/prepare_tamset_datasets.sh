#!/bin/bash

#############################################
# TAMSET-Only Dataset Preparation Pipeline
# Creates reordered and non-reordered versions
# for comparative training and evaluation
#############################################

set -e  # Exit on error

echo "================================================"
echo "TAMSET Dataset Preparation Pipeline"
echo "================================================"
echo ""

# Paths
TAMSET_DIR="/l/users/muhra.almahri/Surgical_COT/temset"
SCRIPTS_DIR="/l/users/muhra.almahri/Surgical_COT/scripts"
OUTPUT_DIR="/l/users/muhra.almahri/Surgical_COT/datasets"

# Input file from Qwen3 generation
QWEN3_OUTPUT="${TAMSET_DIR}/tamset_qa_qwen3_full_reordered.json"

# Check if Qwen3 output exists
if [ ! -f "$QWEN3_OUTPUT" ]; then
    echo "❌ ERROR: Qwen3 output file not found: $QWEN3_OUTPUT"
    echo "Please wait for Qwen3 generation to complete."
    exit 1
fi

echo "✅ Found Qwen3 output: $QWEN3_OUTPUT"
echo ""

#############################################
# Step 1: Create Non-Reordered Baseline
#############################################
echo "Step 1: Creating non-reordered baseline dataset..."
python3 ${SCRIPTS_DIR}/create_tamset_baseline.py \
    --input ${QWEN3_OUTPUT} \
    --output ${OUTPUT_DIR}/tamset_non_reordered.json

if [ $? -eq 0 ]; then
    echo "✅ Non-reordered baseline created"
else
    echo "❌ Failed to create baseline"
    exit 1
fi
echo ""

#############################################
# Step 2: Create Train/Val/Test Splits
#############################################
echo "Step 2: Creating train/val/test splits..."
python3 ${SCRIPTS_DIR}/create_tamset_splits.py \
    --reordered ${QWEN3_OUTPUT} \
    --non_reordered ${OUTPUT_DIR}/tamset_non_reordered.json \
    --output_dir ${OUTPUT_DIR}

if [ $? -eq 0 ]; then
    echo "✅ Dataset splits created"
else
    echo "❌ Failed to create splits"
    exit 1
fi
echo ""

#############################################
# Step 3: Generate Dataset Statistics
#############################################
echo "Step 3: Generating dataset statistics..."
python3 ${SCRIPTS_DIR}/analyze_tamset_datasets.py \
    --reordered ${QWEN3_OUTPUT} \
    --non_reordered ${OUTPUT_DIR}/tamset_non_reordered.json \
    --output ${OUTPUT_DIR}/dataset_statistics.json

if [ $? -eq 0 ]; then
    echo "✅ Statistics generated"
else
    echo "❌ Failed to generate statistics"
    exit 1
fi
echo ""

#############################################
# Summary
#############################################
echo "================================================"
echo "✅ TAMSET Dataset Preparation Complete!"
echo "================================================"
echo ""
echo "Output files:"
echo "  - Reordered (original): ${QWEN3_OUTPUT}"
echo "  - Non-reordered baseline: ${OUTPUT_DIR}/tamset_non_reordered.json"
echo ""
echo "Training datasets:"
echo "  - ${OUTPUT_DIR}/tamset_reordered_train.json"
echo "  - ${OUTPUT_DIR}/tamset_reordered_val.json"
echo "  - ${OUTPUT_DIR}/tamset_reordered_test.json"
echo "  - ${OUTPUT_DIR}/tamset_non_reordered_train.json"
echo "  - ${OUTPUT_DIR}/tamset_non_reordered_val.json"
echo "  - ${OUTPUT_DIR}/tamset_non_reordered_test.json"
echo ""
echo "Statistics: ${OUTPUT_DIR}/dataset_statistics.json"
echo ""
echo "Next steps:"
echo "  1. Train on reordered: sbatch training/train_tamset_reordered.slurm"
echo "  2. Train on baseline: sbatch training/train_tamset_baseline.slurm"
echo "  3. Evaluate: python3 evaluation/compare_tamset_models.py"
echo ""

