#!/bin/bash
# Master script to prepare all datasets for training
# Runs when TAMSET Qwen3 generation completes

echo "=========================================="
echo "DATASET PREPARATION PIPELINE"
echo "=========================================="
echo "Start time: $(date)"
echo ""

cd /l/users/muhra.almahri/Surgical_COT/scripts

# Step 1: Check if TAMSET Qwen3 data is ready
TAMSET_FILE="/l/users/muhra.almahri/Surgical_COT/temset/tamset_qa_qwen3_full_reordered.json"

if [ ! -f "$TAMSET_FILE" ]; then
    echo "⚠️  TAMSET Qwen3 generation not complete yet!"
    echo "   Waiting for: $TAMSET_FILE"
    echo ""
    echo "   This script will:"
    echo "   1. Convert reordered datasets to non-reordered (baseline)"
    echo "   2. Integrate Kvasir-VQA + TAMSET for both versions"
    echo ""
    echo "   You can run steps separately:"
    echo "   - For Kvasir-only: python3 create_non_reordered_datasets.py"
    echo "   - For integration: python3 integrate_datasets.py"
    exit 1
fi

echo "✅ TAMSET Qwen3 data found!"
echo ""

# Step 2: Create non-reordered versions
echo "=========================================="
echo "STEP 1: Creating Non-Reordered Datasets"
echo "=========================================="
python3 create_non_reordered_datasets.py

if [ $? -ne 0 ]; then
    echo "❌ Non-reordered dataset creation failed!"
    exit 1
fi

echo ""
echo "✅ Non-reordered datasets created!"
echo ""

# Step 3: Integrate datasets
echo "=========================================="
echo "STEP 2: Integrating Kvasir-VQA + TAMSET"
echo "=========================================="
python3 integrate_datasets.py

if [ $? -ne 0 ]; then
    echo "❌ Dataset integration failed!"
    exit 1
fi

echo ""
echo "✅ Dataset integration complete!"
echo ""

# Step 4: Summary
echo "=========================================="
echo "🎉 ALL DATASETS PREPARED!"
echo "=========================================="
echo ""
echo "📁 Output location: /l/users/muhra.almahri/Surgical_COT/datasets/"
echo ""
echo "📊 Reordered datasets (3-stage clinical flow):"
echo "   - integrated_train_reordered.json"
echo "   - integrated_val_reordered.json"
echo "   - integrated_test_reordered.json"
echo ""
echo "📊 Non-reordered datasets (baseline):"
echo "   - integrated_train_non_reordered.json"
echo "   - integrated_val_non_reordered.json"
echo "   - integrated_test_non_reordered.json"
echo ""
echo "🚀 READY FOR TRAINING!"
echo ""
echo "Next steps:"
echo "   1. Submit training jobs:"
echo "      cd /l/users/muhra.almahri/Surgical_COT/training"
echo "      sbatch train_reordered.slurm"
echo "      sbatch train_non_reordered.slurm"
echo ""
echo "   2. Monitor training:"
echo "      squeue -u muhra.almahri"
echo "      tail -f training/logs/train_reordered_*.out"
echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="

