#!/bin/bash
#SBATCH --job-name=llava_med_kvasir
#SBATCH --output=slurm/logs/kvasir_llava_med_%j.out
#SBATCH --error=slurm/logs/kvasir_llava_med_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Standalone Job: Zero-Shot + Instruction Fine-tuning for LLaVA-Med v1.5 on Kvasir-VQA
# This script runs the Kvasir-VQA pipeline (zero-shot evaluation then instruction fine-tuning)
# Model: microsoft/llava-med-v1.5-mistral-7b

set -e

echo "=========================================="
echo "LLaVA-Med v1.5: Kvasir-VQA Pipeline"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: 1"
echo "Started: $(date)"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
SCRIPT_DIR="${BASE_DIR}/scripts/evaluation"
QLORA_DIR="${BASE_DIR}/qlora_experiments"
RESULTS_DIR="${QLORA_DIR}/results"
KVASIR_IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"

# Base model
BASE_MODEL="microsoft/llava-med-v1.5-mistral-7b"

# Test data paths
KVASIR_TEST_JSONL="${BASE_DIR}/datasets/kvasir_ULTRA_CONDENSED/test_CATEGORY_BASED.jsonl"
KVASIR_TEST_JSON="/tmp/zeroshot_kvasir_llava_${SLURM_JOB_ID}.json"

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${QLORA_DIR}/slurm/logs"
mkdir -p "${QLORA_DIR}/models/llava_med_kvasir_instruction"

# Load modules
module purge
module load nvidia/cuda/12.0 2>/dev/null || true

# Activate environment
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1

# Use shared cache (model files already downloaded there)
export SHARED_CACHE="/l/users/muhra.almahri/.cache/hf_shared"
export HF_HOME=$SHARED_CACHE
export TRANSFORMERS_CACHE=$SHARED_CACHE/transformers
export HF_DATASETS_CACHE=$SHARED_CACHE/datasets
export HF_HUB_CACHE=$SHARED_CACHE
export TORCH_HOME=$SHARED_CACHE/torch
export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}
mkdir -p ${TRITON_CACHE_DIR}

# Disable DeepSpeed CUDA checks
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_USE_DEEPSPEED=false
export DEEPSPEED_DISABLED=true

# Hugging Face authentication
export HF_TOKEN="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"

# ============================================================================
# Step 1: Zero-Shot Evaluation
# ============================================================================
echo ""
echo "=========================================="
echo "Step 1: Zero-Shot Evaluation"
echo "=========================================="
echo "Started: $(date)"

# Convert JSONL to JSON
python3 << PYTHON_EOF
import json
import sys
import os

jsonl_path = "${KVASIR_TEST_JSONL}"
json_path = "${KVASIR_TEST_JSON}"

try:
    with open(jsonl_path, 'r') as f_in:
        data = [json.loads(line) for line in f_in]
    
    with open(json_path, 'w') as f_out:
        json.dump(data, f_out, indent=2)
    
    print(f"✓ Converted {len(data)} samples from JSONL to JSON")
except Exception as e:
    print(f"Error reading test data: {e}")
    sys.exit(1)
PYTHON_EOF

# Run zero-shot evaluation
python3 "${SCRIPT_DIR}/evaluate_zeroshot.py" \
    --base_model "${BASE_MODEL}" \
    --test_data "${KVASIR_TEST_JSON}" \
    --image_dir "${KVASIR_IMAGE_DIR}" \
    --output "${RESULTS_DIR}/kvasir_zeroshot_llava_med_v15.json" \
    --use_instruction

KVASIR_ZEROSHOT_STATUS=$?
if [ $KVASIR_ZEROSHOT_STATUS -eq 0 ]; then
    echo "✓ Kvasir zero-shot evaluation completed: $(date)"
    echo "  Results: ${RESULTS_DIR}/kvasir_zeroshot_llava_med_v15.json"
else
    echo "✗ Kvasir zero-shot evaluation failed: $(date) (exit code: $KVASIR_ZEROSHOT_STATUS)"
    exit $KVASIR_ZEROSHOT_STATUS
fi

# Clean up temporary file
rm -f "${KVASIR_TEST_JSON}"

# ============================================================================
# Step 2: Instruction Fine-tuning
# ============================================================================
echo ""
echo "=========================================="
echo "Step 2: Instruction Fine-tuning"
echo "=========================================="
echo "Started: $(date)"

cd "${QLORA_DIR}"

# Set dataset environment variables
export DATASET_NAME=kvasir
export DATA_ROOT="${BASE_DIR}/datasets/kvasir_ULTRA_CONDENSED"
export OUTPUT_DIR="${QLORA_DIR}/models/llava_med_kvasir_instruction"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run instruction fine-tuning
python3 train_llava_NATIVE_PROCESSOR.py

KVASIR_TRAIN_STATUS=$?
if [ $KVASIR_TRAIN_STATUS -eq 0 ]; then
    echo "✓ Kvasir instruction fine-tuning completed: $(date)"
    echo "  Model: ${OUTPUT_DIR}"
else
    echo "✗ Kvasir instruction fine-tuning failed: $(date) (exit code: $KVASIR_TRAIN_STATUS)"
    exit $KVASIR_TRAIN_STATUS
fi

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "Kvasir-VQA Pipeline Complete"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Results:"
echo "  Zero-shot: ${RESULTS_DIR}/kvasir_zeroshot_llava_med_v15.json"
echo "  Model: ${OUTPUT_DIR}"
echo "=========================================="

exit 0






