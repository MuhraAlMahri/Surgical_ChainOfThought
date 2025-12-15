#!/bin/bash
#SBATCH --job-name=llava_med_mega
#SBATCH --output=slurm/logs/mega_job_llava_med_%j.out
#SBATCH --error=slurm/logs/mega_job_llava_med_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Mega Job: Zero-Shot + Instruction Fine-tuning for LLaVA-Med v1.5
# Runs 2 jobs in parallel, each using 1 GPU:
#   GPU 0: Zero-shot evaluation on Kvasir-VQA, then Instruction fine-tuning on Kvasir-VQA
#   GPU 1: Zero-shot evaluation on EndoVis2018, then Instruction fine-tuning on EndoVis2018
#
# Model: microsoft/llava-med-v1.5-mistral-7b

set -e

echo "=========================================="
echo "MEGA JOB: LLaVA-Med v1.5 Experiments"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 2 (parallel execution - each GPU handles one dataset)"
echo "Started: $(date)"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
SCRIPT_DIR="${BASE_DIR}/scripts/evaluation"
QLORA_DIR="${BASE_DIR}/qlora_experiments"
RESULTS_DIR="${QLORA_DIR}/results"
KVASIR_IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
ENDOVIS_IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"

# Base model
BASE_MODEL="microsoft/llava-med-v1.5-mistral-7b"

# Test data paths
KVASIR_TEST_JSONL="${BASE_DIR}/datasets/kvasir_ULTRA_CONDENSED/test_CATEGORY_BASED.jsonl"
KVASIR_TEST_JSON="/tmp/zeroshot_kvasir_llava_${SLURM_JOB_ID}.json"
ENDOVIS_TEST_JSONL="${BASE_DIR}/datasets/endovis2018_vqa/test.jsonl"
ENDOVIS_TEST_JSON="/tmp/zeroshot_endovis_llava_${SLURM_JOB_ID}.json"

# Training data paths
KVASIR_TRAIN_DATA="${BASE_DIR}/datasets/kvasir_ULTRA_CONDENSED/train_CATEGORY_BASED.jsonl"
ENDOVIS_TRAIN_DATA="${BASE_DIR}/datasets/endovis2018_vqa/train.jsonl"

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${QLORA_DIR}/slurm/logs"
mkdir -p "${QLORA_DIR}/models/llava_med_kvasir_instruction"
mkdir -p "${QLORA_DIR}/models/llava_med_endovis_instruction"

# Load modules
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1

# Shared cache configuration
export SHARED_CACHE="/l/users/muhra.almahri/.cache/hf_shared"
export HF_HOME=$SHARED_CACHE
export TRANSFORMERS_CACHE=$SHARED_CACHE/transformers
export HF_DATASETS_CACHE=$SHARED_CACHE/datasets
export HF_HUB_CACHE=$SHARED_CACHE
export TORCH_HOME=$SHARED_CACHE/torch

# Aggressively disable DeepSpeed
unset DS_SKIP_CUDA_CHECK
unset DS_ACCELERATOR
unset DEEPSPEED_CONFIG_FILE
export ACCELERATE_USE_DEEPSPEED=false
export DEEPSPEED_DISABLED=true
export ACCELERATE_USE_CUDA=true
export ACCELERATE_USE_CPU=false

# CUDA settings
export CUDNN_DETERMINISTIC=0
export CUDA_LAUNCH_BLOCKING=1
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH

# Hugging Face authentication
export HF_TOKEN="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"

# ============================================================================
# GPU 0: Kvasir-VQA (Zero-shot + Training)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu0
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu0
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 0: Kvasir-VQA Pipeline"
    echo "=========================================="
    echo "Started: $(date)"
    
    # Step 1: Zero-shot evaluation
    echo ""
    echo "--- Step 1: Zero-Shot Evaluation ---"
    echo "Started: $(date)"
    
    # Convert JSONL to JSON
    python3 << PYTHON_EOF
import json
import sys

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
    
    # Step 2: Instruction fine-tuning
    echo ""
    echo "--- Step 2: Instruction Fine-tuning ---"
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
    
    echo ""
    echo "✓ GPU 0 (Kvasir-VQA) Pipeline Complete: $(date)"
) 2>&1 | sed 's/^/[GPU 0] /' &
KVASIR_PID=$!

# ============================================================================
# GPU 1: EndoVis2018 (Zero-shot + Training)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu1
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu1
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 1: EndoVis2018 Pipeline"
    echo "=========================================="
    echo "Started: $(date)"
    
    # Step 1: Zero-shot evaluation
    echo ""
    echo "--- Step 1: Zero-Shot Evaluation ---"
    echo "Started: $(date)"
    
    # Convert JSONL to JSON
    python3 << PYTHON_EOF
import json
import sys

jsonl_path = "${ENDOVIS_TEST_JSONL}"
json_path = "${ENDOVIS_TEST_JSON}"

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
        --test_data "${ENDOVIS_TEST_JSON}" \
        --image_dir "${ENDOVIS_IMAGE_DIR}" \
        --output "${RESULTS_DIR}/endovis_zeroshot_llava_med_v15.json" \
        --use_instruction
    
    ENDOVIS_ZEROSHOT_STATUS=$?
    if [ $ENDOVIS_ZEROSHOT_STATUS -eq 0 ]; then
        echo "✓ EndoVis zero-shot evaluation completed: $(date)"
        echo "  Results: ${RESULTS_DIR}/endovis_zeroshot_llava_med_v15.json"
    else
        echo "✗ EndoVis zero-shot evaluation failed: $(date) (exit code: $ENDOVIS_ZEROSHOT_STATUS)"
        exit $ENDOVIS_ZEROSHOT_STATUS
    fi
    
    # Clean up temporary file
    rm -f "${ENDOVIS_TEST_JSON}"
    
    # Step 2: Instruction fine-tuning
    echo ""
    echo "--- Step 2: Instruction Fine-tuning ---"
    echo "Started: $(date)"
    
    cd "${QLORA_DIR}"
    
    # Set dataset environment variables for EndoVis2018
    export DATASET_NAME=endovis
    export DATA_ROOT="${BASE_DIR}/datasets/endovis2018_vqa"
    export OUTPUT_DIR="${QLORA_DIR}/models/llava_med_endovis_instruction"
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Run instruction fine-tuning
    python3 train_llava_NATIVE_PROCESSOR.py
    
    ENDOVIS_TRAIN_STATUS=$?
    if [ $ENDOVIS_TRAIN_STATUS -eq 0 ]; then
        echo "✓ EndoVis instruction fine-tuning completed: $(date)"
        echo "  Model: ${OUTPUT_DIR}"
    else
        echo "✗ EndoVis instruction fine-tuning failed: $(date) (exit code: $ENDOVIS_TRAIN_STATUS)"
        exit $ENDOVIS_TRAIN_STATUS
    fi
    
    echo ""
    echo "✓ GPU 1 (EndoVis2018) Pipeline Complete: $(date)"
) 2>&1 | sed 's/^/[GPU 1] /' &
ENDOVIS_PID=$!

# ============================================================================
# Wait for all jobs to complete
# ============================================================================
echo ""
echo "=========================================="
echo "All Jobs Started"
echo "=========================================="
echo "GPU 0 (Kvasir Pipeline): PID ${KVASIR_PID}"
echo "GPU 1 (EndoVis Pipeline): PID ${ENDOVIS_PID}"
echo ""
echo "Waiting for all jobs to complete..."
echo ""

# Wait for all background jobs
wait $KVASIR_PID
KVASIR_STATUS=$?

wait $ENDOVIS_PID
ENDOVIS_STATUS=$?

# Extract individual statuses
if [ $KVASIR_STATUS -eq 0 ]; then
    KVASIR_ZEROSHOT_STATUS=0
    KVASIR_TRAIN_STATUS=0
else
    KVASIR_ZEROSHOT_STATUS=$KVASIR_STATUS
    KVASIR_TRAIN_STATUS=$KVASIR_STATUS
fi

if [ $ENDOVIS_STATUS -eq 0 ]; then
    ENDOVIS_ZEROSHOT_STATUS=0
    ENDOVIS_TRAIN_STATUS=0
else
    ENDOVIS_ZEROSHOT_STATUS=$ENDOVIS_STATUS
    ENDOVIS_TRAIN_STATUS=$ENDOVIS_STATUS
fi

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "ALL LLaVA-Med v1.5 EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Status:"
if [ $KVASIR_ZEROSHOT_STATUS -eq 0 ]; then
    echo "  ✓ Kvasir Zero-shot: SUCCESS"
else
    echo "  ✗ Kvasir Zero-shot: FAILED (exit code: $KVASIR_ZEROSHOT_STATUS)"
fi

if [ $ENDOVIS_ZEROSHOT_STATUS -eq 0 ]; then
    echo "  ✓ EndoVis Zero-shot: SUCCESS"
else
    echo "  ✗ EndoVis Zero-shot: FAILED (exit code: $ENDOVIS_ZEROSHOT_STATUS)"
fi

if [ $KVASIR_TRAIN_STATUS -eq 0 ]; then
    echo "  ✓ Kvasir Instruction Fine-tuning: SUCCESS"
else
    echo "  ✗ Kvasir Instruction Fine-tuning: FAILED (exit code: $KVASIR_TRAIN_STATUS)"
fi

if [ $ENDOVIS_TRAIN_STATUS -eq 0 ]; then
    echo "  ✓ EndoVis Instruction Fine-tuning: SUCCESS"
else
    echo "  ✗ EndoVis Instruction Fine-tuning: FAILED (exit code: $ENDOVIS_TRAIN_STATUS)"
fi

echo ""
echo "Results saved to:"
echo "  Kvasir Zero-shot: ${RESULTS_DIR}/kvasir_zeroshot_llava_med_v15.json"
echo "  EndoVis Zero-shot: ${RESULTS_DIR}/endovis_zeroshot_llava_med_v15.json"
echo "  Kvasir Model: ${QLORA_DIR}/models/llava_med_kvasir_instruction"
echo "  EndoVis Model: ${QLORA_DIR}/models/llava_med_endovis_instruction"
echo "=========================================="

# Exit with error if any job failed
if [ $KVASIR_ZEROSHOT_STATUS -ne 0 ] || [ $ENDOVIS_ZEROSHOT_STATUS -ne 0 ] || [ $KVASIR_TRAIN_STATUS -ne 0 ] || [ $ENDOVIS_TRAIN_STATUS -ne 0 ]; then
    exit 1
fi

exit 0

