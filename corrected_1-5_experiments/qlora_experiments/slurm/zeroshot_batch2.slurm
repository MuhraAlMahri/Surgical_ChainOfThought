#!/bin/bash
#SBATCH --job-name=zeroshot_b2
#SBATCH --output=slurm/logs/zeroshot_batch2_%j.out
#SBATCH --error=slurm/logs/zeroshot_batch2_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Zero-Shot Evaluation: Batch 2 (Exp5)
# Evaluates the base model WITHOUT any fine-tuning

set -e

echo "=========================================="
echo "ZERO-SHOT EVALUATION: Batch 2"
echo "Exp5"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4 (using GPU 0 for Exp5)"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
SCRIPT_DIR="${BASE_DIR}/scripts/evaluation"
QLORA_DIR="${BASE_DIR}/qlora_experiments"
RESULTS_DIR="${QLORA_DIR}/results"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"

# Base model
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Test data paths
TEST_DATA_JSONL="${BASE_DIR}/datasets/qlora_experiments/exp1_random/test.jsonl"
TEST_DATA_JSON="/tmp/zeroshot_b2_test_${SLURM_JOB_ID}.json"

# Convert JSONL to JSON
echo "Converting test data from JSONL to JSON..."
python3 << PYTHON_EOF
import json
import sys

jsonl_path = "${TEST_DATA_JSONL}"
json_path = "${TEST_DATA_JSON}"

with open(jsonl_path, 'r') as f_in, open(json_path, 'w') as f_out:
    data = [json.loads(line) for line in f_in]
    json.dump(data, f_out, indent=2)

print(f"Converted {len(data)} samples from JSONL to JSON")
PYTHON_EOF

# Function to run zero-shot evaluation
run_zeroshot_eval() {
    local GPU_ID=$1
    local EXP_NAME=$2
    local OUTPUT_FILE=$3
    
    echo "=========================================="
    echo "GPU ${GPU_ID}: ${EXP_NAME} - Zero-Shot Evaluation"
    echo "=========================================="
    
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    
    python3 "${SCRIPT_DIR}/evaluate_zeroshot.py" \
        --base_model "${BASE_MODEL}" \
        --test_data "${TEST_DATA_JSON}" \
        --image_dir "${IMAGE_DIR}" \
        --output "${OUTPUT_FILE}" \
        --use_instruction
    
    echo "✓ ${EXP_NAME} zero-shot evaluation completed"
    echo "  Results saved to: ${OUTPUT_FILE}"
}

# Run Exp5
echo ""
echo "Starting Exp5 zero-shot evaluation..."
echo ""

# GPU 0: Exp5 - Zero-Shot
run_zeroshot_eval 0 "Exp5-ZeroShot" "${RESULTS_DIR}/exp5_zeroshot.json"

echo ""
echo "=========================================="
echo "BATCH 2 ZERO-SHOT EVALUATION COMPLETED"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Exp5: ${RESULTS_DIR}/exp5_zeroshot.json"
echo "=========================================="

# Clean up temporary file
rm -f "${TEST_DATA_JSON}"

echo "Batch 2 zero-shot evaluation completed successfully!"

