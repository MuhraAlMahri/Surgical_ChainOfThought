#!/bin/bash
#SBATCH --job-name=zeroshot_all
#SBATCH --output=slurm/logs/zeroshot_all_%j.out
#SBATCH --error=slurm/logs/zeroshot_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:3
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Zero-Shot Evaluation for All Experiments
# Evaluates the base model WITHOUT any fine-tuning
# Runs all experiments in parallel on 5 GPUs

set -e

echo "=========================================="
echo "ZERO-SHOT EVALUATION: All Experiments"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 3 (running in batches)"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
SCRIPT_DIR="${BASE_DIR}/scripts/evaluation"
QLORA_DIR="${BASE_DIR}/qlora_experiments"
RESULTS_DIR="${QLORA_DIR}/results"
IMAGE_DIR="${BASE_DIR}/datasets/kvasir_ULTRA_CONDENSED/images"

# Base model
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Test data paths (all experiments use same test data)
TEST_DATA_JSONL="${BASE_DIR}/datasets/qlora_experiments/exp1_random/test.jsonl"
TEST_DATA_JSON="/tmp/zeroshot_test_${SLURM_JOB_ID}.json"

# Convert JSONL to JSON
echo "Converting test data from JSONL to JSON..."
python3 << 'PYTHON_EOF'
import json
import sys

jsonl_path = sys.argv[1]
json_path = sys.argv[2]

with open(jsonl_path, 'r') as f_in, open(json_path, 'w') as f_out:
    data = [json.loads(line) for line in f_in]
    json.dump(data, f_out, indent=2)

print(f"Converted {len(data)} samples from JSONL to JSON")
PYTHON_EOF
"${TEST_DATA_JSONL}" "${TEST_DATA_JSON}"

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

# Run all experiments in parallel
echo ""
echo "Starting zero-shot evaluations in parallel..."
echo ""

# GPU 0: Exp1 - Zero-Shot
run_zeroshot_eval 0 "Exp1-ZeroShot" "${RESULTS_DIR}/exp1_zeroshot.json" &

# GPU 1: Exp2 - Zero-Shot
run_zeroshot_eval 1 "Exp2-ZeroShot" "${RESULTS_DIR}/exp2_zeroshot.json" &

# GPU 2: Exp3 - Zero-Shot
run_zeroshot_eval 2 "Exp3-ZeroShot" "${RESULTS_DIR}/exp3_zeroshot.json" &

# Wait for first batch
wait

echo "First batch (Exp1-3) completed. Starting second batch..."

# GPU 0: Exp4 - Zero-Shot
run_zeroshot_eval 0 "Exp4-ZeroShot" "${RESULTS_DIR}/exp4_zeroshot.json" &

# GPU 1: Exp5 - Zero-Shot
run_zeroshot_eval 1 "Exp5-ZeroShot" "${RESULTS_DIR}/exp5_zeroshot.json" &

# Wait for all jobs to complete
wait

echo ""
echo "=========================================="
echo "ALL ZERO-SHOT EVALUATIONS COMPLETED"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Exp1: ${RESULTS_DIR}/exp1_zeroshot.json"
echo "  - Exp2: ${RESULTS_DIR}/exp2_zeroshot.json"
echo "  - Exp3: ${RESULTS_DIR}/exp3_zeroshot.json"
echo "  - Exp4: ${RESULTS_DIR}/exp4_zeroshot.json"
echo "  - Exp5: ${RESULTS_DIR}/exp5_zeroshot.json"
echo ""
echo "These results show the baseline performance of the"
echo "base model WITHOUT any fine-tuning."
echo "=========================================="

# Clean up temporary file
rm -f "${TEST_DATA_JSON}"

echo "Zero-shot evaluation job completed successfully!"

