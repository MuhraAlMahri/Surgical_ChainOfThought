#!/bin/bash
#SBATCH --job-name=exp3_exp4_eval_2gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp3_exp4_eval_2gpu_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp3_exp4_eval_2gpu_%j.err

# ============================================================================
# Exp3 and Exp4 Evaluations - 2 GPUs
# ============================================================================
# GPU 0: Exp3 - Evaluation (needs Exp3-S3 checkpoint)
# GPU 1: Exp4 - Evaluation (needs Exp4-S3 checkpoint)
# ============================================================================
# NOTE: This job depends on Stage 3 training completing
# ============================================================================

echo "=========================================="
echo "Exp3 and Exp4 Evaluations - 2 GPUs"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Setup
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"
SCRIPTS_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/scripts/evaluation"
IMAGE_ROOT="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
RESULTS_DIR="${BASE_DIR}/results"

# Create results directory
mkdir -p ${RESULTS_DIR}

# Function to run experiment on a specific GPU
run_experiment() {
    local GPU_ID=$1
    local EXP_NAME=$2
    local COMMAND=$3
    
    echo ""
    echo "=========================================="
    echo "Starting ${EXP_NAME} on GPU ${GPU_ID}"
    echo "=========================================="
    
    # Set GPU-specific environment
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu${GPU_ID}
    export TRANSFORMERS_CACHE=${HF_HOME}
    export HF_HUB_CACHE=${HF_HOME}
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu${GPU_ID}
    mkdir -p ${HF_HOME}
    
    # Run command
    eval ${COMMAND} > ${BASE_DIR}/slurm/logs/exp3_exp4_eval_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/exp3_exp4_eval_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
    local EXIT_CODE=$?
    
    echo "=========================================="
    echo "${EXP_NAME} on GPU ${GPU_ID} completed with exit code: ${EXIT_CODE}"
    echo "End time: $(date)"
    echo "=========================================="
    
    # Cleanup
    rm -rf ${HF_HOME}
    rm -rf ${TRITON_CACHE_DIR}
    
    return ${EXIT_CODE}
}

# Show GPU status
nvidia-smi
echo ""

# ============================================================================
# GPU 0: Exp3 Evaluation
# ============================================================================
EXP3_S1_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage1"
EXP3_S2_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage2"
EXP3_S3_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage3"
EXP3_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl"
EXP3_OUTPUT="${RESULTS_DIR}/exp3_evaluation.json"

# Verify checkpoint exists
if [ ! -d "${EXP3_S3_MODEL}" ]; then
    echo "ERROR: Exp3-S3 checkpoint not found at ${EXP3_S3_MODEL}"
    echo "This job requires Stage 3 training to complete first!"
    exit 1
fi
echo "✓ Exp3-S3 checkpoint found: ${EXP3_S3_MODEL}"

# Convert JSONL to JSON for evaluation script
EXP3_TEST_JSON="/tmp/exp3_test_${SLURM_JOB_ID}.json"
if [ -f "${EXP3_TEST_DATA}" ]; then
    echo "Converting Exp3 test data from JSONL to JSON..."
    python3 -c "
import json
with open('${EXP3_TEST_DATA}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${EXP3_TEST_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"
fi

EXP3_EVAL_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp3.py \
    --model_stage1 ${EXP3_S1_MODEL} \
    --model_stage2 ${EXP3_S2_MODEL} \
    --model_stage3 ${EXP3_S3_MODEL} \
    --test_data ${EXP3_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP3_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# GPU 1: Exp4 Evaluation
# ============================================================================
EXP4_S3_MODEL="${BASE_DIR}/models/exp4_curriculum/stage3"
EXP4_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl"
EXP4_OUTPUT="${RESULTS_DIR}/exp4_evaluation.json"

# Verify checkpoint exists
if [ ! -d "${EXP4_S3_MODEL}" ]; then
    echo "ERROR: Exp4-S3 checkpoint not found at ${EXP4_S3_MODEL}"
    echo "This job requires Stage 3 training to complete first!"
    exit 1
fi
echo "✓ Exp4-S3 checkpoint found: ${EXP4_S3_MODEL}"

# Convert JSONL to JSON for evaluation script
EXP4_TEST_JSON="/tmp/exp4_test_${SLURM_JOB_ID}.json"
if [ -f "${EXP4_TEST_DATA}" ]; then
    echo "Converting Exp4 test data from JSONL to JSON..."
    python3 -c "
import json
with open('${EXP4_TEST_DATA}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${EXP4_TEST_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"
fi

EXP4_EVAL_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp4.py \
    --model_path ${EXP4_S3_MODEL} \
    --test_data ${EXP4_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP4_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# Run both evaluations in parallel
# ============================================================================
echo "Starting Exp3 and Exp4 evaluations in parallel..."
echo "=========================================="

run_experiment 0 "Exp3 - Evaluation" "${EXP3_EVAL_CMD}" &
PID3=$!

run_experiment 1 "Exp4 - Evaluation" "${EXP4_EVAL_CMD}" &
PID4=$!

# Wait for both evaluations to complete
wait ${PID3}
EXIT3=$?

wait ${PID4}
EXIT4=$?

# Cleanup temporary files
rm -f ${EXP3_TEST_JSON} ${EXP4_TEST_JSON}

# Report final status
echo ""
echo "=========================================="
echo "Exp3 and Exp4 Evaluations - COMPLETED"
echo "=========================================="
echo "Exp3 Evaluation (GPU 0): Exit code ${EXIT3}"
echo "Exp4 Evaluation (GPU 1): Exit code ${EXIT4}"
echo "End time: $(date)"
echo "=========================================="

# Exit with error if any evaluation failed
if [ ${EXIT3} -ne 0 ] || [ ${EXIT4} -ne 0 ]; then
    echo "ERROR: One or more evaluations failed!"
    exit 1
else
    echo "SUCCESS: All evaluations completed successfully!"
    exit 0
fi







