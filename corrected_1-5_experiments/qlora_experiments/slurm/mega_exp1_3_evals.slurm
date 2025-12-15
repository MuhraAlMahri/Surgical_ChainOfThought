#!/bin/bash
#SBATCH --job-name=mega_exp1_3_evals
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_exp1_3_evals_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_exp1_3_evals_%j.err

# ============================================================================
# MEGA JOB: Exp1, Exp2, Exp3 Evaluations Only
# ============================================================================
# Runs in parallel on 3 GPUs:
#   GPU 0: Exp1 - Evaluation
#   GPU 1: Exp2 - Evaluation
#   GPU 2: Exp3 - Evaluation (needs 3 stage models)
# ============================================================================

echo "=========================================="
echo "MEGA JOB: Exp1, Exp2, Exp3 Evaluations"
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
    eval ${COMMAND} > ${BASE_DIR}/slurm/logs/mega_exp1_3_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_exp1_3_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
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
# Prepare test data (convert JSONL to JSON)
# ============================================================================
TEST_DATA_JSONL="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl"
TEST_DATA_JSON="/tmp/test_data_${SLURM_JOB_ID}.json"

if [ -f "${TEST_DATA_JSONL}" ]; then
    echo "Converting test data from JSONL to JSON..."
    python3 -c "
import json
with open('${TEST_DATA_JSONL}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${TEST_DATA_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"
else
    echo "ERROR: Test data not found at ${TEST_DATA_JSONL}"
    exit 1
fi

# ============================================================================
# GPU 0: Exp1 Evaluation
# ============================================================================
EXP1_MODEL="${BASE_DIR}/models/exp1_random"
EXP1_OUTPUT="${RESULTS_DIR}/exp1_evaluation.json"

EXP1_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp1.py \
    --model_path ${EXP1_MODEL} \
    --test_data ${TEST_DATA_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP1_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# GPU 1: Exp2 Evaluation
# ============================================================================
EXP2_MODEL="${BASE_DIR}/models/exp2_qwen_reordered"
EXP2_OUTPUT="${RESULTS_DIR}/exp2_evaluation.json"

EXP2_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp2.py \
    --model_path ${EXP2_MODEL} \
    --test_data ${TEST_DATA_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP2_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# GPU 2: Exp3 Evaluation
# ============================================================================
EXP3_S1_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage1"
EXP3_S2_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage2"
EXP3_S3_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage3"
EXP3_OUTPUT="${RESULTS_DIR}/exp3_evaluation.json"

# Verify Exp3 models exist
if [ ! -d "${EXP3_S1_MODEL}" ] || [ ! -d "${EXP3_S2_MODEL}" ] || [ ! -d "${EXP3_S3_MODEL}" ]; then
    echo "WARNING: One or more Exp3 models not found. Skipping Exp3 evaluation."
    EXP3_SKIP=1
else
    EXP3_SKIP=0
    EXP3_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp3.py \
        --model_stage1 ${EXP3_S1_MODEL} \
        --model_stage2 ${EXP3_S2_MODEL} \
        --model_stage3 ${EXP3_S3_MODEL} \
        --test_data ${TEST_DATA_JSON} \
        --image_dir ${IMAGE_ROOT} \
        --output ${EXP3_OUTPUT} \
        --base_model Qwen/Qwen3-VL-8B-Instruct"
fi

# ============================================================================
# Run all evaluations in parallel
# ============================================================================
echo ""
echo "=========================================="
echo "Starting Exp1, Exp2, Exp3 evaluations in parallel"
echo "=========================================="

# Start evaluations in parallel
run_experiment 0 "Exp1 - Evaluation" "${EXP1_CMD}" &
PID0=$!

run_experiment 1 "Exp2 - Evaluation" "${EXP2_CMD}" &
PID1=$!

if [ ${EXP3_SKIP} -eq 0 ]; then
    run_experiment 2 "Exp3 - Evaluation" "${EXP3_CMD}" &
    PID2=$!
else
    echo "Skipping Exp3 evaluation (models not found)"
    PID2=""
fi

# Wait for all evaluations to complete
echo ""
echo "Waiting for evaluations to complete..."
wait ${PID0}
EXIT0=$?

wait ${PID1}
EXIT1=$?

if [ -n "${PID2}" ]; then
    wait ${PID2}
    EXIT2=$?
else
    EXIT2=0
fi

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "MEGA JOB COMPLETED"
echo "=========================================="
echo "Evaluations:"
echo "  Exp1: Exit code ${EXIT0}"
echo "  Exp2: Exit code ${EXIT1}"
if [ ${EXP3_SKIP} -eq 0 ]; then
    echo "  Exp3: Exit code ${EXIT2}"
else
    echo "  Exp3: Skipped (models not found)"
fi
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "End time: $(date)"
echo "=========================================="

# Cleanup
rm -f ${TEST_DATA_JSON}
rm -rf /tmp/hf_cache_${SLURM_JOB_ID}_*
rm -rf /tmp/triton_cache_${SLURM_JOB_ID}_*

# Exit with worst exit code
if [ ${EXIT0} -ne 0 ] || [ ${EXIT1} -ne 0 ] || [ ${EXIT2} -ne 0 ]; then
    exit 1
else
    exit 0
fi

