#!/bin/bash
#SBATCH --job-name=qlora_evals_2gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/evals_2gpu_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/evals_2gpu_%j.err

# ============================================================================
# EVALUATIONS - 2 GPUs
# ============================================================================
# GPU 0: Exp2 - Evaluation (re-run with fixed script, independent)
# GPU 1: Exp3 - Evaluation (waits for Exp3-S3 checkpoint)
# After Exp3 eval completes: Exp4 - Evaluation (on GPU 1)
# ============================================================================
# NOTE: Exp3 and Exp4 evaluations depend on Stage 3 training completing
# ============================================================================

echo "=========================================="
echo "EVALUATIONS - 2 GPUs"
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
    eval ${COMMAND} > ${BASE_DIR}/slurm/logs/evals_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/evals_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
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
# GPU 0: Exp2 Evaluation (re-run with fixed script, independent)
# ============================================================================
EXP2_MODEL="${BASE_DIR}/models/exp2_qwen_reordered"
EXP2_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp2_qwen_reordered/test.jsonl"
EXP2_OUTPUT="${RESULTS_DIR}/exp2_evaluation.json"

# Convert JSONL to JSON for evaluation script
EXP2_TEST_JSON="/tmp/exp2_test_${SLURM_JOB_ID}.json"
if [ -f "${EXP2_TEST_DATA}" ]; then
    echo "Converting Exp2 test data from JSONL to JSON..."
    python3 -c "
import json
with open('${EXP2_TEST_DATA}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${EXP2_TEST_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"
fi

EXP2_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp2.py \
    --model_path ${EXP2_MODEL} \
    --test_data ${EXP2_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP2_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# GPU 1: Exp3 Evaluation (waits for Exp3-S3 checkpoint)
# ============================================================================
EXP3_S1_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage1"
EXP3_S2_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage2"
EXP3_S3_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage3"
EXP3_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl"
EXP3_OUTPUT="${RESULTS_DIR}/exp3_evaluation.json"

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

# Exp3 Evaluation command
EXP3_EVAL_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp3.py \
    --model_stage1 ${EXP3_S1_MODEL} \
    --model_stage2 ${EXP3_S2_MODEL} \
    --model_stage3 ${EXP3_S3_MODEL} \
    --test_data ${EXP3_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP3_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# GPU 1: Exp4 Evaluation (waits for Exp4-S3 checkpoint)
# ============================================================================
EXP4_S3_MODEL="${BASE_DIR}/models/exp4_curriculum/stage3"
EXP4_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl"
EXP4_OUTPUT="${RESULTS_DIR}/exp4_evaluation.json"

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

# Exp4 Evaluation command
EXP4_EVAL_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp4.py \
    --model_path ${EXP4_S3_MODEL} \
    --test_data ${EXP4_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP4_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# Phase 1: Run Exp2 evaluation (independent) and Exp3 evaluation in parallel
# ============================================================================
echo "Phase 1: Starting Exp2 and Exp3 evaluations in parallel..."
echo "=========================================="

# Check if Exp3-S3 checkpoint exists
if [ ! -d "${EXP3_S3_MODEL}" ]; then
    echo "WARNING: Exp3-S3 checkpoint not found at ${EXP3_S3_MODEL}"
    echo "Exp3 evaluation will be skipped. Waiting for training job to complete..."
    EXP3_SKIP=1
else
    echo "✓ Exp3-S3 checkpoint found: ${EXP3_S3_MODEL}"
    EXP3_SKIP=0
fi

# Check if Exp4-S3 checkpoint exists
if [ ! -d "${EXP4_S3_MODEL}" ]; then
    echo "WARNING: Exp4-S3 checkpoint not found at ${EXP4_S3_MODEL}"
    echo "Exp4 evaluation will be skipped. Waiting for training job to complete..."
    EXP4_SKIP=1
else
    echo "✓ Exp4-S3 checkpoint found: ${EXP4_S3_MODEL}"
    EXP4_SKIP=0
fi

# Run Exp2 evaluation (independent, can always run)
run_experiment 0 "Exp2 - Evaluation (re-run)" "${EXP2_CMD}" &
PID2=$!

# Run Exp3 evaluation if checkpoint exists
if [ ${EXP3_SKIP} -eq 0 ]; then
    run_experiment 1 "Exp3 - Evaluation" "${EXP3_EVAL_CMD}" &
    PID3=$!
else
    PID3=""
fi

# Wait for Exp2 to complete
wait ${PID2}
EXIT2=$?

# Wait for Exp3 to complete (if it ran)
if [ -n "${PID3}" ]; then
    wait ${PID3}
    EXIT3=$?
else
    EXIT3=0
    echo "Exp3 evaluation skipped (checkpoint not ready)"
fi

# ============================================================================
# Phase 2: Run Exp4 evaluation (after Exp3 completes, on GPU 1)
# ============================================================================
if [ ${EXP4_SKIP} -eq 0 ]; then
    echo ""
    echo "Phase 2: Starting Exp4 evaluation..."
    echo "=========================================="
    
    run_experiment 1 "Exp4 - Evaluation" "${EXP4_EVAL_CMD}"
    EXIT4=$?
else
    echo ""
    echo "Exp4 evaluation skipped (checkpoint not ready)"
    EXIT4=0
fi

# Cleanup temporary files
rm -f ${EXP2_TEST_JSON} ${EXP3_TEST_JSON} ${EXP4_TEST_JSON}

# Report final status
echo ""
echo "=========================================="
echo "EVALUATIONS - COMPLETED"
echo "=========================================="
echo "Exp2 Evaluation (GPU 0, re-run): Exit code ${EXIT2}"
echo "Exp3 Evaluation (GPU 1): Exit code ${EXIT3}"
echo "Exp4 Evaluation (GPU 1): Exit code ${EXIT4}"
echo "End time: $(date)"
echo "=========================================="

# Exit with error if any evaluation failed
if [ ${EXIT2} -ne 0 ] || [ ${EXIT3} -ne 0 ] || [ ${EXIT4} -ne 0 ]; then
    echo "ERROR: One or more evaluations failed!"
    exit 1
else
    echo "SUCCESS: All evaluations completed successfully!"
    exit 0
fi







