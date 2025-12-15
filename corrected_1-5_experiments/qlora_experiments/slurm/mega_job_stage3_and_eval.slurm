#!/bin/bash
#SBATCH --job-name=qlora_mega_s3_eval
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_s3_eval_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_s3_eval_%j.err

# ============================================================================
# MEGA JOB: 4 GPUs Running in Parallel
# ============================================================================
# GPU 0: Exp3-S3 - Training (CXRTrek Stage 3)
# GPU 1: Exp4-S3 - Training (Curriculum Stage 3)
# GPU 2: Exp2 - Evaluation (re-run with fixed script, independent)
# GPU 3: Exp3 - Evaluation (waits for Exp3-S3 to complete)
# After Exp4-S3 completes: Exp4 - Evaluation (on GPU 1)
# ============================================================================
# NOTE: This job runs AFTER Exp3-S2 and Exp4-S2 (from job 156712) complete
# ============================================================================

echo "=========================================="
echo "MEGA JOB - Stage 3 Training + Evaluations"
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
    eval ${COMMAND} > ${BASE_DIR}/slurm/logs/mega_s3_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_s3_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
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
# GPU 0: Exp3-S3 Training
# ============================================================================
EXP3_S3_CONFIG="${BASE_DIR}/configs/exp3_stage3.yaml"
EXP3_S2_CHECKPOINT="${BASE_DIR}/models/exp3_cxrtrek/stage2"
# Note: Exp3 stages are independent, but we verify Exp3-S2 exists
if [ -d "${EXP3_S2_CHECKPOINT}" ]; then
    echo "Verified: Exp3-S2 checkpoint exists at ${EXP3_S2_CHECKPOINT}"
else
    echo "WARNING: Exp3-S2 checkpoint not found at ${EXP3_S2_CHECKPOINT}"
    echo "Proceeding with Exp3-S3 (independent model)..."
fi
EXP3_S3_CMD="python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP3_S3_CONFIG}"

# ============================================================================
# GPU 1: Exp4-S3 Training (depends on Exp4-S2 checkpoint)
# ============================================================================
EXP4_S3_CONFIG="${BASE_DIR}/configs/exp4_stage3.yaml"
EXP4_S2_CHECKPOINT="${BASE_DIR}/models/exp4_curriculum/stage2"
if [ ! -d "${EXP4_S2_CHECKPOINT}" ]; then
    echo "ERROR: Exp4-S2 checkpoint not found at ${EXP4_S2_CHECKPOINT}"
    echo "This job requires Exp4-S2 (from job 156712) to complete first!"
    exit 1
fi
echo "Verified: Exp4-S2 checkpoint exists at ${EXP4_S2_CHECKPOINT}"
EXP4_S3_CMD="python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP4_S3_CONFIG}"

# ============================================================================
# GPU 2: Exp2 Evaluation (re-run with fixed script, independent)
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
# GPU 3: Exp3 Evaluation (waits for Exp3-S3 to complete)
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

# Exp3 Evaluation command (will run after training completes)
EXP3_EVAL_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp3.py \
    --model_stage1 ${EXP3_S1_MODEL} \
    --model_stage2 ${EXP3_S2_MODEL} \
    --model_stage3 ${EXP3_S3_MODEL} \
    --test_data ${EXP3_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP3_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# GPU 3: Exp4 Evaluation (waits for Exp4-S3 to complete)
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

# Exp4 Evaluation command (will run after training completes)
EXP4_EVAL_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp4.py \
    --model_path ${EXP4_S3_MODEL} \
    --test_data ${EXP4_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP4_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# Phase 1: Run training and Exp2 evaluation in parallel (all 4 GPUs)
# ============================================================================
echo "Phase 1: Starting training jobs and Exp2 evaluation in parallel..."
echo "=========================================="

run_experiment 0 "Exp3-S3 - Training" "${EXP3_S3_CMD}" &
PID0=$!

run_experiment 1 "Exp4-S3 - Training" "${EXP4_S3_CMD}" &
PID1=$!

run_experiment 2 "Exp2 - Evaluation (re-run)" "${EXP2_CMD}" &
PID2=$!

# Wait for Exp2 evaluation to complete (independent, doesn't need training)
wait ${PID2}
EXIT2=$?

# Wait for both training jobs to complete
echo ""
echo "Waiting for training jobs to complete..."
echo "=========================================="

wait ${PID0}
EXIT0=$?

wait ${PID1}
EXIT1=$?

# Verify checkpoints exist before running evaluations
echo ""
echo "Verifying checkpoints before running evaluations..."
if [ ! -d "${EXP3_S3_MODEL}" ]; then
    echo "ERROR: Exp3-S3 checkpoint not found at ${EXP3_S3_MODEL}"
    EXIT0=1
fi

if [ ! -d "${EXP4_S3_MODEL}" ]; then
    echo "ERROR: Exp4-S3 checkpoint not found at ${EXP4_S3_MODEL}"
    EXIT1=1
fi

# ============================================================================
# Phase 2: Run Exp3 and Exp4 evaluations in parallel (GPU 2 and 3)
# ============================================================================
if [ ${EXIT0} -eq 0 ] && [ ${EXIT1} -eq 0 ]; then
    echo ""
    echo "Phase 2: Starting Exp3 and Exp4 evaluation jobs in parallel..."
    echo "=========================================="
    
    run_experiment 2 "Exp3 - Evaluation" "${EXP3_EVAL_CMD}" &
    PID3_EVAL=$!
    
    run_experiment 3 "Exp4 - Evaluation" "${EXP4_EVAL_CMD}" &
    PID4_EVAL=$!
    
    # Wait for both evaluation jobs to complete
    echo ""
    echo "Waiting for evaluation jobs to complete..."
    echo "=========================================="
    
    wait ${PID3_EVAL}
    EXIT3=$?
    
    wait ${PID4_EVAL}
    EXIT4=$?
else
    echo ""
    echo "ERROR: Training failed, skipping evaluations..."
    EXIT3=1
    EXIT4=1
fi

# Cleanup temporary files
rm -f ${EXP2_TEST_JSON} ${EXP3_TEST_JSON} ${EXP4_TEST_JSON}

# Report final status
echo ""
echo "=========================================="
echo "MEGA JOB - ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Exp3-S3 Training (GPU 0): Exit code ${EXIT0}"
echo "Exp4-S3 Training (GPU 1): Exit code ${EXIT1}"
echo "Exp2 Evaluation (GPU 2, re-run): Exit code ${EXIT2}"
echo "Exp3 Evaluation (GPU 2/3): Exit code ${EXIT3}"
echo "Exp4 Evaluation (GPU 3): Exit code ${EXIT4}"
echo "End time: $(date)"
echo "=========================================="

# Exit with error if any experiment failed
if [ ${EXIT0} -ne 0 ] || [ ${EXIT1} -ne 0 ] || [ ${EXIT2} -ne 0 ] || [ ${EXIT3} -ne 0 ] || [ ${EXIT4} -ne 0 ]; then
    echo "ERROR: One or more experiments failed!"
    exit 1
else
    echo "SUCCESS: All experiments completed successfully!"
    exit 0
fi

