#!/bin/bash
#SBATCH --job-name=mega_exp4_retrain_evals
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_exp4_retrain_evals_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_exp4_retrain_evals_%j.err

# ============================================================================
# MEGA JOB: Exp4 Retraining + All Evaluations
# ============================================================================
# Phase 1 (Parallel):
#   GPU 0: Exp1 - Evaluation (independent)
#   GPU 1: Exp2 - Evaluation (independent)
#   GPU 2: Exp3 - Evaluation (independent, needs 3 models)
#   GPU 3: Exp4 - Training Stage 1 (sequential, will continue to Stage 2/3)
# ============================================================================
# Phase 2 (After Exp4 training completes):
#   GPU 3: Exp4 - Evaluation
# ============================================================================

echo "=========================================="
echo "MEGA JOB: Exp4 Retraining + All Evaluations"
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
VERIFY_SCRIPT="${BASE_DIR}/verify_exp4_fix.py"

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
    eval ${COMMAND} > ${BASE_DIR}/slurm/logs/mega_exp4_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_exp4_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
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
# GPU 3: Exp4 Training (Sequential: Stage 1 → Stage 2 → Stage 3)
# ============================================================================
EXP4_S1_CONFIG="${BASE_DIR}/configs/exp4_stage1.yaml"
EXP4_S2_CONFIG="${BASE_DIR}/configs/exp4_stage2.yaml"
EXP4_S3_CONFIG="${BASE_DIR}/configs/exp4_stage3.yaml"
EXP4_S1_OUTPUT="${BASE_DIR}/models/exp4_curriculum/stage1"
EXP4_S2_OUTPUT="${BASE_DIR}/models/exp4_curriculum/stage2"
EXP4_S3_OUTPUT="${BASE_DIR}/models/exp4_curriculum/stage3"

# ============================================================================
# Phase 1: Run Exp1, Exp2, Exp3 evaluations and Exp4 Stage 1 in parallel
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: Starting evaluations and Exp4 Stage 1"
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

# Exp4 Stage 1 training (on GPU 3)
echo ""
echo "Starting Exp4 Stage 1 training on GPU 3..."
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu3
mkdir -p ${HF_HOME}

echo "Stage 1: Training..."
python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP4_S1_CONFIG} > ${BASE_DIR}/slurm/logs/mega_exp4_s1_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_exp4_s1_${SLURM_JOB_ID}.err
EXP4_S1_EXIT=$?

if [ ${EXP4_S1_EXIT} -ne 0 ]; then
    echo "❌ Exp4 Stage 1 training failed"
    EXP4_SKIP=1
else
    # Verify Stage 1
    if [ -f "${VERIFY_SCRIPT}" ]; then
        python3 ${VERIFY_SCRIPT} ${EXP4_S1_OUTPUT} || echo "⚠️  Verification failed, but continuing..."
    fi
    echo "✓ Exp4 Stage 1 completed"
    EXP4_SKIP=0
fi

# Wait for evaluations to complete
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

echo ""
echo "Phase 1 completed:"
echo "  Exp1: Exit code ${EXIT0}"
echo "  Exp2: Exit code ${EXIT1}"
if [ ${EXP3_SKIP} -eq 0 ]; then
    echo "  Exp3: Exit code ${EXIT2}"
fi
echo "  Exp4 Stage 1: Exit code ${EXP4_S1_EXIT}"

# ============================================================================
# Phase 2: Exp4 Stage 2 Training (continues from Stage 1)
# ============================================================================
if [ ${EXP4_SKIP} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 2: Exp4 Stage 2 Training"
    echo "=========================================="
    
    echo "Stage 2: Training (continuing from Stage 1)..."
    python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP4_S2_CONFIG} > ${BASE_DIR}/slurm/logs/mega_exp4_s2_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_exp4_s2_${SLURM_JOB_ID}.err
    EXP4_S2_EXIT=$?
    
    if [ ${EXP4_S2_EXIT} -ne 0 ]; then
        echo "❌ Exp4 Stage 2 training failed"
        EXP4_SKIP=1
    else
        # Verify Stage 2
        if [ -f "${VERIFY_SCRIPT}" ]; then
            python3 ${VERIFY_SCRIPT} ${EXP4_S2_OUTPUT} || echo "⚠️  Verification failed, but continuing..."
        fi
        echo "✓ Exp4 Stage 2 completed"
    fi
else
    echo "Skipping Exp4 Stage 2 (Stage 1 failed)"
    EXP4_S2_EXIT=1
fi

# ============================================================================
# Phase 3: Exp4 Stage 3 Training (continues from Stage 2)
# ============================================================================
if [ ${EXP4_SKIP} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 3: Exp4 Stage 3 Training"
    echo "=========================================="
    
    echo "Stage 3: Training (continuing from Stage 2)..."
    python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP4_S3_CONFIG} > ${BASE_DIR}/slurm/logs/mega_exp4_s3_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_exp4_s3_${SLURM_JOB_ID}.err
    EXP4_S3_EXIT=$?
    
    if [ ${EXP4_S3_EXIT} -ne 0 ]; then
        echo "❌ Exp4 Stage 3 training failed"
        EXP4_SKIP=1
    else
        # Verify Stage 3
        if [ -f "${VERIFY_SCRIPT}" ]; then
            python3 ${VERIFY_SCRIPT} ${EXP4_S3_OUTPUT} || echo "⚠️  Verification failed, but continuing..."
        fi
        echo "✓ Exp4 Stage 3 completed"
    fi
else
    echo "Skipping Exp4 Stage 3 (previous stage failed)"
    EXP4_S3_EXIT=1
fi

# ============================================================================
# Phase 4: Exp4 Evaluation (after training completes)
# ============================================================================
if [ ${EXP4_SKIP} -eq 0 ] && [ -d "${EXP4_S3_OUTPUT}" ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 4: Exp4 Evaluation"
    echo "=========================================="
    
    EXP4_OUTPUT="${RESULTS_DIR}/exp4_evaluation.json"
    
    EXP4_EVAL_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp4.py \
        --model_path ${EXP4_S3_OUTPUT} \
        --test_data ${TEST_DATA_JSON} \
        --image_dir ${IMAGE_ROOT} \
        --output ${EXP4_OUTPUT} \
        --base_model Qwen/Qwen3-VL-8B-Instruct"
    
    run_experiment 3 "Exp4 - Evaluation" "${EXP4_EVAL_CMD}"
    EXP4_EVAL_EXIT=$?
else
    echo "Skipping Exp4 evaluation (training failed or model not found)"
    EXP4_EVAL_EXIT=1
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
echo "Exp4 Training:"
echo "  Stage 1: Exit code ${EXP4_S1_EXIT}"
echo "  Stage 2: Exit code ${EXP4_S2_EXIT}"
echo "  Stage 3: Exit code ${EXP4_S3_EXIT}"
echo "  Evaluation: Exit code ${EXP4_EVAL_EXIT}"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "End time: $(date)"
echo "=========================================="

# Cleanup
rm -f ${TEST_DATA_JSON}
rm -rf /tmp/hf_cache_${SLURM_JOB_ID}_*
rm -rf /tmp/triton_cache_${SLURM_JOB_ID}_*

# Exit with worst exit code
if [ ${EXP4_EVAL_EXIT} -ne 0 ]; then
    exit ${EXP4_EVAL_EXIT}
elif [ ${EXP4_S3_EXIT} -ne 0 ]; then
    exit ${EXP4_S3_EXIT}
elif [ ${EXP4_S2_EXIT} -ne 0 ]; then
    exit ${EXP4_S2_EXIT}
elif [ ${EXP4_S1_EXIT} -ne 0 ]; then
    exit ${EXP4_S1_EXIT}
elif [ ${EXIT0} -ne 0 ] || [ ${EXIT1} -ne 0 ] || [ ${EXIT2} -ne 0 ]; then
    exit 1
else
    exit 0
fi

