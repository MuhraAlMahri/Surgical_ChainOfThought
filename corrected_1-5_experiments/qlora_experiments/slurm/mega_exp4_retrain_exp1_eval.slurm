#!/bin/bash
#SBATCH --job-name=mega_exp4_exp1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_exp4_exp1_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_exp4_exp1_%j.err

# ============================================================================
# MEGA JOB: Exp4 Retraining + Exp1 Evaluation
# ============================================================================
# Phase 1 (Parallel):
#   GPU 0: Exp1 - Evaluation (rerun with new metrics)
#   GPU 1: Exp4 - Training Stage 1 (starts curriculum learning)
# ============================================================================
# Phase 2 (Sequential):
#   GPU 1: Exp4 - Training Stage 2 (continues from Stage 1)
#   GPU 1: Exp4 - Training Stage 3 (continues from Stage 2)
# ============================================================================
# Phase 3 (After Exp4 training completes):
#   GPU 1: Exp4 - Evaluation
# ============================================================================

echo "=========================================="
echo "MEGA JOB: Exp4 Retraining + Exp1 Evaluation"
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
    eval ${COMMAND} > ${BASE_DIR}/slurm/logs/mega_exp4_exp1_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_exp4_exp1_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
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
# GPU 1: Exp4 Training (Sequential: Stage 1 → Stage 2 → Stage 3)
# ============================================================================
EXP4_S1_CONFIG="${BASE_DIR}/configs/exp4_stage1.yaml"
EXP4_S2_CONFIG="${BASE_DIR}/configs/exp4_stage2.yaml"
EXP4_S3_CONFIG="${BASE_DIR}/configs/exp4_stage3.yaml"
EXP4_S1_OUTPUT="${BASE_DIR}/models/exp4_curriculum/stage1"
EXP4_S2_OUTPUT="${BASE_DIR}/models/exp4_curriculum/stage2"
EXP4_S3_OUTPUT="${BASE_DIR}/models/exp4_curriculum/stage3"

# ============================================================================
# Phase 1: Run Exp1 evaluation and Exp4 Stage 1 training in parallel
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: Starting Exp1 evaluation and Exp4 Stage 1 training"
echo "=========================================="

# Start Exp1 evaluation in parallel
run_experiment 0 "Exp1 - Evaluation" "${EXP1_CMD}" &
PID0=$!

# Exp4 Stage 1 training (on GPU 1)
echo ""
echo "Starting Exp4 Stage 1 training on GPU 1..."
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu1
mkdir -p ${HF_HOME}

echo "Stage 1: Training..."
python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP4_S1_CONFIG} > ${BASE_DIR}/slurm/logs/mega_exp4_exp1_s1_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_exp4_exp1_s1_${SLURM_JOB_ID}.err
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

# Wait for Exp1 evaluation to complete
echo ""
echo "Waiting for Exp1 evaluation to complete..."
wait ${PID0}
EXIT0=$?

echo ""
echo "Phase 1 completed:"
echo "  Exp1: Exit code ${EXIT0}"
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
    python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP4_S2_CONFIG} > ${BASE_DIR}/slurm/logs/mega_exp4_exp1_s2_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_exp4_exp1_s2_${SLURM_JOB_ID}.err
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
    python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP4_S3_CONFIG} > ${BASE_DIR}/slurm/logs/mega_exp4_exp1_s3_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_exp4_exp1_s3_${SLURM_JOB_ID}.err
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
    
    run_experiment 1 "Exp4 - Evaluation" "${EXP4_EVAL_CMD}"
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
echo "Exp1 Evaluation: Exit code ${EXIT0}"
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
elif [ ${EXIT0} -ne 0 ]; then
    exit ${EXIT0}
else
    exit 0
fi


