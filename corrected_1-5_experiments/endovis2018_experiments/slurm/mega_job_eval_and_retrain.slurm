#!/bin/bash
#SBATCH --job-name=endovis_eval_retrain
#SBATCH --output=slurm/logs/mega_job_eval_retrain_%j.out
#SBATCH --error=slurm/logs/mega_job_eval_retrain_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Mega Job: Evaluate completed experiments + Retrain Exp4 Stage1
# Runs in parallel:
#   GPU 0: Evaluate Exp1 (Random Baseline)
#   GPU 1: Evaluate Exp2 (Qwen Reordered)
#   GPU 2: Evaluate Exp4 Stage2 + Exp5 (Sequential)
#   GPU 3: Retrain Exp4 Stage1 (all losses were 0.0)
# After Exp3 training completes, it will be evaluated separately

set -e

echo "=========================================="
echo "MEGA JOB: Evaluation + Retraining"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4 (parallel execution)"
echo "Started: $(date)"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
QLORA_DIR="${BASE_DIR}/qlora_experiments"
SCRIPTS_DIR="${BASE_DIR}/scripts"
RESULTS_DIR="${EXPERIMENTS_DIR}/results"

# Create directories
mkdir -p "${EXPERIMENTS_DIR}/slurm/logs"
mkdir -p "${RESULTS_DIR}"

# Load modules
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
export DS_SKIP_CUDA_CHECK=1

cd "${EXPERIMENTS_DIR}"

# Test data paths
TEST_DATA_BASE="${BASE_DIR}/datasets/endovis2018_vqa_reordered"
IMAGE_ROOT="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"

# ============================================================================
# GPU 0: Evaluate Exp1 (Random Baseline)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu0
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 0: Evaluating Exp1 (Random Baseline)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp1_random"
    # Exp1 uses original (non-reordered) test data
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa/test.jsonl"
    if [ ! -f "${TEST_DATA}" ]; then
        # Fallback to reordered test data if original doesn't exist
        TEST_DATA="${TEST_DATA_BASE}/exp2_qwen_reordered/test.jsonl"
    fi
    OUTPUT="${RESULTS_DIR}/exp1_evaluation.json"
    
    python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp1.py" \
        --model_path "${MODEL_PATH}" \
        --test_data "${TEST_DATA}" \
        --image_dir "${IMAGE_ROOT}" \
        --output "${OUTPUT}" \
        --base_model "Qwen/Qwen3-VL-8B-Instruct"
    
    EXP1_STATUS=$?
    if [ $EXP1_STATUS -eq 0 ]; then
        echo "✓ Exp1 evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp1 evaluation failed: $(date) (exit code: $EXP1_STATUS)"
    fi
) 2>&1 | sed 's/^/[GPU 0] /' &
EXP1_PID=$!

# ============================================================================
# GPU 1: Evaluate Exp2 (Qwen Reordered)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu1
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 1: Evaluating Exp2 (Qwen Reordered)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp2_qwen_reordered"
    TEST_DATA="${TEST_DATA_BASE}/exp2_qwen_reordered/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp2_evaluation.json"
    
    python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp2.py" \
        --model_path "${MODEL_PATH}" \
        --test_data "${TEST_DATA}" \
        --image_dir "${IMAGE_ROOT}" \
        --output "${OUTPUT}" \
        --base_model "Qwen/Qwen3-VL-8B-Instruct"
    
    EXP2_STATUS=$?
    if [ $EXP2_STATUS -eq 0 ]; then
        echo "✓ Exp2 evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp2 evaluation failed: $(date) (exit code: $EXP2_STATUS)"
    fi
) 2>&1 | sed 's/^/[GPU 1] /' &
EXP2_PID=$!

# ============================================================================
# GPU 2: Evaluate Exp4 Stage2 and Exp5 (Sequential)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=2
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu2
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 2: Evaluating Exp4 Stage2 (Curriculum)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp4_curriculum/stage2"
    TEST_DATA="${TEST_DATA_BASE}/exp4_curriculum/stage2/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp4_stage2_evaluation.json"
    
    python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp4.py" \
        --model_path "${MODEL_PATH}" \
        --test_data "${TEST_DATA}" \
        --image_dir "${IMAGE_ROOT}" \
        --output "${OUTPUT}" \
        --base_model "Qwen/Qwen3-VL-8B-Instruct"
    
    EXP4_STAGE2_STATUS=$?
    if [ $EXP4_STAGE2_STATUS -eq 0 ]; then
        echo "✓ Exp4 Stage2 evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp4 Stage2 evaluation failed: $(date) (exit code: $EXP4_STAGE2_STATUS)"
    fi
    
    echo ""
    echo "=========================================="
    echo "GPU 2: Evaluating Exp5 (Sequential CoT)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp5_sequential_cot"
    TEST_DATA="${TEST_DATA_BASE}/exp5_sequential_cot/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp5_evaluation.json"
    
    python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp5.py" \
        --model_path "${MODEL_PATH}" \
        --test_data "${TEST_DATA}" \
        --image_dir "${IMAGE_ROOT}" \
        --output "${OUTPUT}" \
        --base_model "Qwen/Qwen3-VL-8B-Instruct"
    
    EXP5_STATUS=$?
    if [ $EXP5_STATUS -eq 0 ]; then
        echo "✓ Exp5 evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp5 evaluation failed: $(date) (exit code: $EXP5_STATUS)"
    fi
) 2>&1 | sed 's/^/[GPU 2] /' &
EXP4_STAGE2_PID=$!

# ============================================================================
# GPU 3: Retrain Exp4 Stage1 (all losses were 0.0 - needs retraining)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=3
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu3
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 3: Retraining Exp4 Stage1 (Fixed Script)"
    echo "=========================================="
    echo "Started: $(date)"
    echo "Reason: All training losses were 0.0 (label masking bug)"
    
    # Delete old checkpoint to force fresh training
    rm -rf "${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/checkpoint-*"
    rm -f "${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/adapter_model.safetensors"
    rm -f "${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/adapter_config.json"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp4_stage1.yaml"
    
    # Verify config exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Run training with FIXED script
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    EXP4_STAGE1_STATUS=$?
    if [ $EXP4_STAGE1_STATUS -eq 0 ]; then
        echo "✓ Exp4 Stage1 retraining completed: $(date)"
        echo "  Model: ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1"
    else
        echo "✗ Exp4 Stage1 retraining failed: $(date) (exit code: $EXP4_STAGE1_STATUS)"
        exit $EXP4_STAGE1_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 3] /' &
EXP4_STAGE1_PID=$!

# ============================================================================
# Wait for all parallel jobs
# ============================================================================
echo ""
echo "=========================================="
echo "Waiting for all parallel jobs to complete..."
echo "=========================================="

wait $EXP1_PID
EXP1_FINAL=$?

wait $EXP2_PID
EXP2_FINAL=$?

wait $EXP4_STAGE2_PID
EXP4_STAGE2_FINAL=$?

wait $EXP4_STAGE1_PID
EXP4_STAGE1_FINAL=$?

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "MEGA JOB COMPLETE!"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Results:"
if [ $EXP1_FINAL -eq 0 ]; then
    echo "  ✓ Exp1 evaluation: SUCCESS"
else
    echo "  ✗ Exp1 evaluation: FAILED"
fi

if [ $EXP2_FINAL -eq 0 ]; then
    echo "  ✓ Exp2 evaluation: SUCCESS"
else
    echo "  ✗ Exp2 evaluation: FAILED"
fi

if [ $EXP4_STAGE2_FINAL -eq 0 ]; then
    echo "  ✓ Exp4 Stage2 evaluation: SUCCESS"
else
    echo "  ✗ Exp4 Stage2 evaluation: FAILED"
fi

if [ $EXP5_STATUS -eq 0 ]; then
    echo "  ✓ Exp5 evaluation: SUCCESS"
else
    echo "  ✗ Exp5 evaluation: FAILED"
fi

if [ $EXP4_STAGE1_FINAL -eq 0 ]; then
    echo "  ✓ Exp4 Stage1 retraining: SUCCESS"
else
    echo "  ✗ Exp4 Stage1 retraining: FAILED"
fi

echo ""
echo "Evaluation results saved to: ${RESULTS_DIR}/"
echo "  - exp1_evaluation.json"
echo "  - exp2_evaluation.json"
echo "  - exp4_stage2_evaluation.json"
echo "  - exp5_evaluation.json"
echo ""
echo "Note: Exp3 evaluation will run separately after Exp3 training completes"
echo "=========================================="

# Exit with error if any job failed
if [ $EXP1_FINAL -ne 0 ] || [ $EXP2_FINAL -ne 0 ] || [ $EXP4_STAGE2_FINAL -ne 0 ] || [ $EXP5_STATUS -ne 0 ] || [ $EXP4_STAGE1_FINAL -ne 0 ]; then
    exit 1
fi

