#!/bin/bash
#SBATCH --job-name=endovis_eval_retrain_2gpu
#SBATCH --output=slurm/logs/mega_job_eval_retrain_2gpu_%j.out
#SBATCH --error=slurm/logs/mega_job_eval_retrain_2gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Mega Job: Evaluate completed experiments + Retrain Exp4 Stage1 (2 GPUs)
# GPU 0: Retrain Exp4 Stage1 (with fixed label masking script)
# GPU 1: Evaluate Exp1, Exp2, Exp4 Stage2, Exp5 (sequentially)

set -e

echo "=========================================="
echo "MEGA JOB: Evaluation + Retraining (2 GPUs)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 2"
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
# GPU 0: Retrain Exp4 Stage1 (with fixed label masking script)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=0
    # Use user directory instead of /tmp to avoid disk space issues
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu0
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu0
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 0: Retraining Exp4 Stage1 (Fixed Script)"
    echo "=========================================="
    echo "Started: $(date)"
    echo "Reason: All training losses were 0.0 (label masking bug fixed)"
    
    # Delete old checkpoint to force fresh training
    echo "Deleting old checkpoints and model files..."
    rm -rf "${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/checkpoint-"*
    rm -f "${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/adapter_model.safetensors"
    rm -f "${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/adapter_config.json"
    rm -f "${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/trainer_state.json"
    echo "Old files deleted. Starting fresh training..."
    
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
) 2>&1 | sed 's/^/[GPU 0] /' &
EXP4_STAGE1_PID=$!

# ============================================================================
# GPU 1: Evaluate Exp1, Exp2, Exp4 Stage2, Exp5 (Sequentially)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    # Use user directory instead of /tmp to avoid disk space issues
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu1
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu1
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    # Exp1 Evaluation
    echo ""
    echo "=========================================="
    echo "GPU 1: Evaluating Exp1 (Random Baseline)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp1_random"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa/test.jsonl"
    if [ ! -f "${TEST_DATA}" ]; then
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
    
    # Exp2 Evaluation
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
    
    # Exp4 Stage2 Evaluation
    echo ""
    echo "=========================================="
    echo "GPU 1: Evaluating Exp4 Stage2 (Curriculum)"
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
    
    # Exp5 Evaluation
    echo ""
    echo "=========================================="
    echo "GPU 1: Evaluating Exp5 (Sequential CoT)"
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
    
) 2>&1 | sed 's/^/[GPU 1] /' &
EVAL_PID=$!

# ============================================================================
# Wait for all parallel jobs
# ============================================================================
echo ""
echo "=========================================="
echo "Waiting for all parallel jobs to complete..."
echo "=========================================="

wait $EXP4_STAGE1_PID
EXP4_STAGE1_FINAL=$?

wait $EVAL_PID
EVAL_FINAL=$?

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
if [ $EXP4_STAGE1_FINAL -eq 0 ]; then
    echo "  ✓ Exp4 Stage1 retraining: SUCCESS"
else
    echo "  ✗ Exp4 Stage1 retraining: FAILED"
fi

if [ $EVAL_FINAL -eq 0 ]; then
    echo "  ✓ All evaluations: SUCCESS"
else
    echo "  ✗ Some evaluations: FAILED"
fi

echo ""
echo "Evaluation results saved to: ${RESULTS_DIR}/"
echo "  - exp1_evaluation.json"
echo "  - exp2_evaluation.json"
echo "  - exp4_stage2_evaluation.json"
echo "  - exp5_evaluation.json"
echo ""
echo "Exp4 Stage1 model: ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1"
echo "=========================================="

# Exit with error if any job failed
if [ $EXP4_STAGE1_FINAL -ne 0 ] || [ $EVAL_FINAL -ne 0 ]; then
    exit 1
fi

