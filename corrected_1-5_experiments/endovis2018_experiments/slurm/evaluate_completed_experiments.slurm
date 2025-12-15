#!/bin/bash
#SBATCH --job-name=eval_completed
#SBATCH --output=slurm/logs/eval_completed_%j.out
#SBATCH --error=slurm/logs/eval_completed_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Evaluate Completed Experiments Only
# This job evaluates the 6 experiments that have completed training:
#   - Exp1 (Random Baseline)
#   - Exp2 (Qwen Reordered)
#   - Exp3 Stage1 (Sequential)
#   - Exp3 Stage2 (Sequential)
#   - Exp4 Stage2 (Curriculum)
#   - Exp5 (Sequential CoT)

set -e

echo "=========================================="
echo "Evaluating Completed Experiments"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4 (parallel evaluation)"
echo "Started: $(date)"
echo ""

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
SCRIPTS_DIR="${BASE_DIR}/scripts"
RESULTS_DIR="${EXPERIMENTS_DIR}/results"
IMAGE_ROOT="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${EXPERIMENTS_DIR}/slurm/logs"

# Load modules
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Use user directory instead of /tmp to avoid disk space issues
export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}
mkdir -p ${HF_HOME} ${TRITON_CACHE_DIR}
export DS_SKIP_CUDA_CHECK=1

# CUDA environment
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH

# ============================================================================
# GPU 0: Evaluating Exp1 (Random Baseline)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu0
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu0
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 0: Evaluating Exp1 (Random Baseline)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp1_random"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp1_evaluation.json"
    
    # Verify model exists
    if [ ! -d "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/adapter_model.safetensors" ]; then
        echo "ERROR: Model not found: ${MODEL_PATH}"
        exit 1
    fi
    
    # Run evaluation
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
        exit $EXP1_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 0] /' &
EXP1_PID=$!

# ============================================================================
# GPU 1: Evaluating Exp2 (Qwen Reordered)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu1
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu1
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 1: Evaluating Exp2 (Qwen Reordered)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp2_qwen_reordered"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp2_qwen_reordered/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp2_evaluation.json"
    
    # Verify model exists
    if [ ! -d "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/adapter_model.safetensors" ]; then
        echo "ERROR: Model not found: ${MODEL_PATH}"
        exit 1
    fi
    
    # Run evaluation
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
        exit $EXP2_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 1] /' &
EXP2_PID=$!

# ============================================================================
# GPU 2: Evaluating Exp3 Stage1 (Sequential)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=2
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu2
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu2
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 2: Evaluating Exp3 Stage1 (Sequential)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp3_sequential/stage1"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp3_sequential/stage1/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp3_stage1_evaluation.json"
    
    # Verify model exists
    if [ ! -d "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/adapter_model.safetensors" ]; then
        echo "ERROR: Model not found: ${MODEL_PATH}"
        exit 1
    fi
    
    # Run evaluation (using Exp1 script since it's a single-stage model)
    python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp1.py" \
        --model_path "${MODEL_PATH}" \
        --test_data "${TEST_DATA}" \
        --image_dir "${IMAGE_ROOT}" \
        --output "${OUTPUT}" \
        --base_model "Qwen/Qwen3-VL-8B-Instruct"
    
    EXP3S1_STATUS=$?
    if [ $EXP3S1_STATUS -eq 0 ]; then
        echo "✓ Exp3 Stage1 evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp3 Stage1 evaluation failed: $(date) (exit code: $EXP3S1_STATUS)"
        exit $EXP3S1_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 2] /' &
EXP3S1_PID=$!

# ============================================================================
# GPU 3: Evaluating Exp3 Stage2 (Sequential)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=3
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu3
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu3
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 3: Evaluating Exp3 Stage2 (Sequential)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp3_sequential/stage2"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp3_sequential/stage2/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp3_stage2_evaluation.json"
    
    # Verify model exists
    if [ ! -d "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/adapter_model.safetensors" ]; then
        echo "ERROR: Model not found: ${MODEL_PATH}"
        exit 1
    fi
    
    # Run evaluation (using Exp1 script since it's a single-stage model)
    python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp1.py" \
        --model_path "${MODEL_PATH}" \
        --test_data "${TEST_DATA}" \
        --image_dir "${IMAGE_ROOT}" \
        --output "${OUTPUT}" \
        --base_model "Qwen/Qwen3-VL-8B-Instruct"
    
    EXP3S2_STATUS=$?
    if [ $EXP3S2_STATUS -eq 0 ]; then
        echo "✓ Exp3 Stage2 evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp3 Stage2 evaluation failed: $(date) (exit code: $EXP3S2_STATUS)"
        exit $EXP3S2_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 3] /' &
EXP3S2_PID=$!

# Wait for GPU 0-3 to complete
wait $EXP1_PID
EXP1_FINAL=$?
wait $EXP2_PID
EXP2_FINAL=$?
wait $EXP3S1_PID
EXP3S1_FINAL=$?
wait $EXP3S2_PID
EXP3S2_FINAL=$?

# Check if any failed
if [ $EXP1_FINAL -ne 0 ] || [ $EXP2_FINAL -ne 0 ] || [ $EXP3S1_FINAL -ne 0 ] || [ $EXP3S2_FINAL -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Some evaluations failed"
    echo "=========================================="
    [ $EXP1_FINAL -ne 0 ] && echo "✗ Exp1 failed"
    [ $EXP2_FINAL -ne 0 ] && echo "✗ Exp2 failed"
    [ $EXP3S1_FINAL -ne 0 ] && echo "✗ Exp3 Stage1 failed"
    [ $EXP3S2_FINAL -ne 0 ] && echo "✗ Exp3 Stage2 failed"
    exit 1
fi

# ============================================================================
# GPU 0: Evaluating Exp4 Stage2 (Curriculum) - After GPU 0 is free
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu0_2
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu0_2
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 0: Evaluating Exp4 Stage2 (Curriculum)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp4_curriculum/stage2"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp4_curriculum/stage2/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp4_stage2_evaluation.json"
    
    # Verify model exists
    if [ ! -d "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/adapter_model.safetensors" ]; then
        echo "ERROR: Model not found: ${MODEL_PATH}"
        exit 1
    fi
    
    # Run evaluation
    python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp4.py" \
        --model_path "${MODEL_PATH}" \
        --test_data "${TEST_DATA}" \
        --image_dir "${IMAGE_ROOT}" \
        --output "${OUTPUT}" \
        --base_model "Qwen/Qwen3-VL-8B-Instruct"
    
    EXP4S2_STATUS=$?
    if [ $EXP4S2_STATUS -eq 0 ]; then
        echo "✓ Exp4 Stage2 evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp4 Stage2 evaluation failed: $(date) (exit code: $EXP4S2_STATUS)"
        exit $EXP4S2_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 0] /' &
EXP4S2_PID=$!

# ============================================================================
# GPU 1: Evaluating Exp5 (Sequential CoT) - After GPU 1 is free
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu1_2
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu1_2
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 1: Evaluating Exp5 (Sequential CoT)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp5_sequential_cot"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp5_sequential_cot/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp5_evaluation.json"
    
    # Verify model exists
    if [ ! -d "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}/adapter_model.safetensors" ]; then
        echo "ERROR: Model not found: ${MODEL_PATH}"
        exit 1
    fi
    
    # Run evaluation
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
        exit $EXP5_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 1] /' &
EXP5_PID=$!

# Wait for remaining evaluations
wait $EXP4S2_PID
EXP4S2_FINAL=$?
wait $EXP5_PID
EXP5_FINAL=$?

# Final status check
if [ $EXP4S2_FINAL -ne 0 ] || [ $EXP5_FINAL -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Some evaluations failed"
    echo "=========================================="
    [ $EXP4S2_FINAL -ne 0 ] && echo "✗ Exp4 Stage2 failed"
    [ $EXP5_FINAL -ne 0 ] && echo "✗ Exp5 failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ ALL EVALUATIONS COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "  - exp1_evaluation.json"
echo "  - exp2_evaluation.json"
echo "  - exp3_stage1_evaluation.json"
echo "  - exp3_stage2_evaluation.json"
echo "  - exp4_stage2_evaluation.json"
echo "  - exp5_evaluation.json"
echo "=========================================="

exit 0










