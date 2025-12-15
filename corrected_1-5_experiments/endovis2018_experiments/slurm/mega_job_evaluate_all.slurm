#!/bin/bash
#SBATCH --job-name=endovis_eval_all
#SBATCH --output=slurm/logs/mega_job_eval_all_%j.out
#SBATCH --error=slurm/logs/mega_job_eval_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Mega Job: Evaluating ALL EndoVis2018 Experiments
# This job evaluates all 5 experiments after training with the corrected dataset
# 
# GPU Distribution:
#   GPU 0: Exp1 (Random Baseline)
#   GPU 1: Exp2 (Qwen Reordered)
#   GPU 2: Exp3 (Sequential - Separate Models per Stage)
#   GPU 3: Exp4 (Curriculum - Stage 2 model) and Exp5 (Sequential CoT)
#
# IMPORTANT: All evaluations use the CORRECTED test dataset

set -e

echo "=========================================="
echo "MEGA JOB: Evaluating ALL EndoVis2018 Experiments"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4 (parallel execution)"
echo "Started: $(date)"
echo ""
echo "⚠️  Evaluating with CORRECTED dataset"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
SCRIPTS_DIR="${BASE_DIR}/scripts"
RESULTS_DIR="${EXPERIMENTS_DIR}/results"
IMAGE_ROOT="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"

# Create directories
mkdir -p "${EXPERIMENTS_DIR}/slurm/logs"
mkdir -p "${RESULTS_DIR}"

# Load modules
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
# Use user directory instead of /tmp to avoid disk space issues
export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}
mkdir -p ${HF_HOME} ${TRITON_CACHE_DIR}
# Disable DeepSpeed CUDA checks
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
    if [ ! -d "${MODEL_PATH}" ]; then
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
    if [ ! -d "${MODEL_PATH}" ]; then
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
# GPU 2: Evaluating Exp3 (Sequential - Separate Models per Stage)
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
    echo "GPU 2: Evaluating Exp3 (Sequential)"
    echo "=========================================="
    echo "Started: $(date)"
    
    STAGE1_MODEL="${EXPERIMENTS_DIR}/models/exp3_sequential/stage1"
    STAGE2_MODEL="${EXPERIMENTS_DIR}/models/exp3_sequential/stage2"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp3_sequential/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp3_evaluation.json"
    
    # Verify models exist
    if [ ! -d "${STAGE1_MODEL}" ]; then
        echo "ERROR: Stage 1 model not found: ${STAGE1_MODEL}"
        exit 1
    fi
    if [ ! -d "${STAGE2_MODEL}" ]; then
        echo "ERROR: Stage 2 model not found: ${STAGE2_MODEL}"
        exit 1
    fi
    
    # Use Stage 2 model as placeholder for Stage 3 (won't be used since no Stage 3 samples)
    python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp3.py" \
        --model_stage1 "${STAGE1_MODEL}" \
        --model_stage2 "${STAGE2_MODEL}" \
        --model_stage3 "${STAGE2_MODEL}" \
        --test_data "${TEST_DATA}" \
        --image_dir "${IMAGE_ROOT}" \
        --output "${OUTPUT}" \
        --base_model "Qwen/Qwen3-VL-8B-Instruct"
    
    EXP3_STATUS=$?
    if [ $EXP3_STATUS -eq 0 ]; then
        echo "✓ Exp3 evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp3 evaluation failed: $(date) (exit code: $EXP3_STATUS)"
        exit $EXP3_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 2] /' &
EXP3_PID=$!

# ============================================================================
# GPU 3: Evaluating Exp4 (Curriculum) then Exp5 (Sequential CoT)
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
    echo "GPU 3: Evaluating Exp4 (Curriculum)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp4_curriculum/stage2"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp4_curriculum/stage2/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp4_stage2_evaluation.json"
    
    # Verify model exists
    if [ ! -d "${MODEL_PATH}" ]; then
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
    
    EXP4_STATUS=$?
    if [ $EXP4_STATUS -eq 0 ]; then
        echo "✓ Exp4 evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp4 evaluation failed: $(date) (exit code: $EXP4_STATUS)"
        exit $EXP4_STATUS
    fi
    
    # Now evaluate Exp5
    echo ""
    echo "=========================================="
    echo "GPU 3: Evaluating Exp5 (Sequential CoT)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp5_sequential_cot"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp5_sequential_cot/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp5_evaluation.json"
    
    # Verify model exists
    if [ ! -d "${MODEL_PATH}" ]; then
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
) 2>&1 | sed 's/^/[GPU 3] /' &
EXP4_PID=$!

# ============================================================================
# Wait for all jobs to complete
# ============================================================================
echo ""
echo "=========================================="
echo "All Evaluations Started"
echo "=========================================="
echo "Exp1 (GPU 0): PID ${EXP1_PID}"
echo "Exp2 (GPU 1): PID ${EXP2_PID}"
echo "Exp3 (GPU 2): PID ${EXP3_PID} (Stage 1 & Stage 2 models)"
echo "Exp4 (GPU 3): PID ${EXP4_PID} → then Exp5"
echo ""
echo "Waiting for all jobs to complete..."
echo ""

# Wait for all background jobs
wait $EXP1_PID
EXP1_STATUS=$?

wait $EXP2_PID
EXP2_STATUS=$?

wait $EXP3_PID
EXP3_STATUS=$?

wait $EXP4_PID
GPU3_STATUS=$?
# Extract individual statuses from the combined GPU 3 job
if [ $GPU3_STATUS -eq 0 ]; then
    EXP4_STATUS=0
    EXP5_STATUS=0
else
    EXP4_STATUS=$GPU3_STATUS
    EXP5_STATUS=$GPU3_STATUS
fi

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Status:"
if [ $EXP1_STATUS -eq 0 ]; then
    echo "  ✓ Exp1 (Random Baseline): SUCCESS"
else
    echo "  ✗ Exp1 (Random Baseline): FAILED (exit code: $EXP1_STATUS)"
fi

if [ $EXP2_STATUS -eq 0 ]; then
    echo "  ✓ Exp2 (Qwen Reordered): SUCCESS"
else
    echo "  ✗ Exp2 (Qwen Reordered): FAILED (exit code: $EXP2_STATUS)"
fi

if [ $EXP3_STATUS -eq 0 ]; then
    echo "  ✓ Exp3 (Sequential): SUCCESS"
else
    echo "  ✗ Exp3 (Sequential): FAILED (exit code: $EXP3_STATUS)"
fi

if [ $EXP4_STATUS -eq 0 ]; then
    echo "  ✓ Exp4 (Curriculum): SUCCESS"
else
    echo "  ✗ Exp4 (Curriculum): FAILED (exit code: $EXP4_STATUS)"
fi

if [ $EXP5_STATUS -eq 0 ]; then
    echo "  ✓ Exp5 (Sequential CoT): SUCCESS"
else
    echo "  ✗ Exp5 (Sequential CoT): FAILED (exit code: $EXP5_STATUS)"
fi

echo ""
echo "Results saved to:"
echo "  Exp1: ${RESULTS_DIR}/exp1_evaluation.json"
echo "  Exp2: ${RESULTS_DIR}/exp2_evaluation.json"
echo "  Exp3: ${RESULTS_DIR}/exp3_evaluation.json"
echo "  Exp4: ${RESULTS_DIR}/exp4_stage2_evaluation.json"
echo "  Exp5: ${RESULTS_DIR}/exp5_evaluation.json"
echo ""
echo "⚠️  All evaluations used CORRECTED dataset"
echo "   (fixed mask paths, correct answers)"
echo "=========================================="

# Exit with error if any job failed
if [ $EXP1_STATUS -ne 0 ] || [ $EXP2_STATUS -ne 0 ] || [ $EXP3_STATUS -ne 0 ] || [ $EXP4_STATUS -ne 0 ] || [ $EXP5_STATUS -ne 0 ]; then
    exit 1
fi

exit 0

