#!/bin/bash
#SBATCH --job-name=eval_instr_ft
#SBATCH --output=slurm/logs/mega_job_eval_instr_ft_%j.out
#SBATCH --error=slurm/logs/mega_job_eval_instr_ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Mega Job: Evaluating ALL Instruction Fine-tuned Models
# This job evaluates all 5 instruction fine-tuned experiments
# 
# GPU Distribution:
#   GPU 0: Exp1 (Random Baseline - Instruction Fine-tuned)
#   GPU 1: Exp2 (Qwen Reordered - Instruction Fine-tuned)
#   GPU 2: Exp3 (Sequential - Instruction Fine-tuned)
#   GPU 3: Exp4 (Curriculum - Instruction Fine-tuned) and Exp5 (Sequential CoT - Instruction Fine-tuned)
#
# IMPORTANT: All evaluations use the CORRECTED test dataset

set -e

echo "=========================================="
echo "MEGA JOB: Evaluating ALL Instruction Fine-tuned Models"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4 (parallel execution)"
echo "Started: $(date)"
echo ""
echo "⚠️  Evaluating INSTRUCTION FINE-TUNED models"
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
# GPU 0: Evaluating Exp1 (Random Baseline - Instruction Fine-tuned)
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
    echo "GPU 0: Evaluating Exp1 (Instruction Fine-tuned)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp1_random_instruction"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp1_instruction_evaluation.json"
    
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
        echo "✓ Exp1 instruction evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp1 instruction evaluation failed: $(date) (exit code: $EXP1_STATUS)"
        exit $EXP1_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 0] /' &
EXP1_PID=$!

# ============================================================================
# GPU 1: Evaluating Exp2 (Qwen Reordered - Instruction Fine-tuned)
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
    echo "GPU 1: Evaluating Exp2 (Instruction Fine-tuned)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp2_qwen_reordered_instruction"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp2_qwen_reordered/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp2_instruction_evaluation.json"
    
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
        echo "✓ Exp2 instruction evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp2 instruction evaluation failed: $(date) (exit code: $EXP2_STATUS)"
        exit $EXP2_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 1] /' &
EXP2_PID=$!

# ============================================================================
# GPU 2: Evaluating Exp3 (Sequential - Instruction Fine-tuned)
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
    echo "GPU 2: Evaluating Exp3 (Instruction Fine-tuned)"
    echo "=========================================="
    echo "Started: $(date)"
    
    # For instruction fine-tuned Exp3, we use a single model (not stage-specific)
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp3_sequential_instruction"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp3_sequential/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp3_instruction_evaluation.json"
    
    # Verify model exists
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "ERROR: Model not found: ${MODEL_PATH}"
        exit 1
    fi
    
    # Use the same model for all stages (instruction fine-tuned is a single model)
    python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp3.py" \
        --model_stage1 "${MODEL_PATH}" \
        --model_stage2 "${MODEL_PATH}" \
        --model_stage3 "${MODEL_PATH}" \
        --test_data "${TEST_DATA}" \
        --image_dir "${IMAGE_ROOT}" \
        --output "${OUTPUT}" \
        --base_model "Qwen/Qwen3-VL-8B-Instruct"
    
    EXP3_STATUS=$?
    if [ $EXP3_STATUS -eq 0 ]; then
        echo "✓ Exp3 instruction evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp3 instruction evaluation failed: $(date) (exit code: $EXP3_STATUS)"
        exit $EXP3_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 2] /' &
EXP3_PID=$!

# ============================================================================
# GPU 3: Evaluating Exp4 (Curriculum - Instruction Fine-tuned) then Exp5 (Sequential CoT - Instruction Fine-tuned)
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
    echo "GPU 3: Evaluating Exp4 (Instruction Fine-tuned)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp4_curriculum_instruction"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp4_curriculum/stage2/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp4_instruction_evaluation.json"
    
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
        echo "✓ Exp4 instruction evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp4 instruction evaluation failed: $(date) (exit code: $EXP4_STATUS)"
        exit $EXP4_STATUS
    fi
    
    # Now evaluate Exp5
    echo ""
    echo "=========================================="
    echo "GPU 3: Evaluating Exp5 (Instruction Fine-tuned)"
    echo "=========================================="
    echo "Started: $(date)"
    
    MODEL_PATH="${EXPERIMENTS_DIR}/models/exp5_sequential_cot_instruction"
    TEST_DATA="${BASE_DIR}/datasets/endovis2018_vqa_reordered/exp5_sequential_cot/test.jsonl"
    OUTPUT="${RESULTS_DIR}/exp5_instruction_evaluation.json"
    
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
        echo "✓ Exp5 instruction evaluation completed: $(date)"
        echo "  Results: ${OUTPUT}"
    else
        echo "✗ Exp5 instruction evaluation failed: $(date) (exit code: $EXP5_STATUS)"
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
echo "Exp3 (GPU 2): PID ${EXP3_PID}"
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
echo "ALL INSTRUCTION FINE-TUNED EVALUATIONS COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Status:"
if [ $EXP1_STATUS -eq 0 ]; then
    echo "  ✓ Exp1 (Instruction Fine-tuned): SUCCESS"
else
    echo "  ✗ Exp1 (Instruction Fine-tuned): FAILED (exit code: $EXP1_STATUS)"
fi

if [ $EXP2_STATUS -eq 0 ]; then
    echo "  ✓ Exp2 (Instruction Fine-tuned): SUCCESS"
else
    echo "  ✗ Exp2 (Instruction Fine-tuned): FAILED (exit code: $EXP2_STATUS)"
fi

if [ $EXP3_STATUS -eq 0 ]; then
    echo "  ✓ Exp3 (Instruction Fine-tuned): SUCCESS"
else
    echo "  ✗ Exp3 (Instruction Fine-tuned): FAILED (exit code: $EXP3_STATUS)"
fi

if [ $EXP4_STATUS -eq 0 ]; then
    echo "  ✓ Exp4 (Instruction Fine-tuned): SUCCESS"
else
    echo "  ✗ Exp4 (Instruction Fine-tuned): FAILED (exit code: $EXP4_STATUS)"
fi

if [ $EXP5_STATUS -eq 0 ]; then
    echo "  ✓ Exp5 (Instruction Fine-tuned): SUCCESS"
else
    echo "  ✗ Exp5 (Instruction Fine-tuned): FAILED (exit code: $EXP5_STATUS)"
fi

echo ""
echo "Results saved to:"
echo "  Exp1: ${RESULTS_DIR}/exp1_instruction_evaluation.json"
echo "  Exp2: ${RESULTS_DIR}/exp2_instruction_evaluation.json"
echo "  Exp3: ${RESULTS_DIR}/exp3_instruction_evaluation.json"
echo "  Exp4: ${RESULTS_DIR}/exp4_instruction_evaluation.json"
echo "  Exp5: ${RESULTS_DIR}/exp5_instruction_evaluation.json"
echo ""
echo "⚠️  All evaluations used INSTRUCTION FINE-TUNED models"
echo "   (trained with instruction templates)"
echo "=========================================="

# Exit with error if any job failed
if [ $EXP1_STATUS -ne 0 ] || [ $EXP2_STATUS -ne 0 ] || [ $EXP3_STATUS -ne 0 ] || [ $EXP4_STATUS -ne 0 ] || [ $EXP5_STATUS -ne 0 ]; then
    exit 1
fi

exit 0











