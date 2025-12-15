#!/bin/bash
#SBATCH --job-name=instr_ft_all
#SBATCH --output=slurm/logs/train_all_instruction_finetuning_%j.out
#SBATCH --error=slurm/logs/train_all_instruction_finetuning_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Instruction Fine-tuning for ALL 5 Experiments (2 GPUs)
# Trains instruction fine-tuned models for Exp1, Exp2, Exp3, Exp4, Exp5
# 
# GPU Distribution (Sequential):
#   GPU 0: Exp1 → Exp3 → Exp4
#   GPU 1: Exp2 → Exp5
#
# Memory optimizations:
#   - batch_size: 2
#   - gradient_checkpointing: true
#   - dataloader_num_workers: 16

set -e

echo "=========================================="
echo "Instruction Fine-tuning: ALL 5 Experiments"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 2 (sequential execution)"
echo "Started: $(date)"
echo ""
echo "⚡ MEMORY OPTIMIZED:"
echo "   - batch_size: 2"
echo "   - gradient_checkpointing: true"
echo "   - dataloader_num_workers: 16"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
QLORA_DIR="${BASE_DIR}/qlora_experiments"

# Create directories
mkdir -p "${EXPERIMENTS_DIR}/slurm/logs"
mkdir -p "${EXPERIMENTS_DIR}/models"

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
# GPU 0: Sequential training of Exp1, Exp3, Exp4
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu0
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu0
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    # Exp1
    echo ""
    echo "=========================================="
    echo "GPU 0: Instruction Fine-tuning Exp1"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_instruction_finetuning.py" \
        "${EXPERIMENTS_DIR}/configs/exp1_instruction_finetuning.yaml"
    echo "✓ Exp1 instruction fine-tuning completed: $(date)"
    
    # Exp3
    echo ""
    echo "=========================================="
    echo "GPU 0: Instruction Fine-tuning Exp3"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_instruction_finetuning.py" \
        "${EXPERIMENTS_DIR}/configs/exp3_instruction_finetuning.yaml"
    echo "✓ Exp3 instruction fine-tuning completed: $(date)"
    
    # Exp4
    echo ""
    echo "=========================================="
    echo "GPU 0: Instruction Fine-tuning Exp4"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_instruction_finetuning.py" \
        "${EXPERIMENTS_DIR}/configs/exp4_instruction_finetuning.yaml"
    echo "✓ Exp4 instruction fine-tuning completed: $(date)"
) 2>&1 | sed 's/^/[GPU 0] /' &
GPU0_PID=$!

# ============================================================================
# GPU 1: Sequential training of Exp2, Exp5
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu1
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu1
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    # Exp2
    echo ""
    echo "=========================================="
    echo "GPU 1: Instruction Fine-tuning Exp2"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_instruction_finetuning.py" \
        "${EXPERIMENTS_DIR}/configs/exp2_instruction_finetuning.yaml"
    echo "✓ Exp2 instruction fine-tuning completed: $(date)"
    
    # Exp5
    echo ""
    echo "=========================================="
    echo "GPU 1: Instruction Fine-tuning Exp5"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_instruction_finetuning.py" \
        "${EXPERIMENTS_DIR}/configs/exp5_instruction_finetuning.yaml"
    echo "✓ Exp5 instruction fine-tuning completed: $(date)"
) 2>&1 | sed 's/^/[GPU 1] /' &
GPU1_PID=$!

# ============================================================================
# Wait for all jobs
# ============================================================================
echo ""
echo "=========================================="
echo "All Instruction Fine-tuning Started"
echo "=========================================="
echo "GPU 0: Exp1 → Exp3 → Exp4 (PID ${GPU0_PID})"
echo "GPU 1: Exp2 → Exp5 (PID ${GPU1_PID})"
echo ""
echo "Waiting for completion..."
echo ""

wait $GPU0_PID
GPU0_STATUS=$?

wait $GPU1_PID
GPU1_STATUS=$?

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "ALL INSTRUCTION FINE-TUNING COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""

if [ $GPU0_STATUS -eq 0 ] && [ $GPU1_STATUS -eq 0 ]; then
    echo "  ✓ All instruction fine-tuning: SUCCESS"
else
    echo "  ✗ Some instruction fine-tuning failed"
    echo "    GPU 0 status: ${GPU0_STATUS}"
    echo "    GPU 1 status: ${GPU1_STATUS}"
fi

echo ""
echo "Model outputs:"
echo "  Exp1: ${EXPERIMENTS_DIR}/models/exp1_random_instruction"
echo "  Exp2: ${EXPERIMENTS_DIR}/models/exp2_qwen_reordered_instruction"
echo "  Exp3: ${EXPERIMENTS_DIR}/models/exp3_sequential_instruction"
echo "  Exp4: ${EXPERIMENTS_DIR}/models/exp4_curriculum_instruction"
echo "  Exp5: ${EXPERIMENTS_DIR}/models/exp5_sequential_cot_instruction"
echo ""
echo "⚡ All models trained with instruction templates"
echo "=========================================="

# Exit with error if any job failed
if [ $GPU0_STATUS -ne 0 ] || [ $GPU1_STATUS -ne 0 ]; then
    exit 1
fi

exit 0


