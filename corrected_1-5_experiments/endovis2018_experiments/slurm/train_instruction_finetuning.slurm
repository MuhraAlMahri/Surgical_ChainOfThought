#!/bin/bash
#SBATCH --job-name=instr_finetune
#SBATCH --output=slurm/logs/train_instruction_finetuning_%j.out
#SBATCH --error=slurm/logs/train_instruction_finetuning_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Instruction Fine-tuning Job (2 GPUs)
# Uses Surgery R1 split dataset with instruction templates
# 
# GPU Distribution:
#   GPU 0: Instruction fine-tuning training
#   GPU 1: (available for parallel training if needed)
#
# Optimizations applied:
#   - batch_size: 4 (was 1)
#   - gradient_checkpointing: false (2x speedup)
#   - dataloader_num_workers: 16 (4x parallel I/O)

set -e

echo "=========================================="
echo "Instruction Fine-tuning Training"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 2"
echo "Started: $(date)"
echo ""
echo "⚡ OPTIMIZED CONFIGURATION:"
echo "   - batch_size: 4"
echo "   - gradient_checkpointing: false"
echo "   - dataloader_num_workers: 16"
echo "   - Dataset: Surgery R1 split"
echo "   - Instruction templates: INSTRUCTIONS_PER_CATEGORY.txt"
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
# GPU 0: Instruction Fine-tuning Training
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
    echo "GPU 0: Instruction Fine-tuning"
    echo "=========================================="
    echo "Started: $(date)"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/instruction_finetuning.yaml"
    
    # Verify config exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Run training
    python3 "${QLORA_DIR}/train_instruction_finetuning.py" \
        "${CONFIG_FILE}"
    
    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "✓ Instruction fine-tuning completed: $(date)"
        echo "  Model: ${EXPERIMENTS_DIR}/models/instruction_finetuning"
    else
        echo "✗ Instruction fine-tuning failed: $(date) (exit code: $STATUS)"
        exit $STATUS
    fi
) 2>&1 | sed 's/^/[GPU 0] /' &
TRAIN_PID=$!

# ============================================================================
# Wait for training to complete
# ============================================================================
echo ""
echo "=========================================="
echo "Training Started"
echo "=========================================="
echo "Instruction Fine-tuning (GPU 0): PID ${TRAIN_PID}"
echo ""
echo "Waiting for completion..."
echo ""

wait $TRAIN_PID
TRAIN_STATUS=$?

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "INSTRUCTION FINE-TUNING COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""

if [ $TRAIN_STATUS -eq 0 ]; then
    echo "  ✓ Instruction Fine-tuning: SUCCESS"
else
    echo "  ✗ Instruction Fine-tuning: FAILED (exit code: $TRAIN_STATUS)"
fi

echo ""
echo "Model output:"
echo "  ${EXPERIMENTS_DIR}/models/instruction_finetuning"
echo ""
echo "⚡ Trained with optimized configuration"
echo "   Dataset: Surgery R1 split"
echo "   Instruction templates: INSTRUCTIONS_PER_CATEGORY.txt"
echo "=========================================="

# Exit with error if training failed
if [ $TRAIN_STATUS -ne 0 ]; then
    exit 1
fi

exit 0











