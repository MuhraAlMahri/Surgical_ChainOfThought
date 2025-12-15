#!/bin/bash
#SBATCH --job-name=exp1_instr_ft
#SBATCH --output=slurm/logs/train_exp1_instruction_finetuning_%j.out
#SBATCH --error=slurm/logs/train_exp1_instruction_finetuning_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Exp1 Instruction Fine-tuning (Baseline)
# Uses R1 split (1560 train frames, 228 val frames) with instruction templates
# Matches Kvasir-VQA Exp1 configuration exactly

set -e

echo "=========================================="
echo "Exp1 Instruction Fine-tuning Training"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: 1"
echo "Started: $(date)"
echo ""
echo "⚡ CONFIGURATION (Matches Kvasir-VQA Exp1):"
echo "   - Dataset: R1 split (1560 train frames, 228 val frames)"
echo "   - train_bs: 1"
echo "   - grad_accum: 16 (effective batch = 16)"
echo "   - lr: 5.0e-5"
echo "   - epochs: 5"
echo "   - max_seq_len: 3072"
echo "   - LoRA: r=4, alpha=8"
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
# Exp1 Instruction Fine-tuning Training
# ============================================================================
echo ""
echo "=========================================="
echo "Exp1 Instruction Fine-tuning"
echo "=========================================="
echo "Started: $(date)"
echo ""

CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp1_instruction_finetuning.yaml"

# Verify config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Verify dataset files exist
TRAIN_DATA="${BASE_DIR}/datasets/endovis18_surgery_r1_split/train.jsonl"
VAL_DATA="${BASE_DIR}/datasets/endovis18_surgery_r1_split/val.jsonl"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: Train data not found: ${TRAIN_DATA}"
    exit 1
fi

if [ ! -f "${VAL_DATA}" ]; then
    echo "ERROR: Val data not found: ${VAL_DATA}"
    exit 1
fi

echo "✓ Config file: ${CONFIG_FILE}"
echo "✓ Train data: ${TRAIN_DATA}"
echo "✓ Val data: ${VAL_DATA}"
echo ""

# Run training
python3 "${QLORA_DIR}/train_instruction_finetuning.py" \
    "${CONFIG_FILE}"

STATUS=$?

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "EXP1 INSTRUCTION FINE-TUNING COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""

if [ $STATUS -eq 0 ]; then
    echo "  ✓ Exp1 Instruction Fine-tuning: SUCCESS"
    echo ""
    echo "Model output:"
    echo "  ${EXPERIMENTS_DIR}/models/exp1_random_instruction_r1"
    echo ""
    echo "Configuration:"
    echo "  - Dataset: R1 split (1560 train frames, 228 val frames)"
    echo "  - Matches Kvasir-VQA Exp1 config exactly"
    echo "  - Instruction templates: INSTRUCTIONS_PER_CATEGORY.txt"
else
    echo "  ✗ Exp1 Instruction Fine-tuning: FAILED (exit code: $STATUS)"
    exit $STATUS
fi

echo "=========================================="

exit 0








