#!/bin/bash
#SBATCH --job-name=train_exp5
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/train_exp5_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/train_exp5_%j.err

# ============================================================================
# Exp5 Training - Sequential Chain-of-Thought (RESUMING)
# ============================================================================
# This trains a model specifically for Exp5 sequential CoT inference
# Uses Qwen-reordered dataset (same as Exp2)
# Will auto-resume from latest checkpoint if available
# ============================================================================

echo "=========================================="
echo "Exp5 Training - Sequential Chain-of-Thought"
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
CONFIG="${BASE_DIR}/configs/exp5_sequential_cot.yaml"

# Set environment
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p ${HF_HOME}

# Show GPU status
nvidia-smi
echo ""

echo "Configuration: ${CONFIG}"
echo ""

# Run training
python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${CONFIG}

EXIT_CODE=$?

# Cleanup
rm -rf ${HF_HOME}
rm -rf ${TRITON_CACHE_DIR}

echo ""
echo "=========================================="
echo "Exp5 Training - COMPLETED"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}







