#!/bin/bash
#SBATCH --job-name=qlora_exp4_s3_final
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/batch3_final_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/batch3_final_%j.err

# ============================================================================
# BATCH 3: Final Stage - Exp4-S3 (1 GPU)
# ============================================================================
# Exp4-S3 - Curriculum Stage 3 (~30min, continues from Exp4-S2)
# ============================================================================

echo "=========================================="
echo "BATCH 3 - FINAL STAGE"
echo "Exp4 - Curriculum Stage 3 (Clinical Context)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Setup
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p ${HF_HOME}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"
CONFIG="${BASE_DIR}/configs/exp4_stage3.yaml"

# Verify Stage 2 checkpoint exists
STAGE2_CHECKPOINT="${BASE_DIR}/models/exp4_curriculum/stage2"
if [ ! -d "${STAGE2_CHECKPOINT}" ]; then
    echo "ERROR: Exp4-Stage2 checkpoint not found at ${STAGE2_CHECKPOINT}"
    echo "Batch 3 requires Batch 2 Task 1 (Exp4-S2) to complete first!"
    exit 1
fi

echo "Config: ${CONFIG}"
echo "Loading from Stage 2: ${STAGE2_CHECKPOINT}"
echo ""

nvidia-smi
echo ""

echo "Starting training..."
echo "=========================================="

python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${CONFIG}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

rm -rf ${HF_HOME}
rm -rf ${TRITON_CACHE_DIR}

exit ${EXIT_CODE}

