#!/bin/bash
#SBATCH --job-name=qlora_exp4_s3
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp4_s3_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp4_s3_%j.err

echo "=========================================="
echo "EXP4 CURRICULUM - STAGE 3 (Clinical Context)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

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
    echo "ERROR: Stage 2 checkpoint not found at ${STAGE2_CHECKPOINT}"
    echo "Stage 3 requires Stage 2 to complete first!"
    exit 1
fi

echo "Config: ${CONFIG}"
echo "Loading from Stage 2: ${STAGE2_CHECKPOINT}"
echo ""

nvidia-smi
echo ""

echo "Starting Exp4 Stage 3 training (continuing from Stage 2)..."
echo "=========================================="

python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${CONFIG}

EXIT_CODE=$?

echo "=========================================="
echo "Stage 3 completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

rm -rf ${HF_HOME}
rm -rf ${TRITON_CACHE_DIR}

exit ${EXIT_CODE}

