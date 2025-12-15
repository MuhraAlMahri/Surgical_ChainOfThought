#!/bin/bash
#SBATCH --job-name=qlora_exp4_s2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp4_s2_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp4_s2_%j.err

echo "=========================================="
echo "EXP4 CURRICULUM - STAGE 2 (Findings Identification)"
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
CONFIG="${BASE_DIR}/configs/exp4_stage2.yaml"

# Verify Stage 1 checkpoint exists
STAGE1_CHECKPOINT="${BASE_DIR}/models/exp4_curriculum/stage1"
if [ ! -d "${STAGE1_CHECKPOINT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found at ${STAGE1_CHECKPOINT}"
    echo "Stage 2 requires Stage 1 to complete first!"
    exit 1
fi

echo "Config: ${CONFIG}"
echo "Loading from Stage 1: ${STAGE1_CHECKPOINT}"
echo ""

nvidia-smi
echo ""

echo "Starting Exp4 Stage 2 training (continuing from Stage 1)..."
echo "=========================================="

python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${CONFIG}

EXIT_CODE=$?

echo "=========================================="
echo "Stage 2 completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

rm -rf ${HF_HOME}
rm -rf ${TRITON_CACHE_DIR}

exit ${EXIT_CODE}

