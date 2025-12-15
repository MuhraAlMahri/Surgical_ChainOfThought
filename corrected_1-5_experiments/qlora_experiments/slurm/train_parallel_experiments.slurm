#!/bin/bash
#SBATCH --job-name=qlora_parallel_%a
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --array=0-4  # 5 parallel experiments
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/parallel_%a_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/parallel_%a_%j.err

# ============================================================================
# PARALLEL QLoRA TRAINING - 5 Independent Experiments
# ============================================================================
# Array task mapping:
#   0 = Exp1 (Random Baseline)
#   1 = Exp2 (Qwen Reordered)
#   2 = Exp3-Stage1 (CXRTrek Sequential S1)
#   3 = Exp3-Stage2 (CXRTrek Sequential S2)
#   4 = Exp3-Stage3 (CXRTrek Sequential S3)
# ============================================================================

echo "=========================================="
echo "QLoRA PARALLEL TRAINING - Task ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Setup
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p ${HF_HOME}

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"

# Map array task ID to experiment config
case ${SLURM_ARRAY_TASK_ID} in
    0)
        EXP_NAME="Exp1 - Random Baseline"
        CONFIG="${BASE_DIR}/configs/exp1_random.yaml"
        ;;
    1)
        EXP_NAME="Exp2 - Qwen Reordered"
        CONFIG="${BASE_DIR}/configs/exp2_qwen_reordered.yaml"
        ;;
    2)
        EXP_NAME="Exp3 - CXRTrek Stage 1"
        CONFIG="${BASE_DIR}/configs/exp3_stage1.yaml"
        ;;
    3)
        EXP_NAME="Exp3 - CXRTrek Stage 2"
        CONFIG="${BASE_DIR}/configs/exp3_stage2.yaml"
        ;;
    4)
        EXP_NAME="Exp3 - CXRTrek Stage 3"
        CONFIG="${BASE_DIR}/configs/exp3_stage3.yaml"
        ;;
    *)
        echo "ERROR: Invalid task ID ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo ""
echo "Experiment: ${EXP_NAME}"
echo "Config: ${CONFIG}"
echo ""

# Show GPU info
nvidia-smi
echo ""

# Run training
echo "Starting training..."
echo "=========================================="

python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${CONFIG}

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

# Cleanup cache
rm -rf ${HF_HOME}
rm -rf ${TRITON_CACHE_DIR}

exit ${EXIT_CODE}

