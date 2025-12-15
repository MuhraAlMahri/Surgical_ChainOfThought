#!/bin/bash
#SBATCH --job-name=exp4_curriculum_all
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp4_all_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp4_all_%j.err

# ============================================================================
# EXP4 CURRICULUM LEARNING - ALL STAGES (Sequential)
# ============================================================================
# This script trains Exp4 curriculum learning sequentially:
# Stage 1 → Stage 2 → Stage 3
# Each stage continues training the same LoRA adapter
# ============================================================================

echo "=========================================="
echo "EXP4 CURRICULUM LEARNING - ALL STAGES"
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
VERIFY_SCRIPT="${BASE_DIR}/verify_exp4_fix.py"

# ============================================================================
# STAGE 1: Initial Assessment
# ============================================================================
echo ""
echo "=========================================="
echo "STAGE 1: Initial Assessment"
echo "=========================================="

STAGE1_CONFIG="${BASE_DIR}/configs/exp4_stage1.yaml"
STAGE1_OUTPUT="${BASE_DIR}/models/exp4_curriculum/stage1"

echo "Config: ${STAGE1_CONFIG}"
echo "Output: ${STAGE1_OUTPUT}"
echo ""

nvidia-smi
echo ""

echo "Starting Stage 1 training..."
python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${STAGE1_CONFIG}

STAGE1_EXIT=$?

if [ ${STAGE1_EXIT} -ne 0 ]; then
    echo "❌ Stage 1 training failed with exit code: ${STAGE1_EXIT}"
    exit ${STAGE1_EXIT}
fi

# Verify Stage 1 adapter
echo ""
echo "Verifying Stage 1 adapter..."
if [ -f "${VERIFY_SCRIPT}" ]; then
    python3 ${VERIFY_SCRIPT} ${STAGE1_OUTPUT} || echo "⚠️  Verification script failed, but continuing..."
else
    echo "⚠️  Verification script not found, skipping verification"
fi

# Check if adapter exists
if [ ! -f "${STAGE1_OUTPUT}/adapter_model.safetensors" ] && [ ! -d "${STAGE1_OUTPUT}/checkpoint-"* ]; then
    echo "❌ ERROR: Stage 1 adapter not found at ${STAGE1_OUTPUT}"
    exit 1
fi

echo "✓ Stage 1 completed successfully"
echo "Stage 1 end time: $(date)"
echo ""

# ============================================================================
# STAGE 2: Findings Identification
# ============================================================================
echo "=========================================="
echo "STAGE 2: Findings Identification"
echo "=========================================="

STAGE2_CONFIG="${BASE_DIR}/configs/exp4_stage2.yaml"
STAGE2_OUTPUT="${BASE_DIR}/models/exp4_curriculum/stage2"

echo "Config: ${STAGE2_CONFIG}"
echo "Output: ${STAGE2_OUTPUT}"
echo "Continuing from: ${STAGE1_OUTPUT}"
echo ""

# Verify Stage 1 checkpoint exists
if [ ! -f "${STAGE1_OUTPUT}/adapter_model.safetensors" ] && [ ! -d "${STAGE1_OUTPUT}/checkpoint-"* ]; then
    echo "❌ ERROR: Stage 1 checkpoint not found at ${STAGE1_OUTPUT}"
    exit 1
fi

echo "Starting Stage 2 training (continuing from Stage 1)..."
python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${STAGE2_CONFIG}

STAGE2_EXIT=$?

if [ ${STAGE2_EXIT} -ne 0 ]; then
    echo "❌ Stage 2 training failed with exit code: ${STAGE2_EXIT}"
    exit ${STAGE2_EXIT}
fi

# Verify Stage 2 adapter
echo ""
echo "Verifying Stage 2 adapter..."
if [ -f "${VERIFY_SCRIPT}" ]; then
    python3 ${VERIFY_SCRIPT} ${STAGE2_OUTPUT} || echo "⚠️  Verification script failed, but continuing..."
fi

# Check if adapter exists
if [ ! -f "${STAGE2_OUTPUT}/adapter_model.safetensors" ] && [ ! -d "${STAGE2_OUTPUT}/checkpoint-"* ]; then
    echo "❌ ERROR: Stage 2 adapter not found at ${STAGE2_OUTPUT}"
    exit 1
fi

echo "✓ Stage 2 completed successfully"
echo "Stage 2 end time: $(date)"
echo ""

# ============================================================================
# STAGE 3: Clinical Context
# ============================================================================
echo "=========================================="
echo "STAGE 3: Clinical Context"
echo "=========================================="

STAGE3_CONFIG="${BASE_DIR}/configs/exp4_stage3.yaml"
STAGE3_OUTPUT="${BASE_DIR}/models/exp4_curriculum/stage3"

echo "Config: ${STAGE3_CONFIG}"
echo "Output: ${STAGE3_OUTPUT}"
echo "Continuing from: ${STAGE2_OUTPUT}"
echo ""

# Verify Stage 2 checkpoint exists
if [ ! -f "${STAGE2_OUTPUT}/adapter_model.safetensors" ] && [ ! -d "${STAGE2_OUTPUT}/checkpoint-"* ]; then
    echo "❌ ERROR: Stage 2 checkpoint not found at ${STAGE2_OUTPUT}"
    exit 1
fi

echo "Starting Stage 3 training (continuing from Stage 2)..."
python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${STAGE3_CONFIG}

STAGE3_EXIT=$?

if [ ${STAGE3_EXIT} -ne 0 ]; then
    echo "❌ Stage 3 training failed with exit code: ${STAGE3_EXIT}"
    exit ${STAGE3_EXIT}
fi

# Verify Stage 3 adapter
echo ""
echo "Verifying Stage 3 adapter..."
if [ -f "${VERIFY_SCRIPT}" ]; then
    python3 ${VERIFY_SCRIPT} ${STAGE3_OUTPUT} || echo "⚠️  Verification script failed, but continuing..."
fi

# Check if adapter exists
if [ ! -f "${STAGE3_OUTPUT}/adapter_model.safetensors" ] && [ ! -d "${STAGE3_OUTPUT}/checkpoint-"* ]; then
    echo "❌ ERROR: Stage 3 adapter not found at ${STAGE3_OUTPUT}"
    exit 1
fi

echo "✓ Stage 3 completed successfully"
echo "Stage 3 end time: $(date)"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo "=========================================="
echo "EXP4 CURRICULUM LEARNING - ALL STAGES COMPLETE"
echo "=========================================="
echo "Stage 1: Exit code ${STAGE1_EXIT}"
echo "Stage 2: Exit code ${STAGE2_EXIT}"
echo "Stage 3: Exit code ${STAGE3_EXIT}"
echo ""
echo "Final model: ${STAGE3_OUTPUT}"
echo "End time: $(date)"
echo "=========================================="

# Cleanup
rm -rf ${HF_HOME}
rm -rf ${TRITON_CACHE_DIR}

# Exit with Stage 3's exit code
exit ${STAGE3_EXIT}

