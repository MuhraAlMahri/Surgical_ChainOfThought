#!/bin/bash
#SBATCH --job-name=endovis_exp3
#SBATCH --output=slurm/logs/train_exp3_%j.out
#SBATCH --error=slurm/logs/train_exp3_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Training Exp3: Sequential (Separate Models per Stage)
# Stage 1: Initial Assessment (Model 1)
# Stage 2: Findings Identification (Model 2)
# Stage 3: Skipped (0 samples in EndoVis2018)

set -e

echo "=========================================="
echo "Training Exp3: Sequential (Separate Models)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: 1"
echo "Started: $(date)"
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
# Set cache directories to avoid NFS issues
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
export DS_SKIP_CUDA_CHECK=1
mkdir -p $HF_HOME $TRITON_CACHE_DIR

cd "${EXPERIMENTS_DIR}"

# ============================================================================
# Stage 1: Training Exp3 Stage 1 (Initial Assessment)
# ============================================================================
echo ""
echo "=========================================="
echo "Stage 1: Training Exp3 Stage 1 (Initial Assessment)"
echo "=========================================="
echo "Started: $(date)"

CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp3_stage1.yaml"

# Verify config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Run training Stage 1
python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
    "${CONFIG_FILE}"

STAGE1_STATUS=$?
if [ $STAGE1_STATUS -eq 0 ]; then
    echo "✓ Exp3 Stage 1 training completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp3_sequential/stage1"
else
    echo "✗ Exp3 Stage 1 training failed: $(date) (exit code: $STAGE1_STATUS)"
    exit $STAGE1_STATUS
fi

# ============================================================================
# Stage 2: Training Exp3 Stage 2 (Findings Identification)
# ============================================================================
echo ""
echo "=========================================="
echo "Stage 2: Training Exp3 Stage 2 (Findings Identification)"
echo "=========================================="
echo "Started: $(date)"

CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp3_stage2.yaml"

# Verify config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Run training Stage 2 (separate model, not continuing from Stage 1)
python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
    "${CONFIG_FILE}"

STAGE2_STATUS=$?
if [ $STAGE2_STATUS -eq 0 ]; then
    echo "✓ Exp3 Stage 2 training completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp3_sequential/stage2"
else
    echo "✗ Exp3 Stage 2 training failed: $(date) (exit code: $STAGE2_STATUS)"
    exit $STAGE2_STATUS
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "✓ Exp3 Training Complete!"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Models saved:"
echo "  Stage 1: ${EXPERIMENTS_DIR}/models/exp3_sequential/stage1"
echo "  Stage 2: ${EXPERIMENTS_DIR}/models/exp3_sequential/stage2"
echo "  Stage 3: Skipped (0 samples in EndoVis2018)"
echo "=========================================="
















