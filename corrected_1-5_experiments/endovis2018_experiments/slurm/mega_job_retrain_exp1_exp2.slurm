#!/bin/bash
#SBATCH --job-name=endovis_exp1_exp2
#SBATCH --output=slurm/logs/mega_job_exp1_exp2_%j.out
#SBATCH --error=slurm/logs/mega_job_exp1_exp2_%j.err
#SBATCH --time=06:00:00  # OPTIMIZED: Reduced for faster training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# 2-GPU Job: Training Exp1 and Exp2 with CORRECTED Dataset
# 
# GPU Distribution:
#   GPU 0: Exp1 (Random Baseline)
#   GPU 1: Exp2 (Qwen Reordered)
#
# IMPORTANT: All experiments are retrained with the CORRECTED dataset
#            (fixed mask paths, no more incorrect "none"/"0" answers)

set -e

echo "=========================================="
echo "2-GPU JOB: Training Exp1 and Exp2"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 2 (parallel execution)"
echo "Started: $(date)"
echo ""
echo "⚠️  RETRAINING WITH CORRECTED DATASET"
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
# Use user directory instead of /tmp to avoid disk space issues
export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}
mkdir -p ${HF_HOME} ${TRITON_CACHE_DIR}
# Disable DeepSpeed CUDA checks
export DS_SKIP_CUDA_CHECK=1

# CUDA environment
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH

# ============================================================================
# GPU 0: Training Exp1 (Random Baseline)
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
    echo "GPU 0: Training Exp1 (Random Baseline)"
    echo "=========================================="
    echo "Started: $(date)"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp1_random.yaml"
    
    # Verify config exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Run training
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    EXP1_STATUS=$?
    if [ $EXP1_STATUS -eq 0 ]; then
        echo "✓ Exp1 training completed: $(date)"
        echo "  Model: ${EXPERIMENTS_DIR}/models/exp1_random"
    else
        echo "✗ Exp1 training failed: $(date) (exit code: $EXP1_STATUS)"
        exit $EXP1_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 0] /' &
EXP1_PID=$!

# ============================================================================
# GPU 1: Training Exp2 (Qwen Reordered)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu1
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu1
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 1: Training Exp2 (Qwen Reordered)"
    echo "=========================================="
    echo "Started: $(date)"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp2_qwen_reordered.yaml"
    
    # Verify config exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Run training
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    EXP2_STATUS=$?
    if [ $EXP2_STATUS -eq 0 ]; then
        echo "✓ Exp2 training completed: $(date)"
        echo "  Model: ${EXPERIMENTS_DIR}/models/exp2_qwen_reordered"
    else
        echo "✗ Exp2 training failed: $(date) (exit code: $EXP2_STATUS)"
        exit $EXP2_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 1] /' &
EXP2_PID=$!

# ============================================================================
# Wait for all jobs to complete
# ============================================================================
echo ""
echo "=========================================="
echo "Experiments Started"
echo "=========================================="
echo "Exp1 (GPU 0): PID ${EXP1_PID}"
echo "Exp2 (GPU 1): PID ${EXP2_PID}"
echo ""
echo "Waiting for all jobs to complete..."
echo ""

# Wait for all background jobs
wait $EXP1_PID
EXP1_STATUS=$?

wait $EXP2_PID
EXP2_STATUS=$?

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Status:"
if [ $EXP1_STATUS -eq 0 ]; then
    echo "  ✓ Exp1 (Random Baseline): SUCCESS"
else
    echo "  ✗ Exp1 (Random Baseline): FAILED (exit code: $EXP1_STATUS)"
fi

if [ $EXP2_STATUS -eq 0 ]; then
    echo "  ✓ Exp2 (Qwen Reordered): SUCCESS"
else
    echo "  ✗ Exp2 (Qwen Reordered): FAILED (exit code: $EXP2_STATUS)"
fi

echo ""
echo "Model outputs:"
echo "  Exp1: ${EXPERIMENTS_DIR}/models/exp1_random"
echo "  Exp2: ${EXPERIMENTS_DIR}/models/exp2_qwen_reordered"
echo ""
echo "⚠️  Models trained with CORRECTED dataset"
echo "=========================================="

# Exit with error if any job failed
if [ $EXP1_STATUS -ne 0 ] || [ $EXP2_STATUS -ne 0 ]; then
    exit 1
fi

exit 0





