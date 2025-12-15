#!/bin/bash
#SBATCH --job-name=endovis_retrain_all
#SBATCH --output=slurm/logs/mega_job_retrain_all_%j.out
#SBATCH --error=slurm/logs/mega_job_retrain_all_%j.err
#SBATCH --time=06:00:00  # OPTIMIZED: Target 6 hours total
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Mega Job: Retraining Exp3, Exp4, Exp5 with CORRECTED Dataset
# This job retrains the remaining 3 experiments after fixing the dataset mask path issue
# Note: Exp1 and Exp2 should be trained separately (use mega_job_retrain_exp1_exp2.slurm)
# 
# GPU Distribution:
#   GPU 0: Exp5 (Sequential CoT)
#   GPU 1: (available for future use)
#   GPU 2: Exp3 (Sequential - Stage 1 → Stage 2, separate models)
#   GPU 3: Exp4 (Curriculum - Stage 1 → Stage 2, continuing from Stage 1)
#
# IMPORTANT: All experiments are retrained with the CORRECTED dataset
#            (fixed mask paths, no more incorrect "none"/"0" answers)

set -e

echo "=========================================="
echo "MEGA JOB: Retraining Exp3, Exp4, Exp5"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4 (parallel execution)"
echo "Started: $(date)"
echo ""
echo "⚠️  RETRAINING WITH CORRECTED DATASET"
echo "   All previous results were invalid due to incorrect mask paths"
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
# GPU 0: Training Exp5 (Sequential CoT)
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
    echo "GPU 0: Training Exp5 (Sequential CoT)"
    echo "=========================================="
    echo "Started: $(date)"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp5_sequential_cot.yaml"
    
    # Verify config exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Run training
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    EXP5_STATUS=$?
    if [ $EXP5_STATUS -eq 0 ]; then
        echo "✓ Exp5 training completed: $(date)"
        echo "  Model: ${EXPERIMENTS_DIR}/models/exp5_sequential_cot"
    else
        echo "✗ Exp5 training failed: $(date) (exit code: $EXP5_STATUS)"
        exit $EXP5_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 0] /' &
EXP5_PID=$!

# ============================================================================
# GPU 2: Training Exp3 (Sequential - Separate Models per Stage)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=2
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu2
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu2
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 2: Training Exp3 Stage 1 (Sequential)"
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
    
    echo "✓ Exp3 Stage 1 training completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp3_sequential/stage1"
    
    # Now run Stage 2 (separate model, not continuing from Stage 1)
    echo ""
    echo "=========================================="
    echo "GPU 2: Training Exp3 Stage 2 (Sequential)"
    echo "=========================================="
    echo "Started: $(date)"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp3_stage2.yaml"
    
    # Verify config exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Run training Stage 2 (separate model)
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    echo "✓ Exp3 Stage 2 training completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp3_sequential/stage2"
    
    # Note: Stage 3 has 0 samples (EndoVis2018 has no clinical diagnosis questions)
    echo ""
    echo "Note: Exp3 Stage 3 skipped (0 samples - EndoVis2018 has no clinical diagnosis questions)"
) 2>&1 | sed 's/^/[GPU 2] /' &
EXP3_PID=$!

# ============================================================================
# GPU 3: Training Exp4 (Curriculum - Stage 1 → Stage 2)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=3
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu3
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu3
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 3: Training Exp4 Stage 1 (Curriculum)"
    echo "=========================================="
    echo "Started: $(date)"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp4_stage1.yaml"
    
    # Verify config exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Run training Stage 1
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    echo "✓ Exp4 Stage 1 training completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1"
    
    # Now run Stage 2 (continuing from Stage 1)
    echo ""
    echo "=========================================="
    echo "GPU 3: Training Exp4 Stage 2 (Curriculum)"
    echo "=========================================="
    echo "Started: $(date)"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp4_stage2.yaml"
    
    # Verify Stage 1 checkpoint exists and update config
    echo "Looking for latest Stage 1 checkpoint..."
    LATEST_CHECKPOINT=$(ls -td ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/checkpoint-* 2>/dev/null | head -1)
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        echo "Using latest checkpoint: ${LATEST_CHECKPOINT}"
        # Update config to use latest checkpoint
        sed -i "s|# prev_checkpoint:.*|prev_checkpoint: ${LATEST_CHECKPOINT}|" "${CONFIG_FILE}"
        sed -i "s|prev_checkpoint:.*|prev_checkpoint: ${LATEST_CHECKPOINT}|" "${CONFIG_FILE}"
    else
        echo "ERROR: No Stage 1 checkpoint found! Stage 1 must complete first."
        exit 1
    fi
    
    # Run training Stage 2
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    echo "✓ Exp4 Stage 2 training completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage2"
) 2>&1 | sed 's/^/[GPU 3] /' &
EXP4_PID=$!

# ============================================================================
# Wait for all jobs to complete
# ============================================================================
echo ""
echo "=========================================="
echo "All Experiments Started"
echo "=========================================="
echo "Exp5 (GPU 0): PID ${EXP5_PID}"
echo "Exp3 (GPU 2): PID ${EXP3_PID} (Stage 1 → Stage 2 sequential, separate models)"
echo "Exp4 (GPU 3): PID ${EXP4_PID} (Stage 1 → Stage 2 sequential, curriculum)"
echo ""
echo "Waiting for all jobs to complete..."
echo ""

# Wait for all background jobs
wait $EXP5_PID
EXP5_STATUS=$?

wait $EXP3_PID
EXP3_STATUS=$?

wait $EXP4_PID
EXP4_STATUS=$?

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""
echo "Status:"
if [ $EXP3_STATUS -eq 0 ]; then
    echo "  ✓ Exp3 (Sequential): SUCCESS"
else
    echo "  ✗ Exp3 (Sequential): FAILED (exit code: $EXP3_STATUS)"
fi

if [ $EXP4_STATUS -eq 0 ]; then
    echo "  ✓ Exp4 (Curriculum): SUCCESS"
else
    echo "  ✗ Exp4 (Curriculum): FAILED (exit code: $EXP4_STATUS)"
fi

if [ $EXP5_STATUS -eq 0 ]; then
    echo "  ✓ Exp5 (Sequential CoT): SUCCESS"
else
    echo "  ✗ Exp5 (Sequential CoT): FAILED (exit code: $EXP5_STATUS)"
fi

echo ""
echo "Model outputs:"
echo "  Exp3: ${EXPERIMENTS_DIR}/models/exp3_sequential/stage1 and stage2 (separate models)"
echo "  Exp4: ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage2 (curriculum: stage1→stage2)"
echo "  Exp5: ${EXPERIMENTS_DIR}/models/exp5_sequential_cot"
echo ""
echo "⚠️  All models trained with CORRECTED dataset"
echo "   (fixed mask paths, correct answers)"
echo ""
echo "Note: Exp1 and Exp2 should be trained separately"
echo "=========================================="

# Exit with error if any job failed
if [ $EXP3_STATUS -ne 0 ] || [ $EXP4_STATUS -ne 0 ] || [ $EXP5_STATUS -ne 0 ]; then
    exit 1
fi

exit 0

