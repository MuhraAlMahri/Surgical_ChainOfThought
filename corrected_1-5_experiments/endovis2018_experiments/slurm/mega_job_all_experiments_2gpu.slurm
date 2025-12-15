#!/bin/bash
#SBATCH --job-name=endovis_all_2gpu
#SBATCH --output=slurm/logs/mega_job_all_2gpu_%j.out
#SBATCH --error=slurm/logs/mega_job_all_2gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# OPTIMIZED JOB: Train ALL 5 experiments sequentially on 2 GPUs (6-hour target)
# 
# Optimizations applied:
#   - batch_size: 1 → 4 (4x faster data loading)
#   - grad_accum: 16 → 4 (maintains effective batch size = 16)
#   - gradient_checkpointing: true → false (2x speedup)
#   - dataloader_num_workers: 4 → 16 (4x parallel I/O)
#   - Expected speedup: 4-8x faster
#
# GPU Distribution (Sequential):
#   GPU 0: Exp1 → Exp3 Stage1 → Exp3 Stage2 → Exp4 Stage1 → Exp4 Stage2
#   GPU 1: Exp2 → Exp5
#
# This runs on 2 GPUs to avoid QOSMaxGRESPerUser limit

set -e

echo "=========================================="
echo "OPTIMIZED JOB: All 5 Experiments (2 GPUs)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 2 (sequential execution)"
echo "Started: $(date)"
echo ""
echo "⚡ OPTIMIZED CONFIGURATION:"
echo "   - batch_size: 4 (was 1)"
echo "   - gradient_checkpointing: false (was true)"
echo "   - dataloader_num_workers: 16 (was 4)"
echo "   - Expected speedup: 4-8x"
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
# GPU 0: Sequential training of Exp1, Exp2, Exp5 (better balance)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu0
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu0
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    # Exp1
    echo ""
    echo "=========================================="
    echo "GPU 0: Training Exp1 (Random Baseline)"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp1_random.yaml"
    echo "✓ Exp1 completed: $(date)"
    
    # Exp2
    echo ""
    echo "=========================================="
    echo "GPU 0: Training Exp2 (Qwen Reordered)"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp2_qwen_reordered.yaml"
    echo "✓ Exp2 completed: $(date)"
    
    # Exp5
    echo ""
    echo "=========================================="
    echo "GPU 0: Training Exp5 (Sequential CoT)"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp5_sequential_cot.yaml"
    echo "✓ Exp5 completed: $(date)"
) 2>&1 | sed 's/^/[GPU 0] /' &
GPU0_PID=$!

# ============================================================================
# GPU 1: Sequential training of Exp3, Exp4 (all stages)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}_gpu1
    export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}_gpu1
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    # Exp3 Stage 1
    echo ""
    echo "=========================================="
    echo "GPU 1: Training Exp3 Stage 1"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp3_stage1.yaml"
    echo "✓ Exp3 Stage 1 completed: $(date)"
    
    # Exp3 Stage 2
    echo ""
    echo "=========================================="
    echo "GPU 1: Training Exp3 Stage 2"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp3_stage2.yaml"
    echo "✓ Exp3 Stage 2 completed: $(date)"
    
    # Exp4 Stage 1
    echo ""
    echo "=========================================="
    echo "GPU 1: Training Exp4 Stage 1 (Curriculum)"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp4_stage1.yaml"
    echo "✓ Exp4 Stage 1 completed: $(date)"
    
    # Exp4 Stage 2 (update checkpoint path)
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp4_stage2.yaml"
    LATEST_CHECKPOINT=$(ls -td ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/checkpoint-* 2>/dev/null | head -1)
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        echo "Using checkpoint: ${LATEST_CHECKPOINT}"
        sed -i "s|# prev_checkpoint:.*|prev_checkpoint: ${LATEST_CHECKPOINT}|" "${CONFIG_FILE}"
        sed -i "s|prev_checkpoint:.*|prev_checkpoint: ${LATEST_CHECKPOINT}|" "${CONFIG_FILE}"
    fi
    
    echo ""
    echo "=========================================="
    echo "GPU 1: Training Exp4 Stage 2 (Curriculum)"
    echo "=========================================="
    echo "Started: $(date)"
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    echo "✓ Exp4 Stage 2 completed: $(date)"
) 2>&1 | sed 's/^/[GPU 1] /' &
GPU1_PID=$!

# ============================================================================
# Wait for all jobs
# ============================================================================
echo ""
echo "=========================================="
echo "All Experiments Started"
echo "=========================================="
echo "GPU 0: Exp1 → Exp2 → Exp5 (PID ${GPU0_PID})"
echo "GPU 1: Exp3 → Exp4 (all stages) (PID ${GPU1_PID})"
echo ""
echo "Waiting for completion..."
echo ""

wait $GPU0_PID
GPU0_STATUS=$?

wait $GPU1_PID
GPU1_STATUS=$?

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""

if [ $GPU0_STATUS -eq 0 ] && [ $GPU1_STATUS -eq 0 ]; then
    echo "  ✓ All experiments: SUCCESS"
else
    echo "  ✗ Some experiments failed"
    echo "    GPU 0 status: ${GPU0_STATUS}"
    echo "    GPU 1 status: ${GPU1_STATUS}"
fi

echo ""
echo "⚡ All models trained with OPTIMIZED configuration"
echo "=========================================="

# Exit with error if any job failed
if [ $GPU0_STATUS -ne 0 ] || [ $GPU1_STATUS -ne 0 ]; then
    exit 1
fi

exit 0

