#!/bin/bash
#SBATCH --job-name=endovis_all_opt
#SBATCH --output=slurm/logs/mega_job_all_optimized_%j.out
#SBATCH --error=slurm/logs/mega_job_all_optimized_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# OPTIMIZED MEGA JOB: Train ALL 5 experiments in parallel (6-hour target)
# 
# Optimizations applied:
#   - batch_size: 1 → 4 (4x faster data loading)
#   - grad_accum: 16 → 4 (maintains effective batch size = 16)
#   - gradient_checkpointing: true → false (2x speedup)
#   - dataloader_num_workers: 4 → 16 (4x parallel I/O)
#   - Expected speedup: 4-8x faster
#
# GPU Distribution:
#   GPU 0: Exp1 (Random Baseline)
#   GPU 1: Exp2 (Qwen Reordered)
#   GPU 2: Exp3 (Sequential - Stage 1 → Stage 2, separate models)
#   GPU 3: Exp4 (Curriculum - Stage 1 → Stage 2, continuing from Stage 1)
#   Note: Exp5 will run after Exp1 completes on GPU 0 (sequential)

set -e

echo "=========================================="
echo "OPTIMIZED MEGA JOB: All 5 Experiments"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4 (parallel execution)"
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
# GPU 0: Training Exp1, then Exp5 (Sequential)
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
    
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp1_random.yaml"
    
    EXP1_STATUS=$?
    if [ $EXP1_STATUS -eq 0 ]; then
        echo "✓ Exp1 completed: $(date)"
    else
        echo "✗ Exp1 failed: $(date) (exit code: $EXP1_STATUS)"
        exit $EXP1_STATUS
    fi
    
    echo ""
    echo "=========================================="
    echo "GPU 0: Training Exp5 (Sequential CoT)"
    echo "=========================================="
    echo "Started: $(date)"
    
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp5_sequential_cot.yaml"
    
    EXP5_STATUS=$?
    if [ $EXP5_STATUS -eq 0 ]; then
        echo "✓ Exp5 completed: $(date)"
    else
        echo "✗ Exp5 failed: $(date) (exit code: $EXP5_STATUS)"
        exit $EXP5_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 0] /' &
EXP1_EXP5_PID=$!

# ============================================================================
# GPU 1: Training Exp2
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
    
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp2_qwen_reordered.yaml"
    
    EXP2_STATUS=$?
    if [ $EXP2_STATUS -eq 0 ]; then
        echo "✓ Exp2 completed: $(date)"
    else
        echo "✗ Exp2 failed: $(date) (exit code: $EXP2_STATUS)"
        exit $EXP2_STATUS
    fi
) 2>&1 | sed 's/^/[GPU 1] /' &
EXP2_PID=$!

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
    echo "GPU 2: Training Exp3 Stage 1"
    echo "=========================================="
    echo "Started: $(date)"
    
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp3_stage1.yaml"
    
    echo "✓ Exp3 Stage 1 completed: $(date)"
    
    echo ""
    echo "=========================================="
    echo "GPU 2: Training Exp3 Stage 2"
    echo "=========================================="
    echo "Started: $(date)"
    
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp3_stage2.yaml"
    
    echo "✓ Exp3 Stage 2 completed: $(date)"
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
    
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${EXPERIMENTS_DIR}/configs/exp4_stage1.yaml"
    
    echo "✓ Exp4 Stage 1 completed: $(date)"
    
    # Update Stage 2 config with latest checkpoint
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp4_stage2.yaml"
    LATEST_CHECKPOINT=$(ls -td ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/checkpoint-* 2>/dev/null | head -1)
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        echo "Using checkpoint: ${LATEST_CHECKPOINT}"
        sed -i "s|# prev_checkpoint:.*|prev_checkpoint: ${LATEST_CHECKPOINT}|" "${CONFIG_FILE}"
        sed -i "s|prev_checkpoint:.*|prev_checkpoint: ${LATEST_CHECKPOINT}|" "${CONFIG_FILE}"
    else
        echo "ERROR: No Stage 1 checkpoint found!"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "GPU 3: Training Exp4 Stage 2 (Curriculum)"
    echo "=========================================="
    echo "Started: $(date)"
    
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    echo "✓ Exp4 Stage 2 completed: $(date)"
) 2>&1 | sed 's/^/[GPU 3] /' &
EXP4_PID=$!

# ============================================================================
# Wait for all jobs
# ============================================================================
echo ""
echo "=========================================="
echo "All Experiments Started"
echo "=========================================="
echo "Exp1→Exp5 (GPU 0): PID ${EXP1_EXP5_PID}"
echo "Exp2 (GPU 1): PID ${EXP2_PID}"
echo "Exp3 (GPU 2): PID ${EXP3_PID}"
echo "Exp4 (GPU 3): PID ${EXP4_PID}"
echo ""
echo "Waiting for completion..."
echo ""

wait $EXP1_EXP5_PID
EXP1_EXP5_STATUS=$?

wait $EXP2_PID
EXP2_STATUS=$?

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

if [ $EXP1_EXP5_STATUS -eq 0 ]; then
    echo "  ✓ Exp1 & Exp5: SUCCESS"
else
    echo "  ✗ Exp1 & Exp5: FAILED"
fi

if [ $EXP2_STATUS -eq 0 ]; then
    echo "  ✓ Exp2: SUCCESS"
else
    echo "  ✗ Exp2: FAILED"
fi

if [ $EXP3_STATUS -eq 0 ]; then
    echo "  ✓ Exp3: SUCCESS"
else
    echo "  ✗ Exp3: FAILED"
fi

if [ $EXP4_STATUS -eq 0 ]; then
    echo "  ✓ Exp4: SUCCESS"
else
    echo "  ✗ Exp4: FAILED"
fi

echo ""
echo "⚡ All models trained with OPTIMIZED configuration"
echo "=========================================="

# Exit with error if any job failed
if [ $EXP1_EXP5_STATUS -ne 0 ] || [ $EXP2_STATUS -ne 0 ] || [ $EXP3_STATUS -ne 0 ] || [ $EXP4_STATUS -ne 0 ]; then
    exit 1
fi

exit 0

