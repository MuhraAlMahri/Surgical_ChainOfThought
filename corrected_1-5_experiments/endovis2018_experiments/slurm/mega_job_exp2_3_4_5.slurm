#!/bin/bash
#SBATCH --job-name=endovis_exp2345
#SBATCH --output=slurm/logs/mega_job_exp2345_%j.out
#SBATCH --error=slurm/logs/mega_job_exp2345_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Mega Job: Training Exp2, Exp3, Exp4, Exp5 for EndoVis2018
# Runs 3 jobs in parallel, then Exp5 sequentially:
#   GPU 0: Training Exp2 (Qwen Reordered) → then Exp5 (Sequential CoT)
#   GPU 1: Training Exp3 (Sequential)
#   GPU 2: Training Exp4 Stage 1 (Curriculum - Initial Assessment)
#   GPU 3: Skipped (Hardware ECC error - Exp5 moved to GPU 0)
# IMPORTANT: All jobs run on compute nodes (MBZUAI HPC policy)
# NOTE: Exp4 Stage 2 will run after Stage 1 completes (sequential)
# NOTE: Exp5 will run on GPU 0 after Exp2 completes (GPU 3 has hardware error)

set -e

echo "=========================================="
echo "MEGA JOB: EndoVis2018 Experiments 2,3,4,5"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4 (parallel execution)"
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
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p ${HF_HOME} ${TRITON_CACHE_DIR}
# Disable DeepSpeed CUDA checks
export DS_SKIP_CUDA_CHECK=1

# CUDA environment
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH

# ============================================================================
# GPU 0: Training Exp2 (Qwen Reordered), then Exp5 (Sequential CoT)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu0
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 0: Training Exp2 (Qwen Reordered)"
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
    
    # Now run Exp5 on GPU 0 (GPU 3 has hardware ECC error)
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
EXP2_PID=$!

# ============================================================================
# GPU 1: Training Exp3 (Sequential - Separate Models per Stage)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu1
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 1: Training Exp3 Stage 1 (Sequential)"
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
    echo "GPU 1: Training Exp3 Stage 2 (Sequential)"
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
    # So we skip training Stage 3 model
    echo ""
    echo "Note: Exp3 Stage 3 skipped (0 samples - EndoVis2018 has no clinical diagnosis questions)"
) 2>&1 | sed 's/^/[GPU 1] /' &
EXP3_PID=$!

# ============================================================================
# GPU 2: Training Exp4 Stage 1 (Curriculum - Initial Assessment)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=2
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu2
    export CUDA_HOME=/apps/local/nvidia/cuda-12.0
    export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 2: Training Exp4 Stage 1 (Curriculum)"
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
    echo "GPU 2: Training Exp4 Stage 2 (Curriculum)"
    echo "=========================================="
    echo "Started: $(date)"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp4_stage2.yaml"
    
    # Verify Stage 1 checkpoint exists
    STAGE1_CHECKPOINT="${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/checkpoint-420"
    if [ ! -d "${STAGE1_CHECKPOINT}" ]; then
        echo "WARNING: Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}"
        echo "Looking for latest checkpoint..."
        LATEST_CHECKPOINT=$(ls -td ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage1/checkpoint-* 2>/dev/null | head -1)
        if [ -n "${LATEST_CHECKPOINT}" ]; then
            echo "Using latest checkpoint: ${LATEST_CHECKPOINT}"
            # Update config to use latest checkpoint
            sed -i "s|prev_checkpoint:.*|prev_checkpoint: ${LATEST_CHECKPOINT}|" "${CONFIG_FILE}"
        else
            echo "ERROR: No Stage 1 checkpoint found!"
            exit 1
        fi
    fi
    
    # Run training Stage 2
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    echo "✓ Exp4 Stage 2 training completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage2"
) 2>&1 | sed 's/^/[GPU 2] /' &
EXP4_PID=$!

# ============================================================================
# GPU 3: Skipped (Hardware ECC error detected)
# Exp5 will run on GPU 0 after Exp2 completes
# ============================================================================
echo ""
echo "=========================================="
echo "GPU 3: Skipped (Hardware ECC error)"
echo "Exp5 will run on GPU 0 after Exp2 completes"
echo "=========================================="

# ============================================================================
# Wait for all jobs to complete
# ============================================================================
echo ""
echo "=========================================="
echo "Experiments started"
echo "=========================================="
echo "Exp2 (GPU 0): PID ${EXP2_PID} → then Exp5"
echo "Exp3 (GPU 1): PID ${EXP3_PID} (Stage 1 → Stage 2 sequential, separate models)"
echo "Exp4 (GPU 2): PID ${EXP4_PID} (Stage 1 → Stage 2 sequential, curriculum)"
echo "Exp5 (GPU 0): Will run after Exp2 completes (GPU 3 has hardware error)"
echo ""
echo "Waiting for all jobs to complete..."
echo ""

# Wait for all background jobs
# Note: Exp5 runs on GPU 0 after Exp2, so waiting for EXP2_PID covers both
wait $EXP2_PID
GPU0_STATUS=$?
# Extract individual statuses from the combined GPU 0 job
# Since Exp2 and Exp5 run sequentially, if GPU0_STATUS is 0, both succeeded
# If non-zero, we need to check which one failed (this is handled in the GPU 0 section)
if [ $GPU0_STATUS -eq 0 ]; then
    EXP2_STATUS=0
    EXP5_STATUS=0
else
    # If GPU 0 job failed, we can't distinguish which experiment failed
    # But the error message in the GPU 0 section will indicate which one
    EXP2_STATUS=$GPU0_STATUS
    EXP5_STATUS=$GPU0_STATUS
fi

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
if [ $EXP2_STATUS -eq 0 ]; then
    echo "  ✓ Exp2 (Qwen Reordered): SUCCESS"
else
    echo "  ✗ Exp2 (Qwen Reordered): FAILED (exit code: $EXP2_STATUS)"
fi

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
echo "  Exp2: ${EXPERIMENTS_DIR}/models/exp2_qwen_reordered"
echo "  Exp3: ${EXPERIMENTS_DIR}/models/exp3_sequential/stage1 and stage2 (separate models)"
echo "  Exp4: ${EXPERIMENTS_DIR}/models/exp4_curriculum/stage2 (curriculum: stage1→stage2)"
echo "  Exp5: ${EXPERIMENTS_DIR}/models/exp5_sequential_cot"
echo "=========================================="

# Exit with error if any job failed
if [ $EXP2_STATUS -ne 0 ] || [ $EXP3_STATUS -ne 0 ] || [ $EXP4_STATUS -ne 0 ] || [ $EXP5_STATUS -ne 0 ]; then
    exit 1
fi

exit 0

