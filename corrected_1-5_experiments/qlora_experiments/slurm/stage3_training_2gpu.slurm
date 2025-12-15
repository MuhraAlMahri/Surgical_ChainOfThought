#!/bin/bash
#SBATCH --job-name=qlora_s3_train_2gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/s3_train_2gpu_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/s3_train_2gpu_%j.err

# ============================================================================
# STAGE 3 TRAINING - 2 GPUs
# ============================================================================
# GPU 0: Exp3-S3 - Training (CXRTrek Stage 3)
# GPU 1: Exp4-S3 - Training (Curriculum Stage 3)
# ============================================================================
# NOTE: This job runs AFTER Exp3-S2 and Exp4-S2 (from job 156712) complete
# ============================================================================

echo "=========================================="
echo "STAGE 3 TRAINING - 2 GPUs"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Setup
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"

# Function to run experiment on a specific GPU
run_experiment() {
    local GPU_ID=$1
    local EXP_NAME=$2
    local COMMAND=$3
    
    echo ""
    echo "=========================================="
    echo "Starting ${EXP_NAME} on GPU ${GPU_ID}"
    echo "=========================================="
    
    # Set GPU-specific environment
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu${GPU_ID}
    export TRANSFORMERS_CACHE=${HF_HOME}
    export HF_HUB_CACHE=${HF_HOME}
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu${GPU_ID}
    mkdir -p ${HF_HOME}
    
    # Run command
    eval ${COMMAND} > ${BASE_DIR}/slurm/logs/s3_train_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/s3_train_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
    local EXIT_CODE=$?
    
    echo "=========================================="
    echo "${EXP_NAME} on GPU ${GPU_ID} completed with exit code: ${EXIT_CODE}"
    echo "End time: $(date)"
    echo "=========================================="
    
    # Cleanup
    rm -rf ${HF_HOME}
    rm -rf ${TRITON_CACHE_DIR}
    
    return ${EXIT_CODE}
}

# Show GPU status
nvidia-smi
echo ""

# ============================================================================
# GPU 0: Exp3-S3 Training
# ============================================================================
EXP3_S3_CONFIG="${BASE_DIR}/configs/exp3_stage3.yaml"
EXP3_S2_CHECKPOINT="${BASE_DIR}/models/exp3_cxrtrek/stage2"
# Note: Exp3 stages are independent, but we verify Exp3-S2 exists
if [ -d "${EXP3_S2_CHECKPOINT}" ]; then
    echo "Verified: Exp3-S2 checkpoint exists at ${EXP3_S2_CHECKPOINT}"
else
    echo "WARNING: Exp3-S2 checkpoint not found at ${EXP3_S2_CHECKPOINT}"
    echo "Proceeding with Exp3-S3 (independent model)..."
fi
EXP3_S3_CMD="python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP3_S3_CONFIG}"

# ============================================================================
# GPU 1: Exp4-S3 Training (depends on Exp4-S2 checkpoint)
# ============================================================================
EXP4_S3_CONFIG="${BASE_DIR}/configs/exp4_stage3.yaml"
EXP4_S2_CHECKPOINT="${BASE_DIR}/models/exp4_curriculum/stage2"
if [ ! -d "${EXP4_S2_CHECKPOINT}" ]; then
    echo "ERROR: Exp4-S2 checkpoint not found at ${EXP4_S2_CHECKPOINT}"
    echo "This job requires Exp4-S2 (from job 156712) to complete first!"
    exit 1
fi
echo "Verified: Exp4-S2 checkpoint exists at ${EXP4_S2_CHECKPOINT}"
EXP4_S3_CMD="python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP4_S3_CONFIG}"

# ============================================================================
# Run both training jobs in parallel
# ============================================================================
echo "Starting Exp3-S3 and Exp4-S3 training in parallel..."
echo "=========================================="

run_experiment 0 "Exp3-S3 - Training" "${EXP3_S3_CMD}" &
PID0=$!

run_experiment 1 "Exp4-S3 - Training" "${EXP4_S3_CMD}" &
PID1=$!

# Wait for both training jobs to complete
wait ${PID0}
EXIT0=$?

wait ${PID1}
EXIT1=$?

# Verify checkpoints exist
echo ""
echo "Verifying checkpoints..."
EXP3_S3_MODEL="${BASE_DIR}/models/exp3_cxrtrek/stage3"
EXP4_S3_MODEL="${BASE_DIR}/models/exp4_curriculum/stage3"

if [ ! -d "${EXP3_S3_MODEL}" ]; then
    echo "ERROR: Exp3-S3 checkpoint not found at ${EXP3_S3_MODEL}"
    EXIT0=1
else
    echo "✓ Exp3-S3 checkpoint verified: ${EXP3_S3_MODEL}"
fi

if [ ! -d "${EXP4_S3_MODEL}" ]; then
    echo "ERROR: Exp4-S3 checkpoint not found at ${EXP4_S3_MODEL}"
    EXIT1=1
else
    echo "✓ Exp4-S3 checkpoint verified: ${EXP4_S3_MODEL}"
fi

# Report final status
echo ""
echo "=========================================="
echo "STAGE 3 TRAINING - COMPLETED"
echo "=========================================="
echo "Exp3-S3 Training (GPU 0): Exit code ${EXIT0}"
echo "Exp4-S3 Training (GPU 1): Exit code ${EXIT1}"
echo "End time: $(date)"
echo "=========================================="

# Exit with error if any training failed
if [ ${EXIT0} -ne 0 ] || [ ${EXIT1} -ne 0 ]; then
    echo "ERROR: One or more training jobs failed!"
    exit 1
else
    echo "SUCCESS: Both training jobs completed successfully!"
    exit 0
fi







