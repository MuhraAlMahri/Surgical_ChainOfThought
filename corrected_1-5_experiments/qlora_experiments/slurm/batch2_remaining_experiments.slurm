#!/bin/bash
#SBATCH --job-name=qlora_batch2_3gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/batch2_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/batch2_%j.err

# ============================================================================
# BATCH 2: Remaining Experiments (3 GPUs in parallel)
# ============================================================================
# GPU 0: Exp3-S2 - CXRTrek Stage 2 (~12h)
# GPU 1: Exp4-S2 - Curriculum Stage 2 (~12h, continues from Exp4-S1)
# GPU 2: Exp3-S3 - CXRTrek Stage 3 (~30min)
# ============================================================================

echo "=========================================="
echo "BATCH 2 - 3 GPUs Parallel"
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
    local CONFIG=$3
    
    echo ""
    echo "=========================================="
    echo "Starting ${EXP_NAME} on GPU ${GPU_ID}"
    echo "Config: ${CONFIG}"
    echo "=========================================="
    
    # Set GPU-specific environment
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu${GPU_ID}
    export TRANSFORMERS_CACHE=${HF_HOME}
    export HF_HUB_CACHE=${HF_HOME}
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu${GPU_ID}
    mkdir -p ${HF_HOME}
    
    # Run training
    python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${CONFIG} > ${BASE_DIR}/slurm/logs/batch2_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/batch2_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
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

# Verify prerequisites
STAGE1_CHECKPOINT="${BASE_DIR}/models/exp4_curriculum/stage1"
if [ ! -d "${STAGE1_CHECKPOINT}" ]; then
    echo "ERROR: Exp4-Stage1 checkpoint not found at ${STAGE1_CHECKPOINT}"
    echo "Batch 2 requires Batch 1 (Exp4-S1) to complete first!"
    exit 1
fi
echo "Verified: Exp4-Stage1 checkpoint exists at ${STAGE1_CHECKPOINT}"
echo ""

# Show GPU status
nvidia-smi
echo ""

# Run all 3 experiments in parallel
echo "Starting all 3 experiments in parallel..."
echo "=========================================="

run_experiment 0 "Exp3 - CXRTrek Stage 2" "${BASE_DIR}/configs/exp3_stage2.yaml" &
PID0=$!

run_experiment 1 "Exp4 - Curriculum Stage 2" "${BASE_DIR}/configs/exp4_stage2.yaml" &
PID1=$!

run_experiment 2 "Exp3 - CXRTrek Stage 3" "${BASE_DIR}/configs/exp3_stage3.yaml" &
PID2=$!

# Wait for all experiments to complete
echo ""
echo "Waiting for all experiments to complete..."
echo "=========================================="

wait ${PID0}
EXIT0=$?

wait ${PID1}
EXIT1=$?

wait ${PID2}
EXIT2=$?

# Report final status
echo ""
echo "=========================================="
echo "BATCH 2 - ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Exp3-S2 (GPU 0): Exit code ${EXIT0}"
echo "Exp4-S2 (GPU 1): Exit code ${EXIT1}"
echo "Exp3-S3 (GPU 2): Exit code ${EXIT2}"
echo "End time: $(date)"
echo "=========================================="

# Exit with error if any experiment failed
if [ ${EXIT0} -ne 0 ] || [ ${EXIT1} -ne 0 ] || [ ${EXIT2} -ne 0 ]; then
    echo "ERROR: One or more experiments failed!"
    exit 1
else
    echo "SUCCESS: All experiments completed successfully!"
    exit 0
fi

