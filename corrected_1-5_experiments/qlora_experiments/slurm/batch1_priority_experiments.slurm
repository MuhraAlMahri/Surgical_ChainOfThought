#!/bin/bash
#SBATCH --job-name=qlora_batch1_4gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/batch1_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/batch1_%j.err

# ============================================================================
# BATCH 1: Priority Experiments (4 GPUs in parallel)
# ============================================================================
# GPU 0: Exp1 - Random Baseline (~20h)
# GPU 1: Exp2 - Qwen Reordered (~20h)
# GPU 2: Exp3-S1 - CXRTrek Stage 1 (~7h)
# GPU 3: Exp4-S1 - Curriculum Stage 1 (~7h)
# ============================================================================

echo "=========================================="
echo "BATCH 1 - 4 GPUs Parallel"
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
    python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${CONFIG} > ${BASE_DIR}/slurm/logs/batch1_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/batch1_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
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

# Run all 4 experiments in parallel
echo "Starting all 4 experiments in parallel..."
echo "=========================================="

run_experiment 0 "Exp1 - Random Baseline" "${BASE_DIR}/configs/exp1_random.yaml" &
PID0=$!

run_experiment 1 "Exp2 - Qwen Reordered" "${BASE_DIR}/configs/exp2_qwen_reordered.yaml" &
PID1=$!

run_experiment 2 "Exp3 - CXRTrek Stage 1" "${BASE_DIR}/configs/exp3_stage1.yaml" &
PID2=$!

run_experiment 3 "Exp4 - Curriculum Stage 1" "${BASE_DIR}/configs/exp4_stage1.yaml" &
PID3=$!

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

wait ${PID3}
EXIT3=$?

# Report final status
echo ""
echo "=========================================="
echo "BATCH 1 - ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Exp1 (GPU 0): Exit code ${EXIT0}"
echo "Exp2 (GPU 1): Exit code ${EXIT1}"
echo "Exp3-S1 (GPU 2): Exit code ${EXIT2}"
echo "Exp4-S1 (GPU 3): Exit code ${EXIT3}"
echo "End time: $(date)"
echo "=========================================="

# Exit with error if any experiment failed
if [ ${EXIT0} -ne 0 ] || [ ${EXIT1} -ne 0 ] || [ ${EXIT2} -ne 0 ] || [ ${EXIT3} -ne 0 ]; then
    echo "ERROR: One or more experiments failed!"
    exit 1
else
    echo "SUCCESS: All experiments completed successfully!"
    exit 0
fi

