#!/bin/bash
#SBATCH --job-name=qlora_mega_4gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_job_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/mega_job_%j.err

# ============================================================================
# MEGA JOB: 4 GPUs Running in Parallel
# ============================================================================
# GPU 0: Exp1 - Evaluation (Random Baseline)
# GPU 1: Exp2 - Evaluation (Qwen Reordered)
# GPU 2: Exp3-S2 - Training (CXRTrek Stage 2)
# GPU 3: Exp4-S2 - Training (Curriculum Stage 2)
# ============================================================================
# NOTE: This job runs AFTER Exp3-S1 (156143_2) and Exp4-S1 (156143_3) complete
# ============================================================================

echo "=========================================="
echo "MEGA JOB - 4 GPUs Parallel"
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
SCRIPTS_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/scripts/evaluation"
IMAGE_ROOT="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
RESULTS_DIR="${BASE_DIR}/results"

# Create results directory
mkdir -p ${RESULTS_DIR}

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
    eval ${COMMAND} > ${BASE_DIR}/slurm/logs/mega_gpu${GPU_ID}_${SLURM_JOB_ID}.out 2> ${BASE_DIR}/slurm/logs/mega_gpu${GPU_ID}_${SLURM_JOB_ID}.err
    
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
# GPU 0: Exp1 Evaluation
# ============================================================================
EXP1_MODEL="${BASE_DIR}/models/exp1_random"
EXP1_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl"
EXP1_OUTPUT="${RESULTS_DIR}/exp1_evaluation.json"

# Convert JSONL to JSON for evaluation script
EXP1_TEST_JSON="/tmp/exp1_test_${SLURM_JOB_ID}.json"
if [ -f "${EXP1_TEST_DATA}" ]; then
    echo "Converting Exp1 test data from JSONL to JSON..."
    python3 -c "
import json
with open('${EXP1_TEST_DATA}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${EXP1_TEST_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"
fi

EXP1_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp1.py \
    --model_path ${EXP1_MODEL} \
    --test_data ${EXP1_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP1_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# GPU 1: Exp2 Evaluation
# ============================================================================
EXP2_MODEL="${BASE_DIR}/models/exp2_qwen_reordered"
EXP2_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp2_qwen_reordered/test.jsonl"
EXP2_OUTPUT="${RESULTS_DIR}/exp2_evaluation.json"

# Convert JSONL to JSON for evaluation script
EXP2_TEST_JSON="/tmp/exp2_test_${SLURM_JOB_ID}.json"
if [ -f "${EXP2_TEST_DATA}" ]; then
    echo "Converting Exp2 test data from JSONL to JSON..."
    python3 -c "
import json
with open('${EXP2_TEST_DATA}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${EXP2_TEST_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"
fi

EXP2_CMD="python3 ${SCRIPTS_DIR}/evaluate_exp2.py \
    --model_path ${EXP2_MODEL} \
    --test_data ${EXP2_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP2_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct"

# ============================================================================
# GPU 2: Exp3-S2 Training (independent, but waits for Exp3-S1 to finish)
# ============================================================================
EXP3_S2_CONFIG="${BASE_DIR}/configs/exp3_stage2.yaml"
EXP3_S1_CHECKPOINT="${BASE_DIR}/models/exp3_cxrtrek/stage1"
# Note: Exp3 stages are independent, but we wait for Exp3-S1 job to complete
if [ -d "${EXP3_S1_CHECKPOINT}" ]; then
    echo "Verified: Exp3-S1 checkpoint exists at ${EXP3_S1_CHECKPOINT}"
else
    echo "Note: Exp3-S1 checkpoint not found (may still be training or failed)"
    echo "Proceeding with Exp3-S2 (independent model)..."
fi
EXP3_S2_CMD="python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP3_S2_CONFIG}"

# ============================================================================
# GPU 3: Exp4-S2 Training (depends on Exp4-S1 checkpoint)
# ============================================================================
EXP4_S2_CONFIG="${BASE_DIR}/configs/exp4_stage2.yaml"
EXP4_S1_CHECKPOINT="${BASE_DIR}/models/exp4_curriculum/stage1"
if [ ! -d "${EXP4_S1_CHECKPOINT}" ]; then
    echo "ERROR: Exp4-S1 checkpoint not found at ${EXP4_S1_CHECKPOINT}"
    echo "This job requires Exp4-S1 (156143_3) to complete first!"
    exit 1
fi
echo "Verified: Exp4-S1 checkpoint exists at ${EXP4_S1_CHECKPOINT}"
EXP4_S2_CMD="python3 ${BASE_DIR}/train_qlora_qwen3vl.py ${EXP4_S2_CONFIG}"

# ============================================================================
# Run all 4 experiments in parallel
# ============================================================================
echo "Starting all 4 experiments in parallel..."
echo "=========================================="

run_experiment 0 "Exp1 - Evaluation" "${EXP1_CMD}" &
PID0=$!

run_experiment 1 "Exp2 - Evaluation" "${EXP2_CMD}" &
PID1=$!

run_experiment 2 "Exp3-S2 - Training" "${EXP3_S2_CMD}" &
PID2=$!

run_experiment 3 "Exp4-S2 - Training" "${EXP4_S2_CMD}" &
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

# Cleanup temporary files
rm -f ${EXP1_TEST_JSON} ${EXP2_TEST_JSON}

# Report final status
echo ""
echo "=========================================="
echo "MEGA JOB - ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Exp1 Evaluation (GPU 0): Exit code ${EXIT0}"
echo "Exp2 Evaluation (GPU 1): Exit code ${EXIT1}"
echo "Exp3-S2 Training (GPU 2): Exit code ${EXIT2}"
echo "Exp4-S2 Training (GPU 3): Exit code ${EXIT3}"
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

