#!/bin/bash
#SBATCH --job-name=endovis_mega
#SBATCH --output=slurm/logs/mega_job_%j.out
#SBATCH --error=slurm/logs/mega_job_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Mega Job: Zero-Shot + Training for EndoVis2018
# Runs 4 jobs in parallel, each using 1 GPU:
#   GPU 0: Zero-shot evaluation
#   GPU 1: Training Exp1 (Random Baseline)
#   GPU 2: Training Exp2 (Qwen Reordered)
#   GPU 3: Training Exp3 (Sequential)
# IMPORTANT: All jobs run on compute nodes (MBZUAI HPC policy)

set -e

echo "=========================================="
echo "MEGA JOB: EndoVis2018 Experiments"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 4 (parallel execution)"
echo "Started: $(date)"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
SCRIPT_DIR="${BASE_DIR}/scripts/evaluation"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
QLORA_DIR="${BASE_DIR}/qlora_experiments"
RESULTS_DIR="${EXPERIMENTS_DIR}/results"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"

# Base model
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Test data paths
TEST_DATA_JSONL="${BASE_DIR}/datasets/endovis2018_vqa/test.jsonl"
TEST_DATA_JSON="/tmp/zeroshot_endovis_test_${SLURM_JOB_ID}.json"

# Create directories
mkdir -p "${RESULTS_DIR}"
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

# ============================================================================
# GPU 0: Zero-Shot Evaluation
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu0
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 0: Zero-Shot Evaluation"
    echo "=========================================="
    echo "Started: $(date)"
    
    # Convert JSONL to JSON
    echo "Converting test data from JSONL to JSON..."
    python3 << PYTHON_EOF
import json
import sys

jsonl_path = "${TEST_DATA_JSONL}"
json_path = "${TEST_DATA_JSON}"

try:
    with open(jsonl_path, 'r') as f_in, open(json_path, 'w') as f_out:
        data = [json.loads(line) for line in f_in]
        json.dump(data, f_out, indent=2)
    print(f"Converted {len(data)} samples from JSONL to JSON")
except Exception as e:
    print(f"Error converting data: {e}")
    sys.exit(1)
PYTHON_EOF
    
    # Run zero-shot evaluation
    python3 "${SCRIPT_DIR}/evaluate_zeroshot.py" \
        --base_model "${BASE_MODEL}" \
        --test_data "${TEST_DATA_JSON}" \
        --image_dir "${IMAGE_DIR}" \
        --output "${RESULTS_DIR}/endovis2018_zeroshot.json" \
        --use_instruction
    
    echo "✓ Zero-shot evaluation completed: $(date)"
    echo "  Results: ${RESULTS_DIR}/endovis2018_zeroshot.json"
    
    # Clean up
    rm -f "${TEST_DATA_JSON}"
) 2>&1 | sed 's/^/[GPU 0] /' &
ZEROSHOT_PID=$!

# ============================================================================
# GPU 1: Training Exp1
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=1
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu1
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 1: Training Exp1 (Random Baseline)"
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
    
    echo "✓ Training completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp1_random"
) 2>&1 | sed 's/^/[GPU 1] /' &
TRAIN_PID=$!

# ============================================================================
# GPU 2: Training Exp2 (Qwen Reordered)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=2
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
    mkdir -p $HF_HOME
    
    echo ""
    echo "=========================================="
    echo "GPU 2: Training Exp2 (Qwen Reordered)"
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
    
    echo "✓ Training Exp2 completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp2_qwen_reordered"
) 2>&1 | sed 's/^/[GPU 2] /' &
TRAIN_EXP2_PID=$!

# ============================================================================
# GPU 3: Training Exp3 (Sequential)
# ============================================================================
(
    export CUDA_VISIBLE_DEVICES=3
    export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
    export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu3
    mkdir -p $HF_HOME $TRITON_CACHE_DIR
    
    echo ""
    echo "=========================================="
    echo "GPU 3: Training Exp3 (Sequential)"
    echo "=========================================="
    echo "Started: $(date)"
    
    CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp3_sequential.yaml"
    
    # Verify config exists
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "ERROR: Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Run training
    python3 "${QLORA_DIR}/train_qlora_qwen3vl.py" \
        "${CONFIG_FILE}"
    
    echo "✓ Training Exp3 completed: $(date)"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp3_sequential"
) 2>&1 | sed 's/^/[GPU 3] /' &
TRAIN_EXP3_PID=$!

# ============================================================================
# Wait for all jobs to complete
# ============================================================================
echo ""
echo "=========================================="
echo "Waiting for all jobs to complete..."
echo "=========================================="
echo "Zero-shot PID: ${ZEROSHOT_PID}"
echo "Training Exp1 PID: ${TRAIN_PID}"
echo "Training Exp2 PID: ${TRAIN_EXP2_PID}"
echo "Training Exp3 PID: ${TRAIN_EXP3_PID}"
echo ""

wait $ZEROSHOT_PID
ZEROSHOT_EXIT=$?

wait $TRAIN_PID
TRAIN_EXIT=$?

wait $TRAIN_EXP2_PID
TRAIN_EXP2_EXIT=$?

wait $TRAIN_EXP3_PID
TRAIN_EXP3_EXIT=$?

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "MEGA JOB COMPLETED"
echo "=========================================="
echo "Finished: $(date)"
echo ""

if [ $ZEROSHOT_EXIT -eq 0 ]; then
    echo "✓ Zero-shot evaluation: SUCCESS"
    echo "  Results: ${RESULTS_DIR}/endovis2018_zeroshot.json"
else
    echo "✗ Zero-shot evaluation: FAILED (exit code: ${ZEROSHOT_EXIT})"
fi

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Training Exp1: SUCCESS"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp1_random"
else
    echo "✗ Training Exp1: FAILED (exit code: ${TRAIN_EXIT})"
fi

if [ $TRAIN_EXP2_EXIT -eq 0 ]; then
    echo "✓ Training Exp2: SUCCESS"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp2_qwen_reordered"
else
    echo "✗ Training Exp2: FAILED (exit code: ${TRAIN_EXP2_EXIT})"
fi

if [ $TRAIN_EXP3_EXIT -eq 0 ]; then
    echo "✓ Training Exp3: SUCCESS"
    echo "  Model: ${EXPERIMENTS_DIR}/models/exp3_sequential"
else
    echo "✗ Training Exp3: FAILED (exit code: ${TRAIN_EXP3_EXIT})"
fi

echo ""
echo "=========================================="

# Exit with error if any job failed
if [ $ZEROSHOT_EXIT -ne 0 ] || [ $TRAIN_EXIT -ne 0 ] || [ $TRAIN_EXP2_EXIT -ne 0 ] || [ $TRAIN_EXP3_EXIT -ne 0 ]; then
    exit 1
fi

echo "All jobs completed successfully!"

