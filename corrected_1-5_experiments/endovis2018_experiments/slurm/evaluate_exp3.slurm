#!/bin/bash
#SBATCH --job-name=endovis_eval_exp3
#SBATCH --output=slurm/logs/eval_exp3_%j.out
#SBATCH --error=slurm/logs/eval_exp3_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Evaluate Exp3: Sequential (Separate Models per Stage)
# This requires routing Stage 1 questions to Stage 1 model, Stage 2 to Stage 2 model

set -e

echo "=========================================="
echo "Evaluating Exp3: Sequential (Separate Models)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
SCRIPTS_DIR="${BASE_DIR}/scripts"
RESULTS_DIR="${EXPERIMENTS_DIR}/results"

# Create directories
mkdir -p "${EXPERIMENTS_DIR}/slurm/logs"
mkdir -p "${RESULTS_DIR}"

# Load modules
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH
export DS_SKIP_CUDA_CHECK=1

cd "${EXPERIMENTS_DIR}"

# Test data and image paths
TEST_DATA_BASE="${BASE_DIR}/datasets/endovis2018_vqa_reordered"
TEST_DATA="${TEST_DATA_BASE}/exp3_sequential/test.jsonl"
IMAGE_ROOT="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"
OUTPUT="${RESULTS_DIR}/exp3_evaluation.json"

# Model paths
STAGE1_MODEL="${EXPERIMENTS_DIR}/models/exp3_sequential/stage1"
STAGE2_MODEL="${EXPERIMENTS_DIR}/models/exp3_sequential/stage2"

echo ""
echo "Evaluating Exp3 with stage-specific routing..."
echo "  Stage 1 Model: ${STAGE1_MODEL}"
echo "  Stage 2 Model: ${STAGE2_MODEL}"
echo "  Test Data: ${TEST_DATA}"
echo ""

# Use Exp3 evaluation script (requires all 3 stages, but Stage 3 has 0 samples)
# Use Stage 2 model as placeholder for Stage 3 (won't be used since no Stage 3 samples)
python3 "${SCRIPTS_DIR}/evaluation/evaluate_exp3.py" \
    --model_stage1 "${STAGE1_MODEL}" \
    --model_stage2 "${STAGE2_MODEL}" \
    --model_stage3 "${STAGE2_MODEL}" \
    --test_data "${TEST_DATA}" \
    --image_dir "${IMAGE_ROOT}" \
    --output "${OUTPUT}" \
    --base_model "Qwen/Qwen3-VL-8B-Instruct"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Exp3 evaluation completed: $(date)"
    echo "  Results: ${OUTPUT}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Exp3 evaluation failed: $(date) (exit code: $STATUS)"
    echo "=========================================="
    exit $STATUS
fi

