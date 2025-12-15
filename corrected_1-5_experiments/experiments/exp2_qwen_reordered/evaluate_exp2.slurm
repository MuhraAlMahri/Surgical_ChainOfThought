#!/bin/bash
#SBATCH --job-name=exp2_eval
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH -A cscc-users
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp2_eval_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp2_eval_%j.err"

echo "=========================================="
echo "EXPERIMENT 2: EVALUATION (Qwen Reordered)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets/kvasir_raw_6500_image_level_70_15_15"
IMAGE_DIR="${BASE_DIR}/datasets/Kvasir-VQA/raw/images"
MODEL_PATH="${BASE_DIR}/corrected 1-5 experiments/models/exp2_qwen_reordered"
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/results"
TEST_FILE="${DATASET_DIR}/kvasir_raw_qwen_reordered.json"
OUTPUT_FILE="${OUTPUT_DIR}/exp2_qwen_reordered_results.json"

echo "Model: $MODEL_PATH"
echo "Test data: $TEST_FILE"
echo "Image directory: $IMAGE_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

nvidia-smi
echo ""

python3 ${BASE_DIR}/evaluation/evaluate_vl_model.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_FILE" \
    --image_dir "$IMAGE_DIR" \
    --output "$OUTPUT_FILE" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128

echo ""
echo "Evaluation completed at: $(date)"
echo "Results saved to: $OUTPUT_FILE"





