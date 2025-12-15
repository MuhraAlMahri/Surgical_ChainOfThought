#!/bin/bash
#SBATCH --job-name=eval_exp2
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/eval_exp2_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/eval_exp2_%j.err"

echo "=========================================="
echo "EVALUATION: Experiment 2 - Qwen-Reordered"
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
MODEL_PATH="${BASE_DIR}/corrected 1-5 experiments/models/exp2_qwen_reordered"
TEST_DATA="${BASE_DIR}/corrected 1-5 experiments/datasets/kvasir_raw_6500_image_level_70_15_15/test.json"
IMAGE_DIR="${BASE_DIR}/datasets/Kvasir-VQA/raw/images"
OUTPUT_FILE="${BASE_DIR}/corrected 1-5 experiments/results/exp2_evaluation_results.json"

echo "Model: $MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Image directory: $IMAGE_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

nvidia-smi
echo ""

python3 ${BASE_DIR}/corrected\ 1-5\ experiments/scripts/evaluation/evaluate_exp2.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_DATA" \
    --image_dir "$IMAGE_DIR" \
    --output "$OUTPUT_FILE" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct"

echo ""
echo "Evaluation completed at: $(date)"



