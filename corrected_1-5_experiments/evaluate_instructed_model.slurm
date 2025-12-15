#!/bin/bash
#SBATCH --job-name=exp1_instructed_eval
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/logs/exp1_instructed_eval_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/logs/exp1_instructed_eval_%j.err"

echo "=========================================="
echo "EXPERIMENT 1: EVALUATION (Instructed Model)"
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

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
MODEL_PATH="${BASE_DIR}/models/exp1_random_baseline_instructed/checkpoint-7704"
TEST_FILE="${BASE_DIR}/datasets/kvasir_instructed/test_instructed.json"
OUTPUT_FILE="${BASE_DIR}/results/exp1_instructed_model_results.json"

echo "Model: $MODEL_PATH"
echo "Test data: $TEST_FILE"
echo "Image directory: $IMAGE_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

nvidia-smi
echo ""

python3 ${BASE_DIR}/scripts/evaluation/evaluate_exp1.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_FILE" \
    --image_dir "$IMAGE_DIR" \
    --output "$OUTPUT_FILE" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct"

echo ""
echo "Evaluation completed at: $(date)"
echo "Results saved to: $OUTPUT_FILE"

