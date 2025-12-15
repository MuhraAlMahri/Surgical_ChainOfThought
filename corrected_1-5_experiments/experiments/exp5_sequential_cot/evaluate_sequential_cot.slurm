#!/bin/bash
#SBATCH --job-name=exp5_sequential_cot
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp5_cot_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp5_cot_%j.err"

echo "=========================================="
echo "EXPERIMENT 5: Sequential Chain-of-Thought Evaluation"
echo "Cascading Inference (Stage 1 → Stage 2 → Stage 3)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
# Set Hugging Face cache to /tmp to avoid disk quota issues
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets/kvasir_raw_6500_image_level_70_15_15"
IMAGE_DIR="${BASE_DIR}/datasets/Kvasir-VQA/raw/images"
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/results"

# Model path for Exp5
MODEL_PATH="${BASE_DIR}/corrected 1-5 experiments/models/exp5_sequential_cot"

TEST_FILE="${DATASET_DIR}/test.json"
OUTPUT_FILE="${OUTPUT_DIR}/exp5_sequential_cot_results.json"

echo "Model: $MODEL_PATH"
echo "Test data: $TEST_FILE"
echo "Image directory: $IMAGE_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "WARNING: Model path not found: $MODEL_PATH"
    echo "Please update MODEL_PATH in this script to point to a trained model."
    echo "Options: exp1_random_baseline or exp2_qwen_reordered"
    exit 1
fi

nvidia-smi
echo ""

python3 ${BASE_DIR}/corrected\ 1-5\ experiments/experiments/exp5_sequential_cot/evaluate_sequential_cot.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_FILE" \
    --image_dir "$IMAGE_DIR" \
    --output "$OUTPUT_FILE" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128

echo ""
echo "Evaluation completed at: $(date)"
echo "Results saved to: $OUTPUT_FILE"

