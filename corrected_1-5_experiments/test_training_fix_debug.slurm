#!/bin/bash
#SBATCH --job-name=test_debug
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_debug_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_debug_%j.err"

echo "=========================================="
echo "DEBUG TEST: Find gradient issue"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets/kvasir_raw_6500_image_level_70_15_15"
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/test_debug_output"
TRAIN_FILE="${DATASET_DIR}/train.json"
VAL_FILE="${DATASET_DIR}/val.json"

echo "Using DEBUG version: train_qwen_lora_debug.py"
echo "Train: $TRAIN_FILE"
echo "Val: $VAL_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

nvidia-smi
echo ""

echo "Running with debug logging enabled..."
echo ""

# Run training with DEBUG script for 20 steps
# ONLY use arguments that are in the custom arg parser
python3 ${BASE_DIR}/training/train_qwen_lora_debug.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --num_train_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --dataset_type "reordered" \
    --max_length 512 \
    --max_steps 20 \
    --evaluation_strategy "no"

echo ""
echo "Test completed at: $(date)"
echo "=========================================="
