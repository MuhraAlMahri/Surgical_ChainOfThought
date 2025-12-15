#!/bin/bash
#SBATCH --job-name=exp1_fixed
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=../../logs/exp1_fixed_%j.out
#SBATCH --error=../../logs/exp1_fixed_%j.err

# ============================================================================
# EXP1 WITH OOM FIX
# ============================================================================
# Fixes: smaller images (256), eval batch 1, on-the-fly processing
# ============================================================================

echo "Starting Exp1 training with OOM fixes..."
echo "Time: $(date)"

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected experiments/datasets"
OUTPUT_DIR="${BASE_DIR}/corrected experiments/models/exp1_random_baseline_fixed"
TRAIN_FILE="${DATASET_DIR}/kvasir_baseline_image_level_70_15_15/train.json"
VAL_FILE="${DATASET_DIR}/kvasir_baseline_image_level_70_15_15/val.json"

python3 ${BASE_DIR}/training/train_qwen_lora.py \
  --model_name "Qwen/Qwen2-VL-2B-Instruct" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_type "non_reordered" \
  \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --learning_rate 1e-4 \
  \
  --num_train_epochs 3 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --eval_accumulation_steps 4 \
  \
  --image_max_size 224 \
  --image_dir "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images" \
  --max_length 128 \
  \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --dataloader_num_workers 2 \
  --dataloader_pin_memory False
