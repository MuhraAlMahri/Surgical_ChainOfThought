#!/bin/bash
#SBATCH --job-name=exp4_curr_s3
#SBATCH --partition=cscc-gpu-p
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/logs/exp4_curr_s3.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/logs/exp4_curr_s3.err

set -euo pipefail

# Fallback if SLURM_TMPDIR is not set
SLURM_TMPDIR=${SLURM_TMPDIR:-/tmp}

export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export DEEPSPEED_KERNELS_CACHE_PATH="$SLURM_TMPDIR/deepspeed"
export HF_HOME="$SLURM_TMPDIR/hf"
export TRANSFORMERS_CACHE="$SLURM_TMPDIR/hf/transformers"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$DEEPSPEED_KERNELS_CACHE_PATH" "$HF_HOME" "$TRANSFORMERS_CACHE"

cd /l/users/muhra.almahri/Surgical_COT

python -u training/train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
  --train_json \
  	/l/users/muhra.almahri/Surgical_COT/corrected\ experiments/datasets/kvasir_qwen_stage_ordered_image_level_70_15_15/train.json \
  --val_json \
  	/l/users/muhra.almahri/Surgical_COT/corrected\ experiments/datasets/kvasir_qwen_stage_ordered_image_level_70_15_15/val.json \
  --image_dir \
  	/l/users/muhra.almahri/Surgical_COT/corrected\ experiments/datasets/kvasir_qwen_stage_ordered_image_level_70_15_15/images \
  --output_dir runs/exp4_curriculum/seed_3 \
  --stage_ordering curriculum \
  --curriculum_milestones 0.2,0.6 \
  --curriculum_stages 1|1,2|1,2,3 \
  --seed 3 \
  --image_max_size 224 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --fp16 \
  --gradient_checkpointing \
  --lr 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03


