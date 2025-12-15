#!/bin/bash
#SBATCH --job-name=exp1_rand_s2
#SBATCH --partition=cscc-gpu-p
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/logs/exp1_random_s2.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/logs/exp1_random_s2.err

set -euo pipefail

module load nvidia/cuda/12.0 || true
source ~/miniconda3/bin/activate base || true

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
  --model_name "Qwen/Qwen2-VL-2B-Instruct" \
  --train_file "corrected experiments/datasets/kvasir_baseline_image_level_70_15_15/train.json" \
  --val_file "corrected experiments/datasets/kvasir_baseline_image_level_70_15_15/val.json" \
  --image_dir "datasets/Kvasir-VQA/raw/images" \
  --output_dir "runs/exp1_random/seed_2" \
  --dataset_type "non_reordered" \
  --image_max_size 224 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --evaluation_strategy steps \
  --save_strategy steps


