#!/bin/bash
#SBATCH --job-name=exp1_resume
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=../../logs/exp1_resume_%j.out
#SBATCH --error=../../logs/exp1_resume_%j.err

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export JOB_TMP="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$JOB_TMP/.cache/triton" "$JOB_TMP/.cache/deepspeed"
export XDG_CACHE_HOME="$JOB_TMP/.cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export DEEPSPEED_KERNELS_CACHE_PATH="$XDG_CACHE_HOME/deepspeed"

BASE="/l/users/muhra.almahri/Surgical_COT"
DATA="${BASE}/corrected experiments/datasets"
OUT="${BASE}/corrected experiments/models/exp1_interactive_run"
IMG_DIR="${BASE}/datasets/Kvasir-VQA/raw/images"
TRAIN_FILE="${DATA}/kvasir_baseline_image_level_70_15_15/train.json"
VAL_FILE="${DATA}/kvasir_baseline_image_level_70_15_15/val.json"

python3 ${BASE}/training/train_qwen_lora.py \
  --model_name "Qwen/Qwen2-VL-2B-Instruct" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  --output_dir "$OUT" \
  --dataset_type "non_reordered" \
  --image_dir "$IMG_DIR" \
  --num_train_epochs 5 \
  --batch_size 2 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.05 \
  --image_max_size 224 \
  --max_length 512 \
  --dataloader_num_workers 2 \
  --dataloader_pin_memory False \
  --evaluation_strategy steps \
  --save_strategy steps \
  --save_steps 200 \
  --logging_steps 50
