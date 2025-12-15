#! /bin/bash

#SBATCH --job-name=exp1_sanity           # Job name
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/logs/exp1_sanity_%A.txt  # Stdout
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/logs/exp1_sanity_%A.err   # Stderr
#SBATCH --nodes=1                        # Single node
#SBATCH --ntasks=1                       # Single task
#SBATCH --mem=64G                        # RAM
#SBATCH --cpus-per-task=8                # CPU cores per task
#SBATCH --gres=gpu:1                     # GPUs per node
#SBATCH -p cscc-gpu-p                    # GPU partition
#SBATCH --time=01:00:00                  # 1 hour sanity budget
#SBATCH -q cscc-gpu-qos                  # QoS

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
# Robust node-local caches to avoid quota/permission issues
export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR" "$SLURM_TMPDIR/.cache" "$SLURM_TMPDIR/triton" "$SLURM_TMPDIR/.cache/deepspeed"
export HOME="$SLURM_TMPDIR"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton"
export DEEPSPEED_KERNELS_CACHE_PATH="$SLURM_TMPDIR/.cache/deepspeed"
export HF_HOME="$SLURM_TMPDIR/.cache/hf"
export TRANSFORMERS_CACHE="$SLURM_TMPDIR/.cache/hf/transformers"
export DS_ACCELERATOR=cuda

BASE="/l/users/muhra.almahri/Surgical_COT"
DATA="${BASE}/corrected experiments/datasets"
OUT="${BASE}/corrected experiments/models/exp1_sanity"
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
  --num_train_epochs 1 \
  --max_steps 200 \
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
  --eval_accumulation_steps 1


