#!/bin/bash
#SBATCH --job-name=corr_exp1_kvasir_random
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=../../logs/exp1_random_%j.out
#SBATCH --error=../../logs/exp1_random_%j.err

echo "=========================================="
echo "CORRECTED EXPERIMENT 1: Random Baseline (Kvasir)"
echo "IMAGE-LEVEL SPLIT (ZERO OVERLAP)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected experiments/datasets"
OUTPUT_DIR="${BASE_DIR}/corrected experiments/models/exp1_random_baseline"
TRAIN_FILE="${DATASET_DIR}/kvasir_baseline_image_level_70_15_15/train.json"
VAL_FILE="${DATASET_DIR}/kvasir_baseline_image_level_70_15_15/val.json"
TEST_FILE="${DATASET_DIR}/kvasir_baseline_image_level_70_15_15/test.json"

echo "Train: $TRAIN_FILE"
echo "Val: $VAL_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

nvidia-smi
echo ""

python3 ${BASE_DIR}/training/train_qwen_lora.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "Qwen/Qwen2-VL-2B-Instruct" \
    --num_train_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --dataset_type "non_reordered" \
    --max_length 512 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01

echo ""
echo "Training completed at: $(date)"






