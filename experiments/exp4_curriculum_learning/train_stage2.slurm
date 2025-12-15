#!/bin/bash
#SBATCH --job-name=corr_exp4_kvasir_s2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=../../logs/exp4_s2_%j.out
#SBATCH --error=../../logs/exp4_s2_%j.err

echo "=========================================="
echo "CORRECTED EXPERIMENT 4: Curriculum Learning Stage 2 (Kvasir)"
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
PREV_CHECKPOINT="${BASE_DIR}/corrected experiments/models/exp4_curriculum/stage1"
OUTPUT_DIR="${BASE_DIR}/corrected experiments/models/exp4_curriculum/stage2"
TRAIN_FILE="${DATASET_DIR}/kvasir_reordered_image_level_70_15_15/train.json"
VAL_FILE="${DATASET_DIR}/kvasir_reordered_image_level_70_15_15/val.json"

echo "Training Stage 2 (Findings Identification)"
echo "Train: $TRAIN_FILE"
echo "Val: $VAL_FILE"
echo "Continuing from: $PREV_CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo ""

nvidia-smi
echo ""

python3 ${BASE_DIR}/training/train_qwen_lora.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --prev_checkpoint "$PREV_CHECKPOINT" \
    --model_name "Qwen/Qwen2-1.5B-Instruct" \
    --num_train_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --dataset_type "reordered" \
    --max_length 512

echo ""
echo "Stage 2 completed at: $(date)"






