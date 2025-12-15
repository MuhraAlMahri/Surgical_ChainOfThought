#!/bin/bash
#SBATCH --job-name=curriculum_proper_s3
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/curriculum_proper_stage3_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/curriculum_proper_stage3_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# PROPER CURRICULUM LEARNING - STAGE 3 (Zero Leakage)
# Continues from Stage 2 checkpoint
# Image-based split: Train and test sets have NO overlapping images

echo "=========================================="
echo "CURRICULUM LEARNING - STAGE 3 (PROPER SPLIT)"
echo "Starting at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Activate environment
source ~/.bashrc

# Paths
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning"
TRAIN_DATA="${BASE_DIR}/data/proper_splits/train_stage3.json"
VAL_DATA="${BASE_DIR}/data/proper_splits/test_stage3.json"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
OUTPUT_DIR="${BASE_DIR}/checkpoints/PROPER_SPLIT/curriculum_learning"
PREV_CHECKPOINT="${OUTPUT_DIR}/stage2_best"

echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "Image dir: $IMAGE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Previous checkpoint: $PREV_CHECKPOINT"

# Train Stage 3
python3 ${BASE_DIR}/scripts/train_progressive_stage_PROPER.py \
    --stage 3 \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --images $IMAGE_DIR \
    --output $OUTPUT_DIR \
    --prev_checkpoint $PREV_CHECKPOINT \
    --epochs 3 \
    --batch_size 2 \
    --lr 1e-4 \
    --grad_accum 4 \
    --device cuda

echo "=========================================="
echo "Stage 3 completed at: $(date)"
echo "=========================================="

