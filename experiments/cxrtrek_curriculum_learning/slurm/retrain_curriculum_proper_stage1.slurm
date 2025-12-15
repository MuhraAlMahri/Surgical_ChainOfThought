#!/bin/bash
#SBATCH --job-name=curriculum_proper_s1
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/curriculum_proper_stage1_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/curriculum_proper_stage1_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# PROPER CURRICULUM LEARNING - STAGE 1 (Zero Leakage)
# Image-based split: Train and test sets have NO overlapping images

echo "=========================================="
echo "CURRICULUM LEARNING - STAGE 1 (PROPER SPLIT)"
echo "Starting at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Activate environment
source ~/.bashrc

# Paths
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning"
TRAIN_DATA="${BASE_DIR}/data/proper_splits/train_stage1.json"
VAL_DATA="${BASE_DIR}/data/proper_splits/test_stage1.json"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
OUTPUT_DIR="${BASE_DIR}/checkpoints/PROPER_SPLIT/curriculum_learning"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p ${BASE_DIR}/logs

echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "Image dir: $IMAGE_DIR"
echo "Output dir: $OUTPUT_DIR"

# Train Stage 1
python3 ${BASE_DIR}/scripts/train_progressive_stage_PROPER.py \
    --stage 1 \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --images $IMAGE_DIR \
    --output $OUTPUT_DIR \
    --epochs 3 \
    --batch_size 2 \
    --lr 1e-4 \
    --grad_accum 4 \
    --device cuda

echo "=========================================="
echo "Stage 1 completed at: $(date)"
echo "=========================================="

