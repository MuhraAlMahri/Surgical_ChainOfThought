#!/bin/bash
#SBATCH --job-name=cxrtrek_proper_s1
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/cxrtrek_proper_stage1_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/cxrtrek_proper_stage1_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# PROPER CXRTREK SEQUENTIAL - STAGE 1 SPECIALIST (Zero Leakage)
# Trains one specialized model for Initial Assessment
# Image-based split: Train and test sets have NO overlapping images

echo "=========================================="
echo "CXRTREK SEQUENTIAL - STAGE 1 SPECIALIST (PROPER SPLIT)"
echo "Starting at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Activate environment
source ~/.bashrc

# Paths
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
OUTPUT_DIR="${BASE_DIR}/checkpoints/PROPER_SPLIT/cxrtrek_sequential"

mkdir -p $OUTPUT_DIR
mkdir -p ${BASE_DIR}/logs

# Train Stage 1 specialist model
python3 ${BASE_DIR}/scripts/train_progressive_stage_PROPER.py \
    --stage 1 \
    --train_data ${BASE_DIR}/data/proper_splits/train_stage1.json \
    --val_data ${BASE_DIR}/data/proper_splits/test_stage1.json \
    --images $IMAGE_DIR \
    --output $OUTPUT_DIR \
    --epochs 3 \
    --batch_size 2 \
    --lr 1e-4 \
    --grad_accum 4 \
    --device cuda

echo "=========================================="
echo "CXRTREK Stage 1 specialist completed at: $(date)"
echo "=========================================="

