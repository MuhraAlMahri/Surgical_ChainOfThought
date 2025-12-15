#!/bin/bash
#SBATCH --job-name=cxrtrek_proper_all
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/cxrtrek_proper_all_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/cxrtrek_proper_all_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# PROPER CXRTREK SEQUENTIAL - ALL 3 STAGES (Zero Leakage)
# Trains 3 separate specialized models
# Image-based split: Train and test sets have NO overlapping images

echo "=========================================="
echo "CXRTREK SEQUENTIAL - ALL STAGES (PROPER SPLIT)"
echo "Starting at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Activate environment
source ~/.bashrc
conda activate qwen2vl

# Paths
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/kvasir-vqa/images"
OUTPUT_DIR="${BASE_DIR}/checkpoints/PROPER_SPLIT/cxrtrek_sequential"

mkdir -p $OUTPUT_DIR
mkdir -p ${BASE_DIR}/logs

# Train Stage 1 model
echo ""
echo "=========================================="
echo "Training Stage 1 Specialist Model"
echo "=========================================="
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

echo "Stage 1 model completed!"

# Train Stage 2 model (independent, from scratch)
echo ""
echo "=========================================="
echo "Training Stage 2 Specialist Model"
echo "=========================================="
python3 ${BASE_DIR}/scripts/train_progressive_stage_PROPER.py \
    --stage 2 \
    --train_data ${BASE_DIR}/data/proper_splits/train_stage2.json \
    --val_data ${BASE_DIR}/data/proper_splits/test_stage2.json \
    --images $IMAGE_DIR \
    --output $OUTPUT_DIR \
    --epochs 3 \
    --batch_size 2 \
    --lr 1e-4 \
    --grad_accum 4 \
    --device cuda

echo "Stage 2 model completed!"

# Train Stage 3 model (independent, from scratch)
echo ""
echo "=========================================="
echo "Training Stage 3 Specialist Model"
echo "=========================================="
python3 ${BASE_DIR}/scripts/train_progressive_stage_PROPER.py \
    --stage 3 \
    --train_data ${BASE_DIR}/data/proper_splits/train_stage3.json \
    --val_data ${BASE_DIR}/data/proper_splits/test_stage3.json \
    --images $IMAGE_DIR \
    --output $OUTPUT_DIR \
    --epochs 3 \
    --batch_size 2 \
    --lr 1e-4 \
    --grad_accum 4 \
    --device cuda

echo "Stage 3 model completed!"

echo ""
echo "=========================================="
echo "ALL 3 CXRTREK MODELS COMPLETED!"
echo "Completed at: $(date)"
echo "=========================================="
echo ""
echo "Models saved in: $OUTPUT_DIR"
echo "  - stage1_best/ (Initial Assessment specialist)"
echo "  - stage2_best/ (Findings specialist)"
echo "  - stage3_best/ (Clinical Context specialist)"

