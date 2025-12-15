#!/bin/bash
#SBATCH --job-name=curriculum_s1_proper
#SBATCH --output=../logs/curriculum_s1_proper_%j.out
#SBATCH --error=../logs/curriculum_s1_proper_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# CURRICULUM LEARNING - STAGE 1 (PROPER IMAGE-BASED SPLIT 80/20)
# Trains from base Qwen2-VL-2B-Instruct model

echo "=============================================="
echo "CURRICULUM LEARNING - STAGE 1 (PROPER SPLIT)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Setup environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install/upgrade required packages
pip install -q --upgrade pyarrow datasets transformers peft accelerate

# Paths - PROPER IMAGE-BASED SPLIT (80/20)
TRAIN_DATA="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/data/proper_splits_80_20/train_stage1.json"
VAL_DATA="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/data/proper_splits_80_20/test_stage1.json"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
OUTPUT_DIR="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/checkpoints_proper_80_20"

# Training parameters
STAGE=1
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=5e-6

echo ""
echo "Configuration:"
echo "  Stage: $STAGE (Initial Assessment)"
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  Images: $IMAGE_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "  Starting from: Base Qwen2-VL-2B-Instruct"
echo "  Split: PROPER 80/20 by IMAGE ID (zero overlap)"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
cd /l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning

python scripts/train_progressive_stage.py \
    --stage $STAGE \
    --train_data_path $TRAIN_DATA \
    --val_data_path $VAL_DATA \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --device cuda

echo ""
echo "=============================================="
echo "STAGE 1 TRAINING COMPLETED"
echo "End time: $(date)"
echo "=============================================="
echo "Checkpoint: $OUTPUT_DIR/stage1_best"
echo "Next: Submit Stage 2 training with this checkpoint"
echo "=============================================="



