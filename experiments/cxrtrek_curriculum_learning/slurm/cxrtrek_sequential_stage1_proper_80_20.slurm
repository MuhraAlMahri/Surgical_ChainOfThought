#!/bin/bash
#SBATCH --job-name=cxrtrek_seq_s1_80_20
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/cxrtrek_seq_s1_proper_80_20_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning/logs/cxrtrek_seq_s1_proper_80_20_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# CXRTREK SEQUENTIAL - STAGE 1 SPECIALIST (PROPER 80/20 SPLIT)
# Trains ONE specialized model for Stage 1 ONLY (Initial Assessment)
# Independent training from base model (NO prev_checkpoint)
# Image-based split: Train and test sets have NO overlapping images

echo "=========================================="
echo "CXRTREK SEQUENTIAL - STAGE 1 SPECIALIST"
echo "PROPER 80/20 IMAGE-BASED SPLIT (ZERO OVERLAP)"
echo "Starting at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
module load nvidia/cuda/12.0

# Activate environment
source ~/miniconda3/bin/activate base

# Install/upgrade required packages
pip install -q --upgrade pyarrow datasets transformers peft accelerate

# Paths
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
OUTPUT_DIR="${BASE_DIR}/checkpoints_proper_80_20/cxrtrek_sequential"

mkdir -p $OUTPUT_DIR
mkdir -p ${BASE_DIR}/logs

echo ""
echo "Training Stage 1 Specialist Model..."
echo "Train data: ${BASE_DIR}/data/proper_splits_80_20/train_stage1.json"
echo "Val data: ${BASE_DIR}/data/proper_splits_80_20/test_stage1.json"
echo "Output: ${OUTPUT_DIR}/stage1_best/"
echo ""

# Train Stage 1 specialist model (NO prev_checkpoint - independent training)
python3 ${BASE_DIR}/scripts/train_progressive_stage.py \
    --stage 1 \
    --train_data_path ${BASE_DIR}/data/proper_splits_80_20/train_stage1.json \
    --val_data_path ${BASE_DIR}/data/proper_splits_80_20/test_stage1.json \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --device cuda

echo ""
echo "=========================================="
echo "CXRTREK Sequential Stage 1 completed at: $(date)"
echo "Model saved to: ${OUTPUT_DIR}/stage1_best/"
echo "=========================================="

