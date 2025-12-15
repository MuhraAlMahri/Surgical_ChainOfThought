#!/bin/bash
#SBATCH --job-name=exp4_kvasir_s2
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp4_s2_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp4_s2_%j.err"

echo "=========================================="
echo "EXPERIMENT 4: Curriculum Learning Stage 2 (Kvasir)"
echo "CORRECTED SPLIT (Image-Level, 70/15/15)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
# Set Hugging Face cache to /tmp to avoid disk quota issues
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets"
PREV_CHECKPOINT="${BASE_DIR}/corrected 1-5 experiments/models/exp4_curriculum/stage1"
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/models/exp4_curriculum/stage2"
TRAIN_FILE="${DATASET_DIR}/kvasir_stage_splits_stage2/train.json"
VAL_FILE="${DATASET_DIR}/kvasir_stage_splits_stage2/val.json"

echo "Training Stage 2 (Findings Identification)"
echo "Train: $TRAIN_FILE"
echo "Val: $VAL_FILE"
echo "Continuing from: $PREV_CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if previous checkpoint exists
if [ ! -d "$PREV_CHECKPOINT" ]; then
    echo "ERROR: Previous checkpoint not found: $PREV_CHECKPOINT"
    echo "Please run Stage 1 training first!"
    exit 1
fi

nvidia-smi
echo ""

python3 ${BASE_DIR}/training/train_qwen_lora.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --prev_checkpoint "$PREV_CHECKPOINT" \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --num_train_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-6 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --dataset_type "reordered" \
    --max_length 512

echo ""
echo "Stage 2 completed at: $(date)"
echo "Checkpoint: $OUTPUT_DIR"
echo "Next: Submit Stage 3 training with --prev_checkpoint=$OUTPUT_DIR"

