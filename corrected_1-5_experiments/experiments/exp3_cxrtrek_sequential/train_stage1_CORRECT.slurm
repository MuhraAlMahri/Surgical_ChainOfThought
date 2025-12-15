#!/bin/bash
#SBATCH --job-name=exp3_s1_retry
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp3_s1_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp3_s1_%j.err"

echo "=========================================="
echo "EXPERIMENT 3: CXRTrek Sequential Stage 1 (RETRY)"
echo "CORRECTED SPLIT (Image-Level, 70/15/15)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

# DO NOT SET CUDA_VISIBLE_DEVICES - let SLURM handle it!
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets"
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/models/exp3_cxrtrek_seq/stage1"
TRAIN_FILE="${DATASET_DIR}/kvasir_stage_splits_stage1/train.json"
VAL_FILE="${DATASET_DIR}/kvasir_stage_splits_stage1/val.json"

echo "Training Stage 1 (Initial Assessment)"
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
echo "Stage 1 completed at: $(date)"
