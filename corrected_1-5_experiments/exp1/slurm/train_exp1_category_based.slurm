#!/bin/bash
#SBATCH --job-name=exp1_cat_train
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_category_based_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_category_based_%j.err

echo "=========================================="
echo "EXP1 - ULTRA-CONDENSED INSTRUCTION TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Load CUDA
echo "Loading CUDA module..."
module load nvidia/cuda/12.0

# Activate environment
echo "Activating conda environment..."
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=1
export FLASH_ATTENTION_FORCE_ENABLE=1  # Enable FlashAttention if available
mkdir -p "$HF_HOME"

echo "Cache directories created:"
echo "  HF_HOME: $HF_HOME"
echo "  TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo ""

# Copy model cache to /tmp for faster access
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct"
if [ -d "$SOURCE_MODEL_CACHE" ]; then
    echo "Copying model cache to /tmp..."
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/"
    echo "Model cache copied."
else
    echo "WARNING: Source model cache not found. Will download on first use."
fi
echo ""

# Change to working directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "=========================================="
echo "TRAINING CONFIGURATION"
echo "=========================================="
echo "Config: exp1/config_exp1_category_based.yaml"
echo "Dataset: ULTRA_CONDENSED (advisor-approved)"
echo "Format: General instruction + Minimal templates"
echo "Train samples: 41,079"
echo "Val samples: 8,786"
echo "Tokens per sample: ~241 (instruction) + ~1200 (image)"
echo "GPU Memory estimate: ~25-28GB"
echo ""
echo "Starting training..."
echo "=========================================="
echo ""

# Run training
python3 exp1/train_exp1.py exp1/config_exp1_category_based.yaml

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY"
else
    echo "❌ TRAINING FAILED with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

# Cleanup
echo ""
echo "Cleaning up temporary files..."
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"
echo "Cleanup complete."

exit $EXIT_CODE


