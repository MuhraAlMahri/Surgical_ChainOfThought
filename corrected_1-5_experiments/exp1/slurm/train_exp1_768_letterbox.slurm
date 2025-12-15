#!/bin/bash
#SBATCH --job-name=exp1_768_letterbox
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_768_letterbox_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_768_letterbox_%j.err

echo "========================================================================"
echo "EXP1: 768×768 LETTERBOX TRAINING (Aspect-Ratio Preserving)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Config: exp1/config_exp1_768_letterbox.yaml"
echo "Expected duration: 24-26 hours"
echo "Resolution: 768×768 with letterbox padding (no warping)"
echo "========================================================================"

# Load CUDA
module load nvidia/cuda/12.0

# Activate environment
source ~/miniconda3/bin/activate base

# Performance optimizations for Ampere+ GPUs
export NVIDIA_TF32_OVERRIDE=1
export FLASH_ATTENTION_FORCE_ENABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"

# Copy model cache to /tmp for faster access
echo "Setting up model cache..."
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct"
if [ -d "$SOURCE_MODEL_CACHE" ]; then
    echo "Copying model to /tmp..."
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/"
    echo "✓ Model cache ready"
else
    echo "⚠ Model cache not found, will download on-the-fly"
fi

# Run training
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo ""
echo "Starting training..."
echo "========================================================================"

python3 exp1/train_exp1.py exp1/config_exp1_768_letterbox.yaml

echo ""
echo "========================================================================"
echo "Training completed at: $(date)"
echo "========================================================================"

# Cleanup
echo "Cleaning up temporary files..."
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"

echo "✓ Job complete"






