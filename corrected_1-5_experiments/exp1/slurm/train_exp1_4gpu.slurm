#!/bin/bash
#SBATCH --job-name=exp1_4gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_4gpu_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_4gpu_%j.err

echo "=========================================="
echo "EXP1 - 4-GPU DISTRIBUTED TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs requested: 4"
echo ""

# Load CUDA
echo "Loading CUDA module..."
module load nvidia/cuda/12.0

# Activate environment
echo "Activating conda environment..."
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
echo "Config: exp1/config_exp1_4gpu.yaml"
echo "Dataset: ULTRA_CONDENSED (advisor-approved)"
echo "GPUs: 4 (DistributedDataParallel)"
echo "Train samples: 41,079"
echo "Val samples: 8,786"
echo "Per-device batch size: 1"
echo "Gradient accumulation: 4"
echo "Effective batch size: 4 × 1 × 4 = 16"
echo "Expected time: ~12-13 hours (3.5x speedup)"
echo ""
echo "Starting distributed training..."
echo "=========================================="
echo ""

# Run training with torchrun for multi-GPU
# HuggingFace Trainer automatically uses DDP when multiple GPUs are available
torchrun --nproc_per_node=4 --master_port=29500 exp1/train_exp1.py exp1/config_exp1_4gpu.yaml

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

