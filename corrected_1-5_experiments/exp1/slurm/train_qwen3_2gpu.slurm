#!/bin/bash
#SBATCH --job-name=qwen3_2gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/qwen3_2gpu_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/qwen3_2gpu_%j.err

echo "=========================================="
echo "QWEN3-VL 2-GPU TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2"
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

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1

# DDP settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2

mkdir -p "$HF_HOME"
mkdir -p "$TRITON_CACHE_DIR"

echo "Cache directories created"
echo ""

# Copy model cache
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
if [ -d "$SOURCE_MODEL_CACHE" ]; then
    echo "Copying Qwen3-VL model cache to /tmp..."
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/"
    echo "✅ Model cache copied"
else
    echo "⚠️  Model will download on first use (~16GB)"
fi
echo ""

# Change to working directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "=========================================="
echo "TRAINING CONFIGURATION"
echo "=========================================="
echo "Model: Qwen3-VL-8B-Instruct"
echo "GPUs: 2 (DistributedDataParallel)"
echo "Resolution: Full (~2,900 tokens)"
echo "Memory: 80GB total (40GB × 2)"
echo "Expected: ~25-32 hours with 2-GPU speedup"
echo "=========================================="
echo ""

# Run distributed training
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=2 \
    exp1/train_exp1.py \
    exp1/config_exp1_qwen3_2gpu.yaml

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
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"

exit $EXIT_CODE






