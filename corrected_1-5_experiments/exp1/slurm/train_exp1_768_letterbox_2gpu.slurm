#!/bin/bash
#SBATCH --job-name=exp1_768_2gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_768_letterbox_2gpu_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_768_letterbox_2gpu_%j.err

echo "========================================================================"
echo "EXP1: 768×768 LETTERBOX TRAINING - 2 GPU DISTRIBUTED"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2 (DistributedDataParallel)"
echo "Config: exp1/config_exp1_768_letterbox_2gpu.yaml"
echo "Expected duration: 13-15 hours (1.7-1.8x speedup vs 1 GPU)"
echo "Resolution: 768×768 with letterbox padding (no warping)"
echo "Batch per GPU: 4 samples"
echo "Total batch per step: 8 samples (4 × 2 GPUs)"
echo "Effective batch size: 64 (with grad_accum=8)"
echo "========================================================================"
echo ""

# Load CUDA
echo "Loading CUDA module..."
module load nvidia/cuda/12.0

# Activate environment
echo "Activating conda environment..."
source ~/miniconda3/bin/activate base

# Performance optimizations for Ampere+ GPUs
export NVIDIA_TF32_OVERRIDE=1
export FLASH_ATTENTION_FORCE_ENABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Environment variables
export PYTHONUNBUFFERED=1
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"

# Distributed training settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2

mkdir -p "$HF_HOME"
mkdir -p "$TRITON_CACHE_DIR"
echo "✓ Cache directories created"
echo ""

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
echo ""

# Change to working directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "========================================================================"
echo "DISTRIBUTED TRAINING CONFIGURATION"
echo "========================================================================"
echo "Strategy: PyTorch DistributedDataParallel (DDP)"
echo "Backend: NCCL (optimized for multi-GPU)"
echo "Synchronization: Automatic gradient allreduce"
echo "Expected speedup: 1.7-1.8x vs single GPU"
echo "========================================================================"
echo ""

echo "Starting distributed training..."
echo ""

# Run distributed training with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=2 \
    exp1/train_exp1.py \
    exp1/config_exp1_768_letterbox_2gpu.yaml

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY"
    echo "Training time: ~13-15 hours"
else
    echo "❌ TRAINING FAILED with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "========================================================================"

# Cleanup
echo "Cleaning up temporary files..."
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"

echo "✓ Job complete"

exit $EXIT_CODE






