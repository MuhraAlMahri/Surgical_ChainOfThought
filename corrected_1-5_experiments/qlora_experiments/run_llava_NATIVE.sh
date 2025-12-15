#!/bin/bash
#SBATCH --job-name=llava_NATIVE
#SBATCH --output=logs/llava_native_%j.out
#SBATCH --error=logs/llava_native_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

echo "=========================================="
echo "LLaVA-Med Training - NATIVE PROCESSOR"
echo "Using AutoProcessor (correct image token handling)"
echo "=========================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Load modules
module purge
module load nvidia/cuda/12.0 2>/dev/null || true

# Activate environment
source ~/miniconda3/bin/activate base

# Shared cache configuration
export SHARED_CACHE="/l/users/muhra.almahri/.cache/hf_shared"
export HF_HOME=$SHARED_CACHE
export TRANSFORMERS_CACHE=$SHARED_CACHE/transformers
export HF_DATASETS_CACHE=$SHARED_CACHE/datasets
export HF_HUB_CACHE=$SHARED_CACHE
export TORCH_HOME=$SHARED_CACHE/torch

# CRITICAL: Aggressively disable DeepSpeed completely
unset DS_SKIP_CUDA_CHECK
unset DS_ACCELERATOR
unset DEEPSPEED_CONFIG_FILE
unset DEEPSPEED_AUTOTUNING
unset DEEPSPEED_MULTINODE_LAUNCHER
unset DEEPSPEED_HOSTFILE
unset DEEPSPEED_INIT_METHOD

# Set explicit environment variables to disable DeepSpeed
export ACCELERATE_USE_DEEPSPEED=false
export DEEPSPEED_DISABLED=true
export ACCELERATE_USE_CUDA=true
export ACCELERATE_USE_CPU=false
export ACCELERATE_USE_XPU=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

# CUDA settings - use benchmark mode (not deterministic) for stability
export CUDNN_DETERMINISTIC=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Enable CUDA synchronous execution for better error messages
export CUDA_LAUNCH_BLOCKING=1

# CUDA environment
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH

# Python settings
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Hugging Face authentication (if needed for gated models)
export HF_TOKEN="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"

# Dataset configuration
export DATASET_NAME=kvasir
export DATA_ROOT=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_ULTRA_CONDENSED

echo "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  Data root: $DATA_ROOT"
echo "  Cache: $SHARED_CACHE"
echo "  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "  Using: AutoProcessor (native LLaVA processor)"
echo ""

# Create directories
mkdir -p logs outputs

# Change to script directory
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments

# Run training
echo "Starting training with native AutoProcessor..."
python train_llava_NATIVE_PROCESSOR.py

echo ""
echo "Completed at $(date)"
echo "=========================================="














