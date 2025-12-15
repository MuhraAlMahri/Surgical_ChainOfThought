#!/bin/bash
#SBATCH --job-name=train_cot_all_2gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train_cot_all_2gpu_%j.out
#SBATCH --error=slurm/logs/train_cot_all_2gpu_%j.err

echo "=========================================="
echo "Train All Multi-Head CoT Configurations (2 GPUs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load modules
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

# Environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use /tmp for HuggingFace cache
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Load HuggingFace token
if [ -f ~/.hf_token ]; then
    export HF_TOKEN=$(cat ~/.hf_token)
elif [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
else
    echo "⚠️  WARNING: HF_TOKEN not set. Some models may fail to load."
fi

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create directories
mkdir -p slurm/logs
mkdir -p results/multihead_cot

# GPU info
nvidia-smi
echo ""

# Run combined training
echo "Starting combined training on 2 GPUs..."
python3 -u train_multihead_cot_all_combined.py \
    --output-dir "results/multihead_cot/all_combined_$(date +%Y%m%d_%H%M%S)" \
    --gpu-0 0 \
    --gpu-1 1

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training completed with exit code: $TRAIN_EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

# Clean up temporary cache
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up temporary cache..."
    rm -rf "$HF_HOME"
fi

exit $TRAIN_EXIT_CODE





