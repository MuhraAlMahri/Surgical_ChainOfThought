#!/bin/bash
#SBATCH --job-name=exp1_2gpu_2.5ep
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/exp1_2gpu_2.5ep_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/exp1_2gpu_2.5ep_%j.err

echo "=========================================="
echo "EXP1: RANDOM BASELINE - 2 GPU, 2.5 EPOCHS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2"
echo ""

# Load CUDA
module load nvidia/cuda/12.0

# Activate environment
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"

# Performance optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=1
export FLASH_ATTENTION_FORCE_ENABLE=1

mkdir -p "$HF_HOME"

# Change to working directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo ""
echo "=========================================="
echo "CONFIGURATION"
echo "=========================================="
echo "Model: Qwen3-VL-8B-Instruct"
echo "GPUs: 2 (DistributedDataParallel)"
echo "Resolution: Full (2,900 tokens)"
echo "Epochs: 2.5"
echo "Memory: 80GB total (40GB × 2)"
echo "Expected speedup: 1.8x"
echo "Expected time: ~22-23 hours"
echo ""
echo "Experiment: Random Baseline"
echo "Data: Random shuffle (no clinical ordering)"
echo "=========================================="
echo ""

echo "Starting training..."
echo ""

# Run training with 2 GPUs
srun torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    exp1/train_exp1.py exp1/config_exp1_qwen3_2gpu_2.5epochs.yaml

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ EXP1 TRAINING COMPLETED SUCCESSFULLY"
    echo "Checkpoint: exp1/outputs/"
    echo "Ready for evaluation!"
else
    echo "❌ EXP1 TRAINING FAILED with exit code: $EXIT_CODE"
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

