#!/bin/bash
#SBATCH --job-name=qwen3_4gpu_12h
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=15:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/qwen3_4gpu_12h_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/qwen3_4gpu_12h_%j.err

echo "=========================================="
echo "QWEN3-VL 4-GPU FAST TRAINING (12-Hour Target)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4"
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
export NVIDIA_TF32_OVERRIDE=1

# DDP settings
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export OMP_NUM_THREADS=8

mkdir -p "$HF_HOME"
mkdir -p "$TRITON_CACHE_DIR"

# Copy model cache if available
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
if [ -d "$SOURCE_MODEL_CACHE" ]; then
    echo "Copying Qwen3-VL model cache..."
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/" &
else
    echo "Model will download on first use"
fi

cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "=========================================="
echo "CONFIGURATION"
echo "=========================================="
echo "Model: Qwen3-VL-8B-Instruct"
echo "GPUs: 4 (DistributedDataParallel)"
echo "Resolution: Full (2,900 tokens)"
echo "Epochs: 3 (full training)"
echo "Memory: 160GB total (40GB × 4)"
echo "Expected speedup: 3.6x"
echo "Expected time: ~14 hours (3 full epochs)"
echo "=========================================="
echo ""

# Wait for cache copy to finish
wait

# Run distributed training on 4 GPUs
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    exp1/train_exp1.py \
    exp1/config_exp1_qwen3_4gpu_12h.yaml

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED"
    echo "Final model: exp1/outputs/"
else
    echo "❌ TRAINING FAILED: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

# Cleanup
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"

exit $EXIT_CODE

