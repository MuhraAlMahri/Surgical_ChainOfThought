#!/bin/bash
#SBATCH --job-name=exp2_qwen_2gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp2/slurm/logs/train_qwen_reordered_2gpu_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp2/slurm/logs/train_qwen_reordered_2gpu_%j.err

echo "========================================================================"
echo "EXP2: QWEN REORDERED TRAINING - 2 GPU DISTRIBUTED"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2 (DistributedDataParallel)"
echo "Model: Qwen3-VL-8B-Instruct"
echo "Strategy: Train on Qwen-reordered QA pairs (stages 1,2,3)"
echo "Instructions: ULTRA_CONDENSED"
echo "Resolution: 768×768 letterbox (no warping)"
echo "Expected duration: 13-15 hours"
echo ""
echo "Data: QA pairs reordered by Qwen into clinical stages"
echo "  Stage 1 (35%): Initial Assessment"
echo "  Stage 2 (64%): Findings Identification"
echo "  Stage 3 (0.1%): Clinical Context"
echo "========================================================================"
echo ""

# Load CUDA
module load nvidia/cuda/12.0

# Activate environment
source ~/miniconda3/bin/activate base

# Performance optimizations
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

# Copy Qwen3-VL-8B model cache
echo "Setting up model cache..."
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
if [ -d "$SOURCE_MODEL_CACHE" ]; then
    echo "Copying Qwen3-VL-8B to /tmp..."
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/"
    echo "✓ Model cache ready"
else
    echo "⚠ Model cache not found, will download (~16GB)"
fi
echo ""

# Change to working directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "Starting Qwen reordered training..."
echo ""

# Run distributed training
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=2 \
    exp2/train_exp2_qwen_reordered.py \
    exp2/config_exp2_qwen_reordered_2gpu.yaml

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ EXP2 TRAINING COMPLETED SUCCESSFULLY"
    echo "Checkpoint: exp2/outputs/"
    echo "Ready for evaluation!"
else
    echo "❌ EXP2 TRAINING FAILED with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "========================================================================"

# Cleanup
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"

exit $EXIT_CODE






