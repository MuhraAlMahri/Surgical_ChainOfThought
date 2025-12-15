#!/bin/bash
#SBATCH --job-name=cache_robust
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/cache_robust_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/cache_robust_%j.err

echo "=========================================="
echo "ROBUST VISION CACHING (Manifest-Backed)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Load CUDA
module load nvidia/cuda/12.0

# Activate environment
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1
export QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct"
export IMG_RES=1036
export VISION_CACHE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/vision_cache_v2"
export HF_HOME="$HOME/.cache/huggingface"

# Performance flags
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Configuration:"
echo "  Model: $QWEN_MODEL"
echo "  Resolution: ${IMG_RES}x${IMG_RES}"
echo "  Cache: $VISION_CACHE_DIR"
echo ""

# Change to working directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

# Clean old incomplete cache
echo "Preparing fresh cache directory..."
rm -rf "$VISION_CACHE_DIR"
mkdir -p "$VISION_CACHE_DIR/train"
mkdir -p "$VISION_CACHE_DIR/val"
echo "✅ Clean cache directory created"
echo ""

# Run robust caching
python3 -u exp1/tools/cache_vision_robust.py

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ROBUST CACHING COMPLETED"
    echo ""
    echo "Verification:"
    TRAIN_COUNT=$(find "$VISION_CACHE_DIR/train" -type f -name '*.pt' 2>/dev/null | wc -l)
    VAL_COUNT=$(find "$VISION_CACHE_DIR/val" -type f -name '*.pt' 2>/dev/null | wc -l)
    CACHE_SIZE=$(du -sh "$VISION_CACHE_DIR" 2>/dev/null | cut -f1)
    
    echo "  Train files: $TRAIN_COUNT"
    echo "  Val files:   $VAL_COUNT"
    echo "  Total:       $((TRAIN_COUNT + VAL_COUNT))"
    echo "  Cache size:  $CACHE_SIZE"
    echo ""
    echo "Next: Train with cache"
    echo "  sbatch exp1/slurm/train_exp1_category_based.slurm"
else
    echo "❌ ROBUST CACHING FAILED with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE






