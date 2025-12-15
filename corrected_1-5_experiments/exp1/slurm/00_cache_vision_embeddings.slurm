#!/bin/bash
#SBATCH --job-name=cache_vision
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/cache_vision_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/cache_vision_%j.err

echo "=========================================="
echo "VISION EMBEDDING CACHING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Load CUDA
echo "Loading CUDA module..."
module load nvidia/cuda/12.0

# Activate environment
echo "Activating conda environment..."
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/.cache/huggingface"

echo ""
echo "=========================================="
echo "CACHING CONFIGURATION"
echo "=========================================="
echo "Model: Qwen/Qwen3-VL-8B-Instruct"
echo "Train samples: 41,079"
echo "Val samples: 8,786"
echo "Output: exp1/vision_cache/"
echo ""
echo "This pre-computes vision embeddings once"
echo "Training will be 2-5x faster using cached embeddings!"
echo "=========================================="
echo ""

# Change to working directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

# Run caching
python3 exp1/cache_vision_embeddings.py \
    --model_name "Qwen/Qwen3-VL-8B-Instruct" \
    --train_jsonl "datasets/kvasir_ULTRA_CONDENSED/train_CATEGORY_BASED.jsonl" \
    --val_jsonl "datasets/kvasir_ULTRA_CONDENSED/val_CATEGORY_BASED.jsonl" \
    --image_root "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images" \
    --cache_dir "exp1/vision_cache"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ CACHING COMPLETED SUCCESSFULLY"
    echo ""
    echo "Cache location: exp1/vision_cache/"
    echo "  - train/: 41,079 cached embeddings"
    echo "  - val/: 8,786 cached embeddings"
    echo ""
    echo "Next step: Run training with cached embeddings"
    echo "  sbatch exp1/slurm/train_exp1_category_based.slurm"
else
    echo "❌ CACHING FAILED with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE






