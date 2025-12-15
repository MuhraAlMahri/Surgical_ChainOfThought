#!/bin/bash
#SBATCH --job-name=cache_v2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/cache_v2_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/cache_v2_%j.err

echo "=========================================="
echo "VISION EMBEDDING CACHING V2 (ROBUST)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Activate environment
echo "Activating conda environment..."
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/.cache/huggingface"

echo ""
echo "=========================================="
echo "CACHING CONFIGURATION V2"
echo "=========================================="
echo "Model: Qwen/Qwen3-VL-8B-Instruct"
echo "Train samples: 41,079"
echo "Val samples: 8,786"
echo "Cache: exp1/vision_cache_v2/"
echo ""
echo "Features:"
echo "  - Stable SHA1 cache keys"
echo "  - Atomic writes (no partial files)"
echo "  - Robust exists checks"
echo "  - Handles Qwen3-VL text requirement"
echo "=========================================="
echo ""

# Change to working directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

# Clear old incomplete cache
echo "Cleaning old incomplete cache..."
rm -rf exp1/vision_cache_v2
mkdir -p exp1/vision_cache_v2
echo "✅ Clean cache directory created"
echo ""

# Run caching with V2 script
python3 exp1/cache_vision_embeddings_v2.py \
    --model_name "Qwen/Qwen3-VL-8B-Instruct" \
    --train_jsonl "datasets/kvasir_ULTRA_CONDENSED/train_CATEGORY_BASED.jsonl" \
    --val_jsonl "datasets/kvasir_ULTRA_CONDENSED/val_CATEGORY_BASED.jsonl" \
    --image_root "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images" \
    --cache_dir "exp1/vision_cache_v2" \
    --resolution_id "full_qwen3vl"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ CACHING V2 COMPLETED"
    echo ""
    echo "Verify cache:"
    echo "  Train files: $(find exp1/vision_cache_v2/train -type f -name '*.pt' 2>/dev/null | wc -l)"
    echo "  Val files:   $(find exp1/vision_cache_v2/val -type f -name '*.pt' 2>/dev/null | wc -l)"
    echo "  Cache size:  $(du -sh exp1/vision_cache_v2 2>/dev/null | cut -f1)"
    echo ""
    echo "Next: Update config to use vision_cache_v2"
    echo "  Then: sbatch exp1/slurm/train_exp1_category_based.slurm"
else
    echo "❌ CACHING V2 FAILED with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE






