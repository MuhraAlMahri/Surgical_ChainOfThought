#!/bin/bash
#SBATCH --job-name=categorize_questions_v2
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/categorize_questions_v2_%j.out
#SBATCH --error=slurm/logs/categorize_questions_v2_%j.err

echo "=========================================="
echo "Question Categorization Job (v2 - Fixed)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load modules
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

# Environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Use /tmp for HuggingFace cache to avoid disk quota issues
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create output directory with write permissions check
OUTPUT_DIR="data/categorized"
mkdir -p "$OUTPUT_DIR"

# Test write permissions
if [ ! -w "$OUTPUT_DIR" ]; then
    echo "❌ ERROR: Cannot write to $OUTPUT_DIR"
    exit 1
fi

# Dataset configuration
DATASET="${1:-kvasir}"  # kvasir or endovis
TRAIN_FILE="${2:-datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json}"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Use /tmp for category cache too
CACHE_FILE="/tmp/question_category_cache_$SLURM_JOB_ID.json"

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Input file: $TRAIN_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model: $MODEL_NAME"
echo "  Cache location: $HF_HOME (temporary)"
echo "  Category cache: $CACHE_FILE (temporary)"
echo ""

# Check input file exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ ERROR: Input file not found: $TRAIN_FILE"
    exit 1
fi

# Check disk space (without tail to avoid SLURM rejection)
echo "Disk space check:"
df -h "$OUTPUT_DIR" | awk 'NR>1 {print}'
echo ""

# GPU info
nvidia-smi
echo ""

# Run categorization (cache file will be created in output dir)
echo "Starting question categorization..."
python3 -u data/question_categorizer.py \
    --input "$TRAIN_FILE" \
    --output "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --model "$MODEL_NAME" > "$OUTPUT_DIR/categorization.log" 2>&1

CATEG_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $CATEG_EXIT_CODE -eq 0 ]; then
    echo "Categorization completed successfully!"
    
    # Verify output files were created
    if [ -f "$OUTPUT_DIR/train_categorized.json" ]; then
        COUNT=$(wc -l < "$OUTPUT_DIR/train_categorized.json" 2>/dev/null || echo "0")
        SIZE=$(du -sh "$OUTPUT_DIR/train_categorized.json" 2>/dev/null | awk '{print $1}')
        echo "  ✅ Output file created: $COUNT samples, $SIZE"
    else
        echo "  ⚠️  WARNING: Output file not found!"
    fi
else
    echo "Categorization failed with exit code: $CATEG_EXIT_CODE"
    echo "Check logs: $OUTPUT_DIR/categorization.log"
fi

echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Clean up temporary cache
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up temporary cache..."
    rm -rf "$HF_HOME"
fi

exit $CATEG_EXIT_CODE

