#!/bin/bash
#SBATCH --job-name=categorize_questions_new
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/categorize_questions_new_%j.out
#SBATCH --error=slurm/logs/categorize_questions_new_%j.err

echo "=========================================="
echo "Question Categorization (New Script)"
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

# Create output directory
mkdir -p slurm/logs
mkdir -p results/multihead_cot

# Configuration
KVASIR_PATH="${1:-datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json}"
ENDOVIS_PATH="${2:-datasets/EndoVis2018/raw/metadata/vqa_pairs.json}"
OUTPUT_FILE="${3:-results/multihead_cot/question_categories.json}"
MODEL_NAME="${4:-Qwen/Qwen2.5-7B-Instruct}"

echo "Configuration:"
echo "  Kvasir path: $KVASIR_PATH"
echo "  EndoVis path: $ENDOVIS_PATH"
echo "  Output file: $OUTPUT_FILE"
echo "  Model: $MODEL_NAME"
echo "  Cache location: $HF_HOME (temporary)"
echo ""

# Check input files exist
if [ ! -f "$KVASIR_PATH" ]; then
    echo "⚠️  WARNING: Kvasir file not found: $KVASIR_PATH"
    KVASIR_ARG=""
else
    KVASIR_ARG="--kvasir_path $KVASIR_PATH"
fi

if [ ! -f "$ENDOVIS_PATH" ]; then
    echo "⚠️  WARNING: EndoVis file not found: $ENDOVIS_PATH"
    ENDOVIS_ARG=""
else
    ENDOVIS_ARG="--endovis_path $ENDOVIS_PATH"
fi

if [ -z "$KVASIR_ARG" ] && [ -z "$ENDOVIS_ARG" ]; then
    echo "❌ ERROR: At least one dataset file must be provided"
    exit 1
fi

# GPU info
nvidia-smi
echo ""

# Run categorization
echo "Starting question categorization..."
python3 -u categorize_questions.py \
    $KVASIR_ARG \
    $ENDOVIS_ARG \
    --output "$OUTPUT_FILE" \
    --model "$MODEL_NAME"

CATEG_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $CATEG_EXIT_CODE -eq 0 ]; then
    echo "Categorization completed successfully!"
    
    # Verify output file was created
    if [ -f "$OUTPUT_FILE" ]; then
        SIZE=$(du -sh "$OUTPUT_FILE" 2>/dev/null | awk '{print $1}')
        echo "  ✅ Output file created: $OUTPUT_FILE ($SIZE)"
    else
        echo "  ⚠️  WARNING: Output file not found!"
    fi
else
    echo "Categorization failed with exit code: $CATEG_EXIT_CODE"
fi

echo "End time: $(date)"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Clean up temporary cache
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up temporary cache..."
    rm -rf "$HF_HOME"
fi

exit $CATEG_EXIT_CODE













