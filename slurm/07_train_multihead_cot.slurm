#!/bin/bash
#SBATCH --job-name=train_multihead_cot
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train_multihead_cot_%j.out
#SBATCH --error=slurm/logs/train_multihead_cot_%j.err

echo "=========================================="
echo "Multi-Head CoT Training (New Script)"
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

# Configuration from arguments
# Usage: sbatch slurm/07_train_multihead_cot.slurm <model_type> <dataset> <base_checkpoint> [learning_rate] [epochs] [batch_size]
MODEL_TYPE="${1:-qwen3vl}"  # qwen3vl, medgemma, llava_med
DATASET="${2:-kvasir}"      # kvasir or endovis
BASE_CHECKPOINT="${3}"      # Required: path to fine-tuned checkpoint
LEARNING_RATE="${4:-5e-5}"
EPOCHS="${5:-5}"
BATCH_SIZE="${6:-1}"
GRAD_ACCUM="${7:-16}"

# Validate required arguments
if [ -z "$BASE_CHECKPOINT" ]; then
    echo "❌ ERROR: BASE_CHECKPOINT is required"
    echo "Usage: sbatch slurm/07_train_multihead_cot.slurm <model_type> <dataset> <base_checkpoint> [learning_rate] [epochs] [batch_size]"
    exit 1
fi

# Model-specific defaults
if [ "$MODEL_TYPE" == "medgemma" ]; then
    LEARNING_RATE="${4:-5e-5}"
    EPOCHS="${5:-5}"
    BATCH_SIZE="${6:-1}"
    GRAD_ACCUM="${7:-16}"
fi

# Dataset-specific paths
if [ "$DATASET" == "kvasir" ]; then
    DATA_PATH="datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json"
    IMAGE_BASE_PATH="datasets/Kvasir-VQA/raw/images"
elif [ "$DATASET" == "endovis" ]; then
    # Try multiple possible EndoVis paths
    if [ -f "corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl" ]; then
        DATA_PATH="corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl"
    elif [ -f "datasets/EndoVis2018/raw/metadata/train_vqa_pairs.json" ]; then
        DATA_PATH="datasets/EndoVis2018/raw/metadata/train_vqa_pairs.json"
    else
        echo "❌ ERROR: EndoVis training data not found"
        echo "   Tried: corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl"
        echo "   Tried: datasets/EndoVis2018/raw/metadata/train_vqa_pairs.json"
        exit 1
    fi
    IMAGE_BASE_PATH="datasets/EndoVis2018/raw/images"
else
    echo "❌ ERROR: Unknown dataset: $DATASET"
    exit 1
fi

# Question categories file
QUESTION_CATEGORIES="results/multihead_cot/question_categories.json"

# Output directory
OUTPUT_DIR="results/multihead_cot/${MODEL_TYPE}_${DATASET}_cot_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Model type: $MODEL_TYPE"
echo "  Dataset: $DATASET"
echo "  Base checkpoint: $BASE_CHECKPOINT"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Data path: $DATA_PATH"
echo "  Image base path: $IMAGE_BASE_PATH"
echo "  Question categories: $QUESTION_CATEGORIES"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check files exist (allow HuggingFace model names like "Qwen/Qwen3-VL-8B-Instruct")
if [[ "$BASE_CHECKPOINT" != *"/"* ]] || [[ "$BASE_CHECKPOINT" == *"Qwen"* ]] || [[ "$BASE_CHECKPOINT" == *"google"* ]] || [[ "$BASE_CHECKPOINT" == *"microsoft"* ]]; then
    # This is a HuggingFace model name, skip file check
    echo "ℹ️  Using HuggingFace model: $BASE_CHECKPOINT"
elif [ ! -f "$BASE_CHECKPOINT" ] && [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "❌ ERROR: Base checkpoint not found: $BASE_CHECKPOINT"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$QUESTION_CATEGORIES" ]; then
    echo "⚠️  WARNING: Question categories file not found: $QUESTION_CATEGORIES"
    echo "   Run categorization first: sbatch slurm/06_categorize_questions_new.slurm"
    QUESTION_CATEGORIES=""
fi

# GPU info
nvidia-smi
echo ""

# Run training
echo "Starting multi-head CoT training..."
python3 -u train_multihead_cot.py \
    --model_type "$MODEL_TYPE" \
    --dataset "$DATASET" \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --question_categories "${QUESTION_CATEGORIES:-question_categories.json}" \
    --data_path "$DATA_PATH" \
    --image_base_path "$IMAGE_BASE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 4 \
    --lora_alpha 8

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "  ✅ Checkpoint saved to: $OUTPUT_DIR"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Clean up temporary cache
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up temporary cache..."
    rm -rf "$HF_HOME"
fi

exit $TRAIN_EXIT_CODE



