#!/bin/bash
#SBATCH --job-name=train_unified_cot
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train_unified_%j.out
#SBATCH --error=slurm/logs/train_unified_%j.err

echo "=========================================="
echo "Unified Multi-Head Temporal CoT Training"
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

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create directories
mkdir -p slurm/logs
mkdir -p checkpoints

# Configuration
DATASET="${1:-kvasir}"  # kvasir or endovis
BASE_MODEL="${2:-Qwen/Qwen2-VL-2B-Instruct}"
NUM_EPOCHS="${3:-10}"
BATCH_SIZE="${4:-4}"
LEARNING_RATE="${5:-2e-5}"
GRAD_ACCUM="${6:-4}"

# Dataset-specific paths
if [ "$DATASET" == "kvasir" ]; then
    TRAIN_DATA="data/categorized/train_categorized.json"
    VAL_DATA="data/categorized/val_categorized.json"  # Optional - will skip if not found
    IMAGE_BASE_PATH="datasets/Kvasir-VQA/raw/images"
    TEMPORAL_DATA=""
elif [ "$DATASET" == "endovis" ]; then
    TRAIN_DATA="data/categorized/train_categorized.json"
    VAL_DATA="data/categorized/val_categorized.json"  # Optional - will skip if not found
    IMAGE_BASE_PATH="datasets/EndoVis2018/data/images"
    TEMPORAL_DATA="data/temporal/temporal_structure.json"
else
    echo "❌ ERROR: Unknown dataset: $DATASET"
    exit 1
fi

OUTPUT_DIR="checkpoints/${DATASET}_unified_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Base model: $BASE_MODEL"
echo "  Training mode: unified"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  Image base path: $IMAGE_BASE_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check files exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ ERROR: Training file not found: $TRAIN_DATA"
    echo "   Run question categorization first: sbatch slurm/01_categorize_questions.slurm"
    exit 1
fi

if [ "$DATASET" == "endovis" ] && [ ! -z "$TEMPORAL_DATA" ] && [ ! -f "$TEMPORAL_DATA" ]; then
    echo "⚠️  WARNING: Temporal data file not found: $TEMPORAL_DATA"
    echo "   Continuing without temporal structure..."
    TEMPORAL_DATA=""
fi

# GPU info
nvidia-smi
echo ""

# Check if validation file exists, make it optional
VAL_ARG=""
if [ -f "$VAL_DATA" ]; then
    VAL_ARG="--val-data $VAL_DATA"
    echo "  Validation data: $VAL_DATA"
else
    echo "  ⚠️  Validation file not found: $VAL_DATA (continuing without validation)"
fi

# Run training
echo "Starting unified multi-head training..."
python3 train_multihead_temporal_cot.py \
    --base-model "$BASE_MODEL" \
    --train-data "$TRAIN_DATA" \
    $VAL_ARG \
    --image-base-path "$IMAGE_BASE_PATH" \
    --dataset "$DATASET" \
    --training-mode unified \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --use-temporal \
    --output-dir "$OUTPUT_DIR" \
    ${TEMPORAL_DATA:+--temporal-data "$TEMPORAL_DATA"}

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE

