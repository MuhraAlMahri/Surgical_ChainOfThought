#!/bin/bash
#SBATCH --job-name=eval_cot_only
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/eval_cot_only_%j.out
#SBATCH --error=slurm/logs/eval_cot_only_%j.err

echo "=========================================="
echo "Evaluate CoT Configurations Only"
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
export TOKENIZERS_PARALLELISM=false

# Use /tmp for HuggingFace cache
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create directories
mkdir -p slurm/logs
mkdir -p results/cot_evaluation

# Configuration from arguments
MODEL_TYPE="${1:-qwen3vl}"
DATASET="${2:-kvasir}"
TEST_DATA="${3}"
IMAGE_BASE_PATH="${4}"
COT_CHECKPOINT="${5}"
BASELINE_RESULTS="${6}"

# Dataset-specific defaults
if [ "$DATASET" == "kvasir" ]; then
    TEST_DATA="${TEST_DATA:-datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json}"
    IMAGE_BASE_PATH="${IMAGE_BASE_PATH:-datasets/Kvasir-VQA/raw/images}"
elif [ "$DATASET" == "endovis" ]; then
    TEST_DATA="${TEST_DATA:-datasets/EndoVis2018/raw/metadata/test_vqa_pairs.json}"
    IMAGE_BASE_PATH="${IMAGE_BASE_PATH:-datasets/EndoVis2018/raw/images}"
fi

OUTPUT_DIR="results/cot_evaluation/${MODEL_TYPE}_${DATASET}_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Model type: $MODEL_TYPE"
echo "  Dataset: $DATASET"
echo "  Test data: $TEST_DATA"
echo "  Image base path: $IMAGE_BASE_PATH"
echo "  CoT checkpoint: ${COT_CHECKPOINT:-None}"
echo "  Baseline results: ${BASELINE_RESULTS:-None}"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check files exist
if [ ! -f "$TEST_DATA" ]; then
    echo "❌ ERROR: Test data file not found: $TEST_DATA"
    exit 1
fi

# Run evaluation
echo "Starting CoT evaluation..."
python3 -u evaluate_cot_only.py \
    --model-type "$MODEL_TYPE" \
    --dataset "$DATASET" \
    --test-data "$TEST_DATA" \
    --image-base-path "$IMAGE_BASE_PATH" \
    --output "$OUTPUT_DIR" \
    ${COT_CHECKPOINT:+--cot-checkpoint "$COT_CHECKPOINT"} \
    ${BASELINE_RESULTS:+--baseline-results "$BASELINE_RESULTS"}

EVAL_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ CoT evaluation completed successfully!"
    echo "   Results: $OUTPUT_DIR"
else
    echo "❌ CoT evaluation failed with exit code: $EVAL_EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

# Clean up temporary cache
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up temporary cache..."
    rm -rf "$HF_HOME"
fi

exit $EVAL_EXIT_CODE







