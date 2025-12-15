#!/bin/bash
#SBATCH --job-name=eval_multihead_cot
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/eval_multihead_cot_%j.out
#SBATCH --error=slurm/logs/eval_multihead_cot_%j.err

echo "=========================================="
echo "Multi-Head CoT Evaluation"
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

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create directories
mkdir -p slurm/logs
mkdir -p results/multihead_cot/evaluation

# Configuration from arguments
# Usage: sbatch slurm/08_evaluate_multihead_cot.slurm <checkpoint> <model_type> <dataset> [test_data] [baseline_results]
CHECKPOINT="${1}"           # Required: path to model checkpoint
MODEL_TYPE="${2:-qwen3vl}" # qwen3vl, medgemma, llava_med
DATASET="${3:-kvasir}"     # kvasir or endovis
TEST_DATA="${4}"           # Optional: path to test data
BASELINE_RESULTS="${5}"    # Optional: path to baseline results JSON

# Validate required arguments
if [ -z "$CHECKPOINT" ]; then
    echo "❌ ERROR: CHECKPOINT is required"
    echo "Usage: sbatch slurm/08_evaluate_multihead_cot.slurm <checkpoint> <model_type> <dataset> [test_data] [baseline_results]"
    exit 1
fi

# Dataset-specific defaults
if [ "$DATASET" == "kvasir" ]; then
    TEST_DATA="${4:-datasets/Kvasir-VQA/raw/metadata/test_metadata.json}"
    IMAGE_BASE_PATH="datasets/Kvasir-VQA/raw/images"
elif [ "$DATASET" == "endovis" ]; then
    TEST_DATA="${4:-datasets/EndoVis2018/raw/metadata/test_vqa_pairs.json}"
    IMAGE_BASE_PATH="datasets/EndoVis2018/raw/images"
else
    echo "❌ ERROR: Unknown dataset: $DATASET"
    exit 1
fi

# Question categories file
QUESTION_CATEGORIES="results/multihead_cot/question_categories.json"

# Output directory
OUTPUT_DIR="results/multihead_cot/evaluation"

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Model type: $MODEL_TYPE"
echo "  Dataset: $DATASET"
echo "  Test data: $TEST_DATA"
echo "  Image base path: $IMAGE_BASE_PATH"
echo "  Question categories: $QUESTION_CATEGORIES"
echo "  Baseline results: ${BASELINE_RESULTS:-None}"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check files exist
if [ ! -f "$CHECKPOINT" ] && [ ! -d "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "❌ ERROR: Test data file not found: $TEST_DATA"
    exit 1
fi

if [ ! -f "$QUESTION_CATEGORIES" ]; then
    echo "⚠️  WARNING: Question categories file not found: $QUESTION_CATEGORIES"
    QUESTION_CATEGORIES="question_categories.json"
fi

# GPU info
nvidia-smi
echo ""

# Build evaluation command
EVAL_CMD="python3 -u evaluate_multihead.py \
    --checkpoint \"$CHECKPOINT\" \
    --model-type \"$MODEL_TYPE\" \
    --test-data \"$TEST_DATA\" \
    --image-base-path \"$IMAGE_BASE_PATH\" \
    --question-categories \"$QUESTION_CATEGORIES\" \
    --dataset \"$DATASET\" \
    --output \"$OUTPUT_DIR\""

if [ ! -z "$BASELINE_RESULTS" ] && [ -f "$BASELINE_RESULTS" ]; then
    EVAL_CMD="$EVAL_CMD --baseline-results \"$BASELINE_RESULTS\""
fi

# Run evaluation
echo "Starting evaluation..."
eval $EVAL_CMD

EVAL_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "  ✅ Results saved to: $OUTPUT_DIR"
else
    echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
fi
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Clean up temporary cache
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up temporary cache..."
    rm -rf "$HF_HOME"
fi

exit $EVAL_EXIT_CODE













