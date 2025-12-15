#!/bin/bash
#SBATCH --job-name=eval_fixed_epoch5
#SBATCH --output=slurm/logs/eval_fixed_epoch5_%j.out
#SBATCH --error=slurm/logs/eval_fixed_epoch5_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  FIXED EVALUATION: Epoch 5 Checkpoint                    ║"
echo "║  Using model.generate() for proper answer generation     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Start: $(date)"
echo ""

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export HF_TOKEN=${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}

# Set HuggingFace cache to workspace to avoid home quota issues
export HF_HOME=$SLURM_SUBMIT_DIR/.hf_cache
export TRANSFORMERS_CACHE=$SLURM_SUBMIT_DIR/.hf_cache/transformers
export HF_HUB_CACHE=$SLURM_SUBMIT_DIR/.hf_cache/hub
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_HUB_CACHE
echo "Using HuggingFace cache: $HF_HOME"
echo ""

# Configuration
EPOCH5_CHECKPOINT="${SLURM_SUBMIT_DIR}/results/qwen3vl_kvasir_cot_5epochs/checkpoint_epoch_5.pt"
BASE_MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
MODEL_TYPE="qwen3vl"
DATASET="kvasir"

# Data paths
TEST_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_baseline_image_level_70_15_15/test.json"
if [ ! -f "$TEST_DATA" ]; then
    TEST_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/test.json"
fi

IMAGE_BASE="${SLURM_SUBMIT_DIR}/datasets/Kvasir-VQA/raw/images"
if [ ! -d "$IMAGE_BASE" ]; then
    IMAGE_BASE="${SLURM_SUBMIT_DIR}/datasets/kvasir/images"
fi

QUESTION_CATEGORIES="${SLURM_SUBMIT_DIR}/question_categories.json"
if [ ! -f "$QUESTION_CATEGORIES" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" ]; then
        cp "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" "$QUESTION_CATEGORIES"
        echo "✓ Copied question_categories.json from results directory"
    else
        echo "⚠️  Warning: question_categories.json not found, creating empty file"
        echo '{"kvasir": {}, "endovis": {}}' > "$QUESTION_CATEGORIES"
    fi
fi

# Output directory
EVAL_OUTPUT="${SLURM_SUBMIT_DIR}/results/eval_epoch5_qwen3vl_kvasir_FIXED"

echo "📋 Configuration:"
echo "   Checkpoint: $EPOCH5_CHECKPOINT"
echo "   Base model: $BASE_MODEL_NAME"
echo "   Test data: $TEST_DATA"
echo "   Images: $IMAGE_BASE"
echo "   Output: $EVAL_OUTPUT"
echo ""

# Run fixed evaluation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Running FIXED Evaluation (using model.generate())"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Start time: $(date)"
echo ""

python evaluate_epoch3_checkpoint_FIXED.py \
    --checkpoint "$EPOCH5_CHECKPOINT" \
    --base_checkpoint "$BASE_MODEL_NAME" \
    --model_type "$MODEL_TYPE" \
    --dataset "$DATASET" \
    --test_data "$TEST_DATA" \
    --image_base_path "$IMAGE_BASE" \
    --question_categories "$QUESTION_CATEGORIES" \
    --output "$EVAL_OUTPUT"

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Fixed evaluation completed"
    echo ""
    echo "📊 Results saved to: $EVAL_OUTPUT"
    echo ""
    echo "Expected improvements:"
    echo "   • Real accuracy values (not 0%)"
    echo "   • Full answer predictions (not single tokens)"
    echo "   • Proper comparison with baseline"
else
    echo "❌ FAILED: Evaluation (exit code: $EXIT_CODE)"
    echo "Check logs for details"
fi

echo ""
echo "End: $(date)"


