#!/bin/bash
#SBATCH --job-name=eval_kvasir_zeroshot_cot
#SBATCH --output=slurm/logs/eval_kvasir_zeroshot_cot_%j.out
#SBATCH --error=slurm/logs/eval_kvasir_zeroshot_cot_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Kvasir Zeroshot+COT Evaluation (Task 1)                ║"
echo "║  Evaluating on all 8,984 test samples                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/bin/activate base

# Install peft if needed (to user-writable location)
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PIP_TARGET="/l/users/muhra.almahri/.local/lib/python${PYTHON_VERSION}/site-packages"
export PYTHONPATH="${PIP_TARGET}:${PYTHONPATH}"
mkdir -p ${PIP_TARGET}
pip install --target ${PIP_TARGET} -q peft 2>/dev/null || echo "Note: peft may already be installed"

export PYTHONUNBUFFERED=1

# Set HuggingFace cache
export HF_HOME=/l/users/muhra.almahri/.cache/hf_shared
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}

# Set working directory
cd /l/users/muhra.almahri/Surgical_COT

# Configuration
CHECKPOINT="results/qwen3vl_kvasir_cot_5epochs/checkpoint_epoch_5.pt"
BASE_CHECKPOINT="Qwen/Qwen3-VL-8B-Instruct"
DATA_PATH="/l/users/muhra.almahri/datasets/kvasir-vqa"
IMAGE_BASE_PATH="/l/users/muhra.almahri/datasets/kvasir-vqa/images"
OUTPUT_FILE="results/qwen3vl_kvasir_zeroshot_cot_FULL.json"
QUESTION_CATEGORIES="question_categories.json"

echo "📋 Configuration:"
echo "   Checkpoint: $CHECKPOINT"
echo "   Base model: $BASE_CHECKPOINT"
echo "   Dataset: Kvasir"
echo "   Data path: $DATA_PATH"
echo "   Image base path: $IMAGE_BASE_PATH"
echo "   Output file: $OUTPUT_FILE"
echo "   Question categories: $QUESTION_CATEGORIES"
echo ""

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Verify data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ ERROR: Data path not found: $DATA_PATH"
    exit 1
fi

# Verify test.json exists
if [ ! -f "$DATA_PATH/test.json" ]; then
    echo "❌ ERROR: test.json not found in: $DATA_PATH"
    echo "   Looking for: $DATA_PATH/test.json"
    exit 1
fi

# Verify image directory exists
if [ ! -d "$IMAGE_BASE_PATH" ]; then
    echo "❌ ERROR: Image directory not found: $IMAGE_BASE_PATH"
    exit 1
fi

# GPU info
echo "🖥️  GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Run evaluation
echo "🚀 Starting evaluation..."
echo ""

python evaluate_multihead_cot_comprehensive.py \
    --checkpoint "$CHECKPOINT" \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --model_name qwen3vl \
    --dataset kvasir \
    --data_path "$DATA_PATH" \
    --image_base_path "$IMAGE_BASE_PATH" \
    --question_categories "$QUESTION_CATEGORIES" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 1 \
    --use_flexible_matching

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo ""
    echo "📊 Final Results:"
    echo ""
    python3 -c "
import json
try:
    with open('$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    print(f\"Overall Accuracy: {data['overall_accuracy']:.2%} ({data['correct']}/{data['total']})\")
    print(f\"\nPer-Category Accuracies:\")
    for cat, acc in data['category_accuracies'].items():
        print(f\"  {cat}: {acc:.2%} ({data['category_correct'][cat]}/{data['category_total'][cat]})\")
except Exception as e:
    print(f\"Could not parse results: {e}\")
"
    echo ""
    echo "📁 Results saved to: $OUTPUT_FILE"
else
    echo "❌ Evaluation failed with exit code: $EXIT_CODE"
    echo "Check logs for details: slurm/logs/eval_kvasir_zeroshot_cot_${SLURM_JOB_ID}.err"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  JOB COMPLETED                                           ║"
echo "╚══════════════════════════════════════════════════════════╝"





