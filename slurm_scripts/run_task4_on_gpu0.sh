#!/bin/bash
# Standalone script to run Task 4 (EndoVis Fine-tuned+COT Training) on GPU 0
# This can be run manually or submitted as a separate job

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Task 4: EndoVis Fine-tuned+COT Training on GPU 0       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
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

# ============================================================================
# CONFIGURATION (same as mega job)
# ============================================================================

# Fine-tuned checkpoint
ENDOVIS_FT_CHECKPOINT="corrected_1-5_experiments/endovis2018_experiments/models/exp1_random"

# Training data
ENDOVIS_TRAIN_DATA="corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl"

# Image directory
ENDOVIS_IMAGE_DIR="datasets/EndoVis2018/raw/images"

# Question categories
QUESTION_CATEGORIES="question_categories.json"

# Output directory
ENDOVIS_TRAIN_OUTPUT="results/qwen3vl_endovis_finetuned_cot"

# Log file
LOG_FILE="slurm/logs/task4_endovis_train_gpu0_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "📋 Configuration:"
echo "   Base checkpoint: $ENDOVIS_FT_CHECKPOINT"
echo "   Training data: $ENDOVIS_TRAIN_DATA"
echo "   Image directory: $ENDOVIS_IMAGE_DIR"
echo "   Output: $ENDOVIS_TRAIN_OUTPUT"
echo "   Log: $LOG_FILE"
echo ""

# ============================================================================
# RUN TASK 4 ON GPU 0
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 GPU 0 - TASK 4: Fine-tuned+COT EndoVis Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Start time: $(date)"
echo ""

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

python train_multihead_cot.py \
    --model_type qwen3vl \
    --dataset endovis \
    --base_checkpoint "$ENDOVIS_FT_CHECKPOINT" \
    --question_categories "$QUESTION_CATEGORIES" \
    --data_path "$ENDOVIS_TRAIN_DATA" \
    --image_base_path "$ENDOVIS_IMAGE_DIR" \
    --output_dir "$ENDOVIS_TRAIN_OUTPUT" \
    --learning_rate 5e-5 \
    --epochs 5 \
    --batch_size 1 \
    --grad_accum 16 \
    --weight_decay 0.01 \
    --max_seq_len 3072 \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 4 \
    --lora_alpha 8 \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo ""
echo "Task 4 completed at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Task 4 (EndoVis Fine-tuned+COT) completed successfully"
else
    echo "❌ Task 4 (EndoVis Fine-tuned+COT) failed with exit code: $EXIT_CODE"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Log file: $LOG_FILE"
echo ""

exit $EXIT_CODE


