#!/bin/bash
#SBATCH --job-name=medgemma_kvasir_cot
#SBATCH --output=slurm/logs/train_medgemma_kvasir_finetuned_cot_%j.out
#SBATCH --error=slurm/logs/train_medgemma_kvasir_finetuned_cot_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Train MedGemma-4B Kvasir Fine-tuned+COT (Epochs 1-5)   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/bin/activate base

# Install peft if needed
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
# CONFIGURATION
# ============================================================================

# Base model
BASE_MODEL="google/medgemma-4b-it"

# Fine-tuned checkpoint (instruction fine-tuned)
KVASIR_FT_BASE_DIR="corrected_1-5_experiments/exp1/models/exp1_medgemma4b_instruction"
KVASIR_FT_CHECKPOINT=""
if [ -d "$KVASIR_FT_BASE_DIR" ]; then
    # Find latest checkpoint
    LATEST_CHECKPOINT=$(ls -d ${KVASIR_FT_BASE_DIR}/checkpoint-* 2>/dev/null | sort -V | awk 'END{print}')
    if [ -n "$LATEST_CHECKPOINT" ]; then
        KVASIR_FT_CHECKPOINT="$LATEST_CHECKPOINT"
        echo "✓ Found Kvasir fine-tuned checkpoint: $KVASIR_FT_CHECKPOINT"
    else
        # Use root directory if no checkpoints
        KVASIR_FT_CHECKPOINT="$KVASIR_FT_BASE_DIR"
        echo "✓ Using Kvasir fine-tuned checkpoint directory: $KVASIR_FT_CHECKPOINT"
    fi
else
    echo "⚠️  Kvasir fine-tuned checkpoint not found, using base model"
    KVASIR_FT_CHECKPOINT="$BASE_MODEL"
fi

# Data paths
KVASIR_TRAIN_DATA="corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json"
KVASIR_IMAGE_DIR="datasets/Kvasir-VQA/raw/images"

# Question categories
QUESTION_CATEGORIES="question_categories.json"

# Output directory
KVASIR_TRAIN_OUTPUT="results/medgemma_kvasir_finetuned_cot"

echo "📋 Configuration:"
echo "   Base checkpoint: $KVASIR_FT_CHECKPOINT"
echo "   Training data: $KVASIR_TRAIN_DATA"
echo "   Image directory: $KVASIR_IMAGE_DIR"
echo "   Output: $KVASIR_TRAIN_OUTPUT"
echo "   Question categories: $QUESTION_CATEGORIES"
echo ""

# ============================================================================
# TRAINING
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting Training: MedGemma Kvasir Fine-tuned+COT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Start time: $(date)"
echo ""

python train_multihead_cot.py \
    --model_type medgemma \
    --dataset kvasir \
    --base_checkpoint "$KVASIR_FT_CHECKPOINT" \
    --question_categories "$QUESTION_CATEGORIES" \
    --data_path "$KVASIR_TRAIN_DATA" \
    --image_base_path "$KVASIR_IMAGE_DIR" \
    --output_dir "$KVASIR_TRAIN_OUTPUT" \
    --learning_rate 5e-5 \
    --epochs 5 \
    --batch_size 4 \
    --grad_accum 4 \
    --weight_decay 0.01 \
    --max_seq_len 3072 \
    --bf16 \
    --lora_r 4 \
    --lora_alpha 8 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training completed at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
    echo ""
    echo "Checkpoints saved to: $KVASIR_TRAIN_OUTPUT"
    ls -lh "$KVASIR_TRAIN_OUTPUT" 2>/dev/null || echo "⚠️  Output directory not found"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit $EXIT_CODE

