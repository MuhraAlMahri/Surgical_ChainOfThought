#!/bin/bash
#SBATCH --job-name=task4_endovis_finetuned_cot
#SBATCH --output=slurm/logs/task4_endovis_finetuned_cot_%j.out
#SBATCH --error=slurm/logs/task4_endovis_finetuned_cot_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Task 4: EndoVis Fine-tuned+COT Training on GPU 0        ║"
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

# ============================================================================
# CONFIGURATION (same as mega job Task 4)
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

echo "📋 Configuration:"
echo "   Base checkpoint: $ENDOVIS_FT_CHECKPOINT"
echo "   Training data: $ENDOVIS_TRAIN_DATA"
echo "   Image directory: $ENDOVIS_IMAGE_DIR"
echo "   Output: $ENDOVIS_TRAIN_OUTPUT"
echo "   Time limit: 72 hours"
echo ""

# Verify checkpoint exists
if [ ! -d "$ENDOVIS_FT_CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found: $ENDOVIS_FT_CHECKPOINT"
    exit 1
fi

# Verify training data exists
if [ ! -f "$ENDOVIS_TRAIN_DATA" ]; then
    echo "❌ ERROR: Training data not found: $ENDOVIS_TRAIN_DATA"
    exit 1
fi

# ============================================================================
# RUN TASK 4 ON GPU 0
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 GPU 0 - TASK 4: Fine-tuned+COT EndoVis Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Start time: $(date)"
echo ""

# Use GPU 0 (only GPU available in this job)
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
    --lora_alpha 8

EXIT_CODE=$?

echo ""
echo "Task 4 completed at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Task 4 (EndoVis Fine-tuned+COT) completed successfully on GPU 0"
else
    echo "❌ Task 4 (EndoVis Fine-tuned+COT) failed with exit code: $EXIT_CODE"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit $EXIT_CODE

