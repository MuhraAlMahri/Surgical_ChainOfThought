#!/bin/bash
#SBATCH --job-name=medgemma_task4_fix
#SBATCH --output=slurm/logs/medgemma_task4_fixed_%j.out
#SBATCH --error=slurm/logs/medgemma_task4_fixed_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MedGemma Task 4: EndoVis Fine-tuned+COT Training (Fixed)║"
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
export CUDA_VISIBLE_DEVICES=0

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

# Base checkpoint (LoRA adapter)
ENDOVIS_FT_CHECKPOINT="corrected_1-5_experiments/endovis2018_experiments/models/exp1_medgemma4b_instruction_r1"

# Training data
ENDOVIS_TRAIN_DATA="corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl"
ENDOVIS_IMAGE_DIR="datasets/EndoVis2018/raw/images"

# Question categories
QUESTION_CATEGORIES="question_categories.json"

# Output directory
ENDOVIS_TRAIN_OUTPUT="results/medgemma_endovis_finetuned_cot"

echo "📋 Configuration:"
echo "   Base checkpoint: $ENDOVIS_FT_CHECKPOINT"
echo "   Training data: $ENDOVIS_TRAIN_DATA"
echo "   Output: $ENDOVIS_TRAIN_OUTPUT"
echo ""
echo "   ✅ FIX: Processor will be loaded from base model 'google/medgemma-4b-it'"
echo "   ✅ FIX: Model will be loaded from checkpoint path"
echo ""

# Check checkpoint exists
if [ ! -d "$ENDOVIS_FT_CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found: $ENDOVIS_FT_CHECKPOINT"
    exit 1
fi

# Run training
echo "Starting training..."
python train_multihead_cot.py \
    --model_type medgemma \
    --dataset endovis \
    --base_checkpoint "$ENDOVIS_FT_CHECKPOINT" \
    --question_categories "$QUESTION_CATEGORIES" \
    --data_path "$ENDOVIS_TRAIN_DATA" \
    --image_base_path "$ENDOVIS_IMAGE_DIR" \
    --output_dir "$ENDOVIS_TRAIN_OUTPUT" \
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
echo "╔══════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║  TRAINING COMPLETED SUCCESSFULLY                         ║"
else
    echo "║  TRAINING FAILED                                         ║"
fi
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo "Output: $ENDOVIS_TRAIN_OUTPUT"
echo "Exit code: $EXIT_CODE"
echo ""



