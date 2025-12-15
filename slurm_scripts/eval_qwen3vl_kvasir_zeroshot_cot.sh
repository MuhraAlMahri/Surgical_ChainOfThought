#!/bin/bash
#SBATCH --job-name=eval_qwen3vl_kvasir
#SBATCH --output=slurm/logs/eval_qwen3vl_kvasir_zeroshot_cot_%j.out
#SBATCH --error=slurm/logs/eval_qwen3vl_kvasir_zeroshot_cot_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Qwen3-VL Kvasir Zeroshot+COT Evaluation                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/bin/activate base

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

# Checkpoint
COT_CHECKPOINT="results/qwen3vl_kvasir_cot_5epochs/checkpoint_epoch_5.pt"
BASE_CHECKPOINT="Qwen/Qwen3-VL-8B-Instruct"

# Data paths
DATA_DIR="corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15"
IMAGE_DIR="datasets/Kvasir-VQA/raw/images"

# Question categories
QUESTION_CATEGORIES="question_categories.json"

# Output
OUTPUT_FILE="results/qwen3vl_kvasir_zeroshot_cot_FULL.json"

echo "📋 Configuration:"
echo "   Checkpoint: $COT_CHECKPOINT"
echo "   Base model: $BASE_CHECKPOINT"
echo "   Data path: $DATA_DIR"
echo "   Output: $OUTPUT_FILE"
echo ""

# Check checkpoint exists
if [ ! -f "$COT_CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found: $COT_CHECKPOINT"
    exit 1
fi

# Run evaluation
echo "Starting evaluation..."
python evaluate_multihead_cot_comprehensive.py \
    --checkpoint "$COT_CHECKPOINT" \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --model_name qwen3vl \
    --dataset kvasir \
    --data_path "$DATA_DIR" \
    --image_base_path "$IMAGE_DIR" \
    --question_categories "$QUESTION_CATEGORIES" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 1 \
    --use_flexible_matching

EXIT_CODE=$?

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║  EVALUATION COMPLETED SUCCESSFULLY                      ║"
else
    echo "║  EVALUATION FAILED                                       ║"
fi
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo "Output: $OUTPUT_FILE"
echo "Exit code: $EXIT_CODE"
echo ""

