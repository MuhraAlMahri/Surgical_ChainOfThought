#!/bin/bash
#SBATCH --job-name=train_qwen3vl_endovis
#SBATCH --output=slurm/logs/train_qwen3vl_endovis_%j.out
#SBATCH --error=slurm/logs/train_qwen3vl_endovis_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TRAINING: Qwen3-VL + EndoVis                            ║"
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

# Fix PEFT version
echo "🔧 Upgrading PEFT to >= 0.18.0..."
pip install --upgrade peft>=0.18.0 --quiet
echo "✓ PEFT upgraded"
echo ""

# Configuration
BASE_CHECKPOINT="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/qlora_experiments/models/qwen3vl_endovis_instruction/best_model"
if [ ! -d "$BASE_CHECKPOINT" ]; then
    BASE_CHECKPOINT="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/endovis2018_experiments/models/exp2_qwen_reordered_instruction/checkpoint-625"
    if [ ! -d "$BASE_CHECKPOINT" ]; then
        BASE_CHECKPOINT="Qwen/Qwen3-VL-8B-Instruct"
        echo "⚠️  Using base model: $BASE_CHECKPOINT"
    fi
fi

DATA_PATH="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/endovis18_surgery_r1_split/train.json"
IMAGE_PATH="${SLURM_SUBMIT_DIR}/datasets/EndoVis2018/raw/images"
if [ ! -d "$IMAGE_PATH" ]; then
    IMAGE_PATH="${SLURM_SUBMIT_DIR}/datasets/EndoVis-18-VQLA/images"
fi

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/qwen3vl_endovis_cot_5epochs"
QUESTION_CATEGORIES="${SLURM_SUBMIT_DIR}/question_categories.json"

if [ ! -f "$QUESTION_CATEGORIES" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" ]; then
        cp "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" "$QUESTION_CATEGORIES"
    else
        echo '{"kvasir": {}, "endovis": {}}' > "$QUESTION_CATEGORIES"
    fi
fi

cd $SLURM_SUBMIT_DIR

echo "📋 Configuration:"
echo "   Base checkpoint: $BASE_CHECKPOINT"
echo "   Data: $DATA_PATH"
echo "   Images: $IMAGE_PATH"
echo "   Output: $OUTPUT_DIR"
echo ""

# Run training
python train_multihead_cot.py \
    --model_type "qwen3vl" \
    --dataset "endovis" \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --image_base_path "$IMAGE_PATH" \
    --question_categories "$QUESTION_CATEGORIES" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 2e-5 \
    --epochs 5 \
    --batch_size 1 \
    --grad_accum 16 \
    --bf16 \
    --lora_r 8 \
    --lora_alpha 8 \
    --weight_decay 0.01 \
    --device cuda

EXIT_CODE=$?

echo ""
echo "End: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Training completed"
else
    echo "❌ FAILED: Training (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE









