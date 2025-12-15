#!/bin/bash
#SBATCH --job-name=mega_training_2gpu
#SBATCH --output=slurm/logs/mega_training_2gpu_%j.out
#SBATCH --error=slurm/logs/mega_training_2gpu_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MEGA TRAINING JOB: 2 GPUs                               ║"
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

# Fix PEFT version (do this once for all tasks)
echo "🔧 Upgrading PEFT to >= 0.18.0..."
pip install --upgrade peft>=0.18.0 --quiet
echo "✓ PEFT upgraded"
echo ""

# Fix HuggingFace authentication (needed for MedGemma)
echo "🔐 Setting up HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN not set. MedGemma requires authentication."
    echo "   Please set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable"
else
    echo "✓ HuggingFace token found"
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi
echo ""

# Question categories
QUESTION_CATEGORIES="${SLURM_SUBMIT_DIR}/question_categories.json"
if [ ! -f "$QUESTION_CATEGORIES" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" ]; then
        cp "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" "$QUESTION_CATEGORIES"
    else
        echo '{"kvasir": {}, "endovis": {}}' > "$QUESTION_CATEGORIES"
    fi
fi

cd $SLURM_SUBMIT_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

# MedGemma + EndoVis
MEDGEMMA_ENDOVIS_BASE="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/qlora_experiments/models/medgemma_endovis_instruction/best_model"
if [ ! -d "$MEDGEMMA_ENDOVIS_BASE" ]; then
    MEDGEMMA_ENDOVIS_BASE="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/endovis2018_experiments/models/exp1_medgemma4b_instruction_r1"
    if [ ! -d "$MEDGEMMA_ENDOVIS_BASE" ]; then
        MEDGEMMA_ENDOVIS_BASE="google/medgemma-4b-it"
        echo "⚠️  Using base model for MedGemma+EndoVis: $MEDGEMMA_ENDOVIS_BASE"
    fi
fi
MEDGEMMA_ENDOVIS_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/endovis18_surgery_r1_split/train.json"
MEDGEMMA_ENDOVIS_IMAGES="${SLURM_SUBMIT_DIR}/datasets/EndoVis2018/raw/images"
if [ ! -d "$MEDGEMMA_ENDOVIS_IMAGES" ]; then
    MEDGEMMA_ENDOVIS_IMAGES="${SLURM_SUBMIT_DIR}/datasets/EndoVis-18-VQLA/images"
fi
MEDGEMMA_ENDOVIS_OUTPUT="${SLURM_SUBMIT_DIR}/results/medgemma_endovis_cot_5epochs"

# Optional: Add more training tasks here if needed
# For now, we'll run MedGemma + EndoVis on GPU 0
# GPU 1 can be used for another task or left idle

echo "📋 Configuration:"
echo "   Task 1 (GPU 0): MedGemma + EndoVis"
echo "   Task 2 (GPU 1): Available for additional training"
echo ""

# ============================================================================
# FUNCTIONS
# ============================================================================

# Function to run training
run_training() {
    local gpu_id=$1
    local model_type=$2
    local dataset=$3
    local base_checkpoint=$4
    local data_path=$5
    local image_path=$6
    local output_dir=$7
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 GPU $gpu_id: Training $model_type on $dataset"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start time: $(date)"
    echo "   Base checkpoint: $base_checkpoint"
    echo "   Data: $data_path"
    echo "   Images: $image_path"
    echo "   Output: $output_dir"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python train_multihead_cot.py \
        --model_type "$model_type" \
        --dataset "$dataset" \
        --base_checkpoint "$base_checkpoint" \
        --data_path "$data_path" \
        --image_base_path "$image_path" \
        --question_categories "$QUESTION_CATEGORIES" \
        --output_dir "$output_dir" \
        --learning_rate 2e-5 \
        --epochs 5 \
        --batch_size 1 \
        --grad_accum 16 \
        --bf16 \
        --lora_r 8 \
        --lora_alpha 8 \
        --weight_decay 0.01 \
        --device cuda
    
    local exit_code=$?
    echo ""
    echo "End time: $(date)"
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: Training completed"
    else
        echo "❌ FAILED: Training (exit code: $exit_code)"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    return $exit_code
}

# ============================================================================
# EXECUTION
# ============================================================================

# Run MedGemma + EndoVis on GPU 0
echo "⏳ Starting MedGemma + EndoVis training on GPU 0..."
run_training 0 medgemma endovis \
    "$MEDGEMMA_ENDOVIS_BASE" \
    "$MEDGEMMA_ENDOVIS_DATA" \
    "$MEDGEMMA_ENDOVIS_IMAGES" \
    "$MEDGEMMA_ENDOVIS_OUTPUT" > slurm/logs/gpu0_medgemma_endovis.log 2>&1 &
MEDGEMMA_ENDOVIS_PID=$!

echo "   MedGemma+EndoVis PID: $MEDGEMMA_ENDOVIS_PID (GPU 0)"
echo ""

# Wait for training to complete
wait $MEDGEMMA_ENDOVIS_PID
MEDGEMMA_ENDOVIS_EXIT=$?

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MEGA TRAINING JOB COMPLETED                              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Task Results:"
echo "  MedGemma+EndoVis (GPU 0): exit code $MEDGEMMA_ENDOVIS_EXIT"
echo ""
echo "End: $(date)"
echo ""

if [ $MEDGEMMA_ENDOVIS_EXIT -eq 0 ]; then
    echo "✅ SUCCESS: All training tasks completed!"
    echo ""
    echo "📊 Results:"
    echo "   • MedGemma+EndoVis: $MEDGEMMA_ENDOVIS_OUTPUT"
    exit 0
else
    echo "⚠️  SOME TASKS FAILED - Check logs for details"
    echo ""
    echo "📁 Logs:"
    echo "   • GPU 0 (MedGemma+EndoVis): slurm/logs/gpu0_medgemma_endovis.log"
    exit 1
fi









