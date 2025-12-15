#!/bin/bash
#SBATCH --job-name=eval_train_2gpu
#SBATCH --output=slurm/logs/eval_train_2gpu_%j.out
#SBATCH --error=slurm/logs/eval_train_2gpu_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  DUAL-TRACK: EVALUATE EPOCH 3 + TRAIN TO EPOCH 5         ║"
echo "║  Using 2 GPUs                                             ║"
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
CHECKPOINT_EPOCH3="results/multihead_cot/qwen3vl_kvasir_cot_20251208_233609/checkpoint_epoch_3.pt"
# Use HuggingFace base model name (not local checkpoint)
# The checkpoint was trained with Qwen3-VL-8B-Instruct (hidden size 4096)
# NOT Qwen2-VL-2B-Instruct (hidden size 1536)
BASE_MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
echo "Using base model: $BASE_MODEL_NAME"
MODEL_TYPE="qwen3vl"
DATASET="kvasir"

# Data paths (adjust these based on your actual paths)
# Try common paths - user should verify these exist
TRAIN_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_baseline_image_level_70_15_15/train.json"
if [ ! -f "$TRAIN_DATA" ]; then
    # Try alternative path
    TRAIN_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json"
fi

TEST_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_baseline_image_level_70_15_15/test.json"
if [ ! -f "$TEST_DATA" ]; then
    # Try alternative path
    TEST_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/test.json"
fi

IMAGE_BASE="${SLURM_SUBMIT_DIR}/datasets/Kvasir-VQA/raw/images"
if [ ! -d "$IMAGE_BASE" ]; then
    # Try alternative path
    IMAGE_BASE="${SLURM_SUBMIT_DIR}/datasets/kvasir/images"
fi
QUESTION_CATEGORIES="${SLURM_SUBMIT_DIR}/question_categories.json"
# Ensure question_categories.json exists
if [ ! -f "$QUESTION_CATEGORIES" ]; then
    # Try to copy from results directory
    if [ -f "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" ]; then
        cp "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" "$QUESTION_CATEGORIES"
        echo "✓ Copied question_categories.json from results directory"
    else
        echo "⚠️  Warning: question_categories.json not found, creating empty file"
        echo '{"kvasir": {}, "endovis": {}}' > "$QUESTION_CATEGORIES"
    fi
fi

# Output directories
EVAL_OUTPUT="${SLURM_SUBMIT_DIR}/results/eval_epoch3_qwen3vl_kvasir"
TRAIN_OUTPUT="${SLURM_SUBMIT_DIR}/results/qwen3vl_kvasir_cot_5epochs"

echo "📋 Configuration:"
echo "   GPU 0: Evaluate Epoch 3 checkpoint"
echo "   GPU 1: Train Epoch 3 → 5"
echo ""
echo "   Checkpoint: $CHECKPOINT_EPOCH3"
echo "   Base model: $BASE_MODEL_NAME"
echo ""

# Function to run evaluation on GPU 0
run_evaluation() {
    local gpu_id=$1
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 GPU $gpu_id: Evaluating Epoch 3 Checkpoint"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start time: $(date)"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python evaluate_epoch3_checkpoint.py \
        --checkpoint "$CHECKPOINT_EPOCH3" \
        --base_checkpoint "$BASE_MODEL_NAME" \
        --model_type "$MODEL_TYPE" \
        --dataset "$DATASET" \
        --test_data "$TEST_DATA" \
        --image_base_path "$IMAGE_BASE" \
        --question_categories "$QUESTION_CATEGORIES" \
        --output "$EVAL_OUTPUT"
    
    local exit_code=$?
    echo ""
    echo "End time: $(date)"
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: Evaluation completed"
    else
        echo "❌ FAILED: Evaluation (exit code: $exit_code)"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    return $exit_code
}

# Function to run training on GPU 1
run_training() {
    local gpu_id=$1
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 GPU $gpu_id: Training Epoch 3 → 5"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start time: $(date)"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python resume_training_epoch3to5.py \
        --checkpoint "$CHECKPOINT_EPOCH3" \
        --base_checkpoint "$BASE_MODEL_NAME" \
        --model_type "$MODEL_TYPE" \
        --dataset "$DATASET" \
        --data_path "$TRAIN_DATA" \
        --image_base_path "$IMAGE_BASE" \
        --question_categories "$QUESTION_CATEGORIES" \
        --output_dir "$TRAIN_OUTPUT" \
        --total_epochs 5 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --batch_size 1 \
        --grad_accum 16 \
        --bf16 \
        --lora_r 8 \
        --lora_alpha 8 \
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

# Run both tasks in parallel
echo "⏳ Starting parallel execution..."
echo ""

# Start evaluation on GPU 0 in background
run_evaluation 0 > slurm/logs/gpu0_evaluation.log 2>&1 &
EVAL_PID=$!

# Start training on GPU 1 in background
run_training 1 > slurm/logs/gpu1_training.log 2>&1 &
TRAIN_PID=$!

echo "   Evaluation PID: $EVAL_PID (GPU 0)"
echo "   Training PID: $TRAIN_PID (GPU 1)"
echo ""
echo "⏳ Waiting for both tasks to complete..."

# Wait for both processes
wait $EVAL_PID
EVAL_EXIT=$?

wait $TRAIN_PID
TRAIN_EXIT=$?

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALL TASKS COMPLETED                                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Evaluation (GPU 0) exit code: $EVAL_EXIT"
echo "Training (GPU 1) exit code: $TRAIN_EXIT"
echo ""
echo "End: $(date)"
echo ""

# Check final status
if [ $EVAL_EXIT -eq 0 ] && [ $TRAIN_EXIT -eq 0 ]; then
    echo "✅ ALL TASKS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 Results:"
    echo "   • Evaluation: $EVAL_OUTPUT"
    echo "   • Training: $TRAIN_OUTPUT"
    exit 0
else
    echo "⚠️  SOME TASKS FAILED - Check logs for details"
    echo ""
    echo "📁 Logs:"
    echo "   • GPU 0 (Evaluation): slurm/logs/gpu0_evaluation.log"
    echo "   • GPU 1 (Training): slurm/logs/gpu1_training.log"
    exit 1
fi

