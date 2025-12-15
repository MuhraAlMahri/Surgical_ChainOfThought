#!/bin/bash
#SBATCH --job-name=mega_cot_2gpu
#SBATCH --output=slurm/logs/mega_cot_2gpu_%j.out
#SBATCH --error=slurm/logs/mega_cot_2gpu_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MEGA JOB: EVALUATION + 3 TRAINING TASKS                ║"
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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Evaluation: Epoch 5 checkpoint (Qwen3-VL + Kvasir)
EPOCH5_CHECKPOINT="${SLURM_SUBMIT_DIR}/results/qwen3vl_kvasir_cot_5epochs/checkpoint_epoch_5.pt"
BASE_MODEL_QWEN3VL="Qwen/Qwen3-VL-8B-Instruct"

# Training configurations
# Qwen3-VL + EndoVis
# Try to find checkpoint, fallback to base model if not found
QWEN3VL_ENDOVIS_BASE="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/qlora_experiments/models/qwen3vl_endovis_instruction/best_model"
if [ ! -d "$QWEN3VL_ENDOVIS_BASE" ]; then
    # Try alternative paths
    QWEN3VL_ENDOVIS_BASE="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/endovis2018_experiments/models/exp2_qwen_reordered_instruction/checkpoint-625"
    if [ ! -d "$QWEN3VL_ENDOVIS_BASE" ]; then
        # Fallback to base HuggingFace model
        QWEN3VL_ENDOVIS_BASE="Qwen/Qwen3-VL-8B-Instruct"
        echo "⚠️  Using base model for Qwen3-VL+EndoVis: $QWEN3VL_ENDOVIS_BASE"
    fi
fi
QWEN3VL_ENDOVIS_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/endovis18_surgery_r1_split/train.json"
QWEN3VL_ENDOVIS_IMAGES="${SLURM_SUBMIT_DIR}/datasets/EndoVis2018/raw/images"
if [ ! -d "$QWEN3VL_ENDOVIS_IMAGES" ]; then
    QWEN3VL_ENDOVIS_IMAGES="${SLURM_SUBMIT_DIR}/datasets/EndoVis-18-VQLA/images"
fi

# MedGemma + Kvasir
MEDGEMMA_KVASIR_BASE="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/qlora_experiments/models/medgemma_kvasir_instruction/best_model"
if [ ! -d "$MEDGEMMA_KVASIR_BASE" ]; then
    # Try alternative paths
    MEDGEMMA_KVASIR_BASE="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/endovis2018_experiments/models/exp1_medgemma4b_instruction_r1"
    if [ ! -d "$MEDGEMMA_KVASIR_BASE" ]; then
        # Fallback to base HuggingFace model
        MEDGEMMA_KVASIR_BASE="google/medgemma-4b-it"
        echo "⚠️  Using base model for MedGemma+Kvasir: $MEDGEMMA_KVASIR_BASE"
    fi
fi
MEDGEMMA_KVASIR_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_baseline_image_level_70_15_15/train.json"
if [ ! -f "$MEDGEMMA_KVASIR_DATA" ]; then
    MEDGEMMA_KVASIR_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json"
fi
MEDGEMMA_KVASIR_IMAGES="${SLURM_SUBMIT_DIR}/datasets/Kvasir-VQA/raw/images"
if [ ! -d "$MEDGEMMA_KVASIR_IMAGES" ]; then
    MEDGEMMA_KVASIR_IMAGES="${SLURM_SUBMIT_DIR}/datasets/kvasir/images"
fi

# MedGemma + EndoVis
MEDGEMMA_ENDOVIS_BASE="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/qlora_experiments/models/medgemma_endovis_instruction/best_model"
if [ ! -d "$MEDGEMMA_ENDOVIS_BASE" ]; then
    # Try alternative paths
    MEDGEMMA_ENDOVIS_BASE="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/endovis2018_experiments/models/exp1_medgemma4b_instruction_r1"
    if [ ! -d "$MEDGEMMA_ENDOVIS_BASE" ]; then
        # Fallback to base HuggingFace model
        MEDGEMMA_ENDOVIS_BASE="google/medgemma-4b-it"
        echo "⚠️  Using base model for MedGemma+EndoVis: $MEDGEMMA_ENDOVIS_BASE"
    fi
fi
MEDGEMMA_ENDOVIS_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/endovis18_surgery_r1_split/train.json"
MEDGEMMA_ENDOVIS_IMAGES="${SLURM_SUBMIT_DIR}/datasets/EndoVis2018/raw/images"
if [ ! -d "$MEDGEMMA_ENDOVIS_IMAGES" ]; then
    MEDGEMMA_ENDOVIS_IMAGES="${SLURM_SUBMIT_DIR}/datasets/EndoVis-18-VQLA/images"
fi

# Test data for evaluation
KVASIR_TEST_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_baseline_image_level_70_15_15/test.json"
if [ ! -f "$KVASIR_TEST_DATA" ]; then
    KVASIR_TEST_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/test.json"
fi

# Question categories
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

# Output directories
EVAL_OUTPUT="${SLURM_SUBMIT_DIR}/results/eval_epoch5_qwen3vl_kvasir"
QWEN3VL_ENDOVIS_OUTPUT="${SLURM_SUBMIT_DIR}/results/qwen3vl_endovis_cot_5epochs"
MEDGEMMA_KVASIR_OUTPUT="${SLURM_SUBMIT_DIR}/results/medgemma_kvasir_cot_5epochs"
MEDGEMMA_ENDOVIS_OUTPUT="${SLURM_SUBMIT_DIR}/results/medgemma_endovis_cot_5epochs"

echo "📋 Configuration:"
echo "   Task 1 (GPU 0): Evaluate Epoch 5 checkpoint (Qwen3-VL + Kvasir)"
echo "   Task 2 (GPU 1): Train Qwen3-VL + EndoVis"
echo "   Task 3 (GPU 0): Train MedGemma + Kvasir (after eval completes)"
echo "   Task 4 (GPU 1): Train MedGemma + EndoVis (after task 2 completes)"
echo ""

# ============================================================================
# FUNCTIONS
# ============================================================================

# Function to run evaluation on GPU 0
run_evaluation_epoch5() {
    local gpu_id=$1
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 GPU $gpu_id: Evaluating Epoch 5 Checkpoint (Qwen3-VL + Kvasir)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start time: $(date)"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python evaluate_epoch3_checkpoint_FIXED.py \
        --checkpoint "$EPOCH5_CHECKPOINT" \
        --base_checkpoint "$BASE_MODEL_QWEN3VL" \
        --model_type "qwen3vl" \
        --dataset "kvasir" \
        --test_data "$KVASIR_TEST_DATA" \
        --image_base_path "$MEDGEMMA_KVASIR_IMAGES" \
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
# EXECUTION PLAN
# ============================================================================

# Phase 1: Start evaluation on GPU 0 and training on GPU 1 in parallel
echo "⏳ Phase 1: Starting evaluation (GPU 0) and Qwen3-VL+EndoVis training (GPU 1)..."
echo ""

# Start evaluation on GPU 0 in background
run_evaluation_epoch5 0 > slurm/logs/gpu0_evaluation.log 2>&1 &
EVAL_PID=$!

# Start Qwen3-VL + EndoVis training on GPU 1 in background
run_training 1 qwen3vl endovis \
    "$QWEN3VL_ENDOVIS_BASE" \
    "$QWEN3VL_ENDOVIS_DATA" \
    "$QWEN3VL_ENDOVIS_IMAGES" \
    "$QWEN3VL_ENDOVIS_OUTPUT" > slurm/logs/gpu1_qwen3vl_endovis.log 2>&1 &
QWEN3VL_ENDOVIS_PID=$!

echo "   Evaluation PID: $EVAL_PID (GPU 0)"
echo "   Qwen3-VL+EndoVis PID: $QWEN3VL_ENDOVIS_PID (GPU 1)"
echo ""

# Wait for evaluation to complete
echo "⏳ Waiting for evaluation to complete..."
wait $EVAL_PID
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
else
    echo "⚠️  Evaluation failed (exit code: $EVAL_EXIT), but continuing with training..."
fi
echo ""

# Phase 2: Start MedGemma + Kvasir training on GPU 0 (now free)
echo "⏳ Phase 2: Starting MedGemma+Kvasir training on GPU 0..."
run_training 0 medgemma kvasir \
    "$MEDGEMMA_KVASIR_BASE" \
    "$MEDGEMMA_KVASIR_DATA" \
    "$MEDGEMMA_KVASIR_IMAGES" \
    "$MEDGEMMA_KVASIR_OUTPUT" > slurm/logs/gpu0_medgemma_kvasir.log 2>&1 &
MEDGEMMA_KVASIR_PID=$!

echo "   MedGemma+Kvasir PID: $MEDGEMMA_KVASIR_PID (GPU 0)"
echo ""

# Wait for Qwen3-VL + EndoVis to complete
echo "⏳ Waiting for Qwen3-VL+EndoVis training to complete..."
wait $QWEN3VL_ENDOVIS_PID
QWEN3VL_ENDOVIS_EXIT=$?

if [ $QWEN3VL_ENDOVIS_EXIT -eq 0 ]; then
    echo "✅ Qwen3-VL+EndoVis training completed successfully"
else
    echo "⚠️  Qwen3-VL+EndoVis training failed (exit code: $QWEN3VL_ENDOVIS_EXIT)"
fi
echo ""

# Phase 3: Start MedGemma + EndoVis training on GPU 1 (now free)
echo "⏳ Phase 3: Starting MedGemma+EndoVis training on GPU 1..."
run_training 1 medgemma endovis \
    "$MEDGEMMA_ENDOVIS_BASE" \
    "$MEDGEMMA_ENDOVIS_DATA" \
    "$MEDGEMMA_ENDOVIS_IMAGES" \
    "$MEDGEMMA_ENDOVIS_OUTPUT" > slurm/logs/gpu1_medgemma_endovis.log 2>&1 &
MEDGEMMA_ENDOVIS_PID=$!

echo "   MedGemma+EndoVis PID: $MEDGEMMA_ENDOVIS_PID (GPU 1)"
echo ""

# Wait for all remaining tasks
echo "⏳ Waiting for all remaining tasks to complete..."
wait $MEDGEMMA_KVASIR_PID
MEDGEMMA_KVASIR_EXIT=$?

wait $MEDGEMMA_ENDOVIS_PID
MEDGEMMA_ENDOVIS_EXIT=$?

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALL TASKS COMPLETED                                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Task Results:"
echo "  1. Evaluation (GPU 0): exit code $EVAL_EXIT"
echo "  2. Qwen3-VL+EndoVis (GPU 1): exit code $QWEN3VL_ENDOVIS_EXIT"
echo "  3. MedGemma+Kvasir (GPU 0): exit code $MEDGEMMA_KVASIR_EXIT"
echo "  4. MedGemma+EndoVis (GPU 1): exit code $MEDGEMMA_ENDOVIS_EXIT"
echo ""
echo "End: $(date)"
echo ""

# Check final status
ALL_SUCCESS=true
if [ $EVAL_EXIT -ne 0 ]; then ALL_SUCCESS=false; fi
if [ $QWEN3VL_ENDOVIS_EXIT -ne 0 ]; then ALL_SUCCESS=false; fi
if [ $MEDGEMMA_KVASIR_EXIT -ne 0 ]; then ALL_SUCCESS=false; fi
if [ $MEDGEMMA_ENDOVIS_EXIT -ne 0 ]; then ALL_SUCCESS=false; fi

if [ "$ALL_SUCCESS" = true ]; then
    echo "✅ ALL TASKS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 Results:"
    echo "   • Evaluation: $EVAL_OUTPUT"
    echo "   • Qwen3-VL+EndoVis: $QWEN3VL_ENDOVIS_OUTPUT"
    echo "   • MedGemma+Kvasir: $MEDGEMMA_KVASIR_OUTPUT"
    echo "   • MedGemma+EndoVis: $MEDGEMMA_ENDOVIS_OUTPUT"
    exit 0
else
    echo "⚠️  SOME TASKS FAILED - Check logs for details"
    echo ""
    echo "📁 Logs:"
    echo "   • GPU 0 (Evaluation): slurm/logs/gpu0_evaluation.log"
    echo "   • GPU 1 (Qwen3-VL+EndoVis): slurm/logs/gpu1_qwen3vl_endovis.log"
    echo "   • GPU 0 (MedGemma+Kvasir): slurm/logs/gpu0_medgemma_kvasir.log"
    echo "   • GPU 1 (MedGemma+EndoVis): slurm/logs/gpu1_medgemma_endovis.log"
    exit 1
fi

