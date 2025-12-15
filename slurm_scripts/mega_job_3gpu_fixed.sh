#!/bin/bash
#SBATCH --job-name=mega_cot_3gpu
#SBATCH --output=slurm/logs/mega_cot_3gpu_%j.out
#SBATCH --error=slurm/logs/mega_cot_3gpu_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --mem=240G
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MEGA JOB: FIX EVAL + TRAIN + EVAL (3 GPUs)             ║"
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

# Check and upgrade PEFT version
echo "🔧 Checking PEFT version..."
PEFT_INSTALL_DIR="${SLURM_SUBMIT_DIR}/.hf_cache/peft_packages"
mkdir -p "$PEFT_INSTALL_DIR"

# First check if PEFT is already installed and meets version requirement
PEFT_OK=false
if python3 -c "import peft; from packaging import version; v=peft.__version__; exit(0 if version.parse(v) >= version.parse('0.18.0') else 1)" 2>/dev/null; then
    PEFT_VERSION=$(python3 -c "import peft; print(peft.__version__)" 2>/dev/null)
    echo "✓ PEFT already installed: version $PEFT_VERSION (meets requirement >=0.18.0)"
    PEFT_OK=true
else
    # Try to install to workspace
    echo "   PEFT not found or version < 0.18.0, installing to workspace..."
    export PYTHONPATH="${PEFT_INSTALL_DIR}:${PYTHONPATH}"
    pip install --target="$PEFT_INSTALL_DIR" --upgrade peft>=0.18.0 2>&1 | tail -10
    
    # Verify installation
    if python3 -c "import sys; sys.path.insert(0, '${PEFT_INSTALL_DIR}'); import peft; print(f'✓ PEFT installed: {peft.__version__}')" 2>/dev/null; then
        echo "✓ PEFT installed to workspace: $PEFT_INSTALL_DIR"
        PEFT_OK=true
    else
        echo "⚠️  PEFT installation to workspace failed, trying conda environment..."
        # Last resort: try installing to conda (may fail due to quota, but worth trying)
        pip install --upgrade peft>=0.18.0 2>&1 | tail -5 || echo "   Installation failed (quota issue expected)"
        if python3 -c "import peft; from packaging import version; v=peft.__version__; exit(0 if version.parse(v) >= version.parse('0.18.0') else 1)" 2>/dev/null; then
            PEFT_VERSION=$(python3 -c "import peft; print(peft.__version__)" 2>/dev/null)
            echo "✓ PEFT available in conda: version $PEFT_VERSION"
            PEFT_OK=true
        fi
    fi
fi

if [ "$PEFT_OK" = false ]; then
    echo "⚠️  WARNING: PEFT >= 0.18.0 not available. Training may fail."
fi
echo ""

# Fix HuggingFace authentication
echo "🔐 Setting up HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN not set"
else
    echo "✓ HuggingFace token found"
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi
echo ""

cd $SLURM_SUBMIT_DIR

# ============================================================================
# TASK 1: FIX ZEROSHOT+COT EVALUATION (GPU 0) - 30 minutes
# ============================================================================

run_fix_evaluation() {
    local gpu_id=$1
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 GPU $gpu_id: Fixing Zeroshot+COT Evaluation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start time: $(date)"
    
    INPUT_FILE="${SLURM_SUBMIT_DIR}/results/eval_epoch5_qwen3vl_kvasir_FIXED/evaluation_epoch5_qwen3vl_kvasir.json"
    OUTPUT_FILE="${SLURM_SUBMIT_DIR}/results/eval_epoch5_qwen3vl_kvasir_FIXED/evaluation_epoch5_qwen3vl_kvasir_FIXED.json"
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "⚠️  Input file not found: $INPUT_FILE"
        echo "   Trying alternative location..."
        INPUT_FILE="${SLURM_SUBMIT_DIR}/results/eval_epoch5_qwen3vl_kvasir/evaluation_epoch5_qwen3vl_kvasir.json"
    fi
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "❌ Input file not found. Skipping evaluation fix."
        return 1
    fi
    
    CUDA_VISIBLE_DEVICES=$gpu_id python fix_evaluation_zeroshot_cot.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE"
    
    local exit_code=$?
    echo ""
    echo "End time: $(date)"
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: Evaluation fixed"
    else
        echo "❌ FAILED: Evaluation fix (exit code: $exit_code)"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    return $exit_code
}

# ============================================================================
# TASK 2: TRAIN FINE-TUNED+COT (GPU 1) - 6-8 hours
# ============================================================================

run_train_finetuned_cot() {
    local gpu_id=$1
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 GPU $gpu_id: Training Fine-tuned+COT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start time: $(date)"
    
    # Ensure PYTHONPATH includes PEFT installation directory
    PEFT_INSTALL_DIR="${SLURM_SUBMIT_DIR}/.hf_cache/peft_packages"
    export PYTHONPATH="${PEFT_INSTALL_DIR}:${PYTHONPATH}"
    echo "   PYTHONPATH set to include: $PEFT_INSTALL_DIR"
    
    # Find fine-tuned checkpoint (92.79% accuracy model)
    # Try multiple possible locations
    POSSIBLE_PATHS=(
        "${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/qlora_experiments/models/qwen3vl_kvasir_instruction/best_model"
        "${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/qlora_experiments/models/exp1_qwen3vl_kvasir/best_model"
        "${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/qlora_experiments/models/exp1_random_instruction_r1/best_model"
        "${SLURM_SUBMIT_DIR}/checkpoints/qwen3vl_kvasir_finetuned"
    )
    
    BASE_CHECKPOINT=""
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -d "$path" ]; then
            # Check if it has adapter files (LoRA) or model files
            if [ -f "$path/adapter_config.json" ] || [ -f "$path/config.json" ] || [ -f "$path/pytorch_model.bin" ] || [ -f "$path/model.safetensors" ]; then
                BASE_CHECKPOINT="$path"
                echo "✓ Found fine-tuned checkpoint: $BASE_CHECKPOINT"
                break
            fi
        fi
    done
    
    # If still not found, try to find any qwen3vl kvasir model
    if [ -z "$BASE_CHECKPOINT" ]; then
        FOUND_CHECKPOINT=$(find "${SLURM_SUBMIT_DIR}/corrected_1-5_experiments" -type d -path "*/models/*qwen3vl*kvasir*" 2>/dev/null | head -1)
        if [ -n "$FOUND_CHECKPOINT" ] && [ -d "$FOUND_CHECKPOINT" ]; then
            # Check for best_model subdirectory
            if [ -d "$FOUND_CHECKPOINT/best_model" ]; then
                BASE_CHECKPOINT="$FOUND_CHECKPOINT/best_model"
            else
                BASE_CHECKPOINT="$FOUND_CHECKPOINT"
            fi
            echo "✓ Found fine-tuned checkpoint: $BASE_CHECKPOINT"
        fi
    fi
    
    # Final fallback
    if [ -z "$BASE_CHECKPOINT" ] || [ ! -d "$BASE_CHECKPOINT" ]; then
        echo "⚠️  Fine-tuned checkpoint not found at expected locations"
        echo "   Will use base model (this will train Zeroshot+COT, not Fine-tuned+COT)"
        echo "   To train Fine-tuned+COT, please ensure the fine-tuned checkpoint exists"
        BASE_CHECKPOINT="Qwen/Qwen3-VL-8B-Instruct"
    fi
    
    DATA_PATH="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json"
    if [ ! -f "$DATA_PATH" ]; then
        DATA_PATH="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_baseline_image_level_70_15_15/train.json"
    fi
    
    IMAGE_PATH="${SLURM_SUBMIT_DIR}/datasets/Kvasir-VQA/raw/images"
    if [ ! -d "$IMAGE_PATH" ]; then
        IMAGE_PATH="${SLURM_SUBMIT_DIR}/datasets/kvasir/images"
    fi
    
    OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/qwen3vl_kvasir_finetuned_cot_5epochs"
    QUESTION_CATEGORIES="${SLURM_SUBMIT_DIR}/question_categories.json"
    
    if [ ! -f "$QUESTION_CATEGORIES" ]; then
        if [ -f "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" ]; then
            cp "${SLURM_SUBMIT_DIR}/results/multihead_cot/question_categories.json" "$QUESTION_CATEGORIES"
        else
            echo '{"kvasir": {}, "endovis": {}}' > "$QUESTION_CATEGORIES"
        fi
    fi
    
    echo "   Base checkpoint: $BASE_CHECKPOINT"
    echo "   Data: $DATA_PATH"
    echo "   Images: $IMAGE_PATH"
    echo "   Output: $OUTPUT_DIR"
    
    # Ensure PYTHONPATH is set for this subprocess
    PEFT_INSTALL_DIR="${SLURM_SUBMIT_DIR}/.hf_cache/peft_packages"
    export PYTHONPATH="${PEFT_INSTALL_DIR}:${PYTHONPATH}"
    
    # Verify PEFT is accessible
    echo "   Verifying PEFT import..."
    python3 -c "import sys; sys.path.insert(0, '${PEFT_INSTALL_DIR}'); import peft; print(f'✓ PEFT version: {peft.__version__}')" 2>&1 || echo "   ⚠️  PEFT import test failed"
    
    # Use existing train_multihead_cot.py with fine-tuned checkpoint
    CUDA_VISIBLE_DEVICES=$gpu_id env PYTHONPATH="${PEFT_INSTALL_DIR}:${PYTHONPATH}" python train_multihead_cot.py \
        --model_type "qwen3vl" \
        --dataset "kvasir" \
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
# TASK 3: EVALUATE FINE-TUNED+COT (GPU 2) - 2 hours (waits for Task 2)
# ============================================================================

run_evaluate_finetuned_cot() {
    local gpu_id=$1
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 GPU $gpu_id: Evaluating Fine-tuned+COT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start time: $(date)"
    
    CHECKPOINT="${SLURM_SUBMIT_DIR}/results/qwen3vl_kvasir_finetuned_cot_5epochs/checkpoint_epoch_5.pt"
    BASE_CHECKPOINT="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/qlora_experiments/models/qwen3vl_kvasir_instruction/best_model"
    if [ ! -d "$BASE_CHECKPOINT" ]; then
        BASE_CHECKPOINT="Qwen/Qwen3-VL-8B-Instruct"
    fi
    
    TEST_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/test.json"
    if [ ! -f "$TEST_DATA" ]; then
        TEST_DATA="${SLURM_SUBMIT_DIR}/corrected_1-5_experiments/datasets/kvasir_baseline_image_level_70_15_15/test.json"
    fi
    
    IMAGE_PATH="${SLURM_SUBMIT_DIR}/datasets/Kvasir-VQA/raw/images"
    if [ ! -d "$IMAGE_PATH" ]; then
        IMAGE_PATH="${SLURM_SUBMIT_DIR}/datasets/kvasir/images"
    fi
    
    OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/eval_finetuned_cot_qwen3vl_kvasir"
    QUESTION_CATEGORIES="${SLURM_SUBMIT_DIR}/question_categories.json"
    
    echo "   Checkpoint: $CHECKPOINT"
    echo "   Base checkpoint: $BASE_CHECKPOINT"
    echo "   Test data: $TEST_DATA"
    echo "   Output: $OUTPUT_DIR"
    
    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "❌ Checkpoint not found: $CHECKPOINT"
        echo "   Training may not have completed yet or failed."
        return 1
    fi
    
    echo "   ✓ Checkpoint found, starting evaluation..."
    
    # Use existing evaluation script
    CUDA_VISIBLE_DEVICES=$gpu_id python evaluate_epoch3_checkpoint_FIXED.py \
        --checkpoint "$CHECKPOINT" \
        --base_checkpoint "$BASE_CHECKPOINT" \
        --model_type "qwen3vl" \
        --dataset "kvasir" \
        --test_data "$TEST_DATA" \
        --image_base_path "$IMAGE_PATH" \
        --question_categories "$QUESTION_CATEGORIES" \
        --output "$OUTPUT_DIR"
    
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

# ============================================================================
# EXECUTION PLAN
# ============================================================================

echo "📋 Execution Plan:"
echo "   Task 1 (GPU 0): Fix Zeroshot+COT evaluation (~30 min)"
echo "   Task 2 (GPU 1): Train Fine-tuned+COT (~6-8 hours)"
echo "   Task 3 (GPU 2): Evaluate Fine-tuned+COT (~2 hours, waits for Task 2)"
echo ""

# Phase 1: Start Task 1 and Task 2 in parallel
echo "⏳ Phase 1: Starting Task 1 (GPU 0) and Task 2 (GPU 1) in parallel..."
echo ""

run_fix_evaluation 0 > slurm/logs/gpu0_fix_eval.log 2>&1 &
FIX_EVAL_PID=$!

run_train_finetuned_cot 1 > slurm/logs/gpu1_train_ft_cot.log 2>&1 &
TRAIN_FT_COT_PID=$!

echo "   Fix Evaluation PID: $FIX_EVAL_PID (GPU 0)"
echo "   Train Fine-tuned+COT PID: $TRAIN_FT_COT_PID (GPU 1)"
echo ""

# Wait for Task 1 to complete
wait $FIX_EVAL_PID
FIX_EVAL_EXIT=$?

if [ $FIX_EVAL_EXIT -eq 0 ]; then
    echo "✅ Task 1 (Fix Evaluation) completed successfully"
else
    echo "⚠️  Task 1 (Fix Evaluation) failed (exit code: $FIX_EVAL_EXIT)"
fi
echo ""

# Phase 2: Wait for Task 2, then start Task 3
echo "⏳ Phase 2: Waiting for Task 2 (Training) to complete, then starting Task 3 (Evaluation)..."
echo ""

wait $TRAIN_FT_COT_PID
TRAIN_FT_COT_EXIT=$?

if [ $TRAIN_FT_COT_EXIT -eq 0 ]; then
    echo "✅ Task 2 (Train Fine-tuned+COT) completed successfully"
else
    echo "⚠️  Task 2 (Train Fine-tuned+COT) failed (exit code: $TRAIN_FT_COT_EXIT)"
fi
echo ""

# Start Task 3
run_evaluate_finetuned_cot 2 > slurm/logs/gpu2_eval_ft_cot.log 2>&1 &
EVAL_FT_COT_PID=$!

echo "   Evaluate Fine-tuned+COT PID: $EVAL_FT_COT_PID (GPU 2)"
echo ""

# Wait for Task 3
wait $EVAL_FT_COT_PID
EVAL_FT_COT_EXIT=$?

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALL TASKS COMPLETED                                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Task Results:"
echo "  1. Fix Zeroshot+COT Evaluation (GPU 0): exit code $FIX_EVAL_EXIT"
echo "  2. Train Fine-tuned+COT (GPU 1): exit code $TRAIN_FT_COT_EXIT"
echo "  3. Evaluate Fine-tuned+COT (GPU 2): exit code $EVAL_FT_COT_EXIT"
echo ""
echo "End: $(date)"
echo ""

# Check final status
ALL_SUCCESS=true
if [ $FIX_EVAL_EXIT -ne 0 ]; then ALL_SUCCESS=false; fi
if [ $TRAIN_FT_COT_EXIT -ne 0 ]; then ALL_SUCCESS=false; fi
if [ $EVAL_FT_COT_EXIT -ne 0 ]; then ALL_SUCCESS=false; fi

if [ "$ALL_SUCCESS" = true ]; then
    echo "✅ ALL TASKS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📊 Results:"
    echo "   • Fixed Zeroshot+COT evaluation: results/eval_epoch5_qwen3vl_kvasir_FIXED/evaluation_epoch5_qwen3vl_kvasir_FIXED.json"
    echo "   • Fine-tuned+COT training: results/qwen3vl_kvasir_finetuned_cot_5epochs/"
    echo "   • Fine-tuned+COT evaluation: results/eval_finetuned_cot_qwen3vl_kvasir/"
    exit 0
else
    echo "⚠️  SOME TASKS FAILED - Check logs for details"
    echo ""
    echo "📁 Logs:"
    echo "   • GPU 0 (Fix Eval): slurm/logs/gpu0_fix_eval.log"
    echo "   • GPU 1 (Train): slurm/logs/gpu1_train_ft_cot.log"
    echo "   • GPU 2 (Eval): slurm/logs/gpu2_eval_ft_cot.log"
    exit 1
fi

