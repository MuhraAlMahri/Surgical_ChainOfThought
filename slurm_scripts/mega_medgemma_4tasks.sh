#!/bin/bash
#SBATCH --job-name=mega_medgemma_4tasks
#SBATCH --output=slurm/logs/mega_medgemma_4tasks_%j.out
#SBATCH --error=slurm/logs/mega_medgemma_4tasks_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MEGA JOB: MedGemma-4B 4 Tasks (2 GPUs)                 ║"
echo "║  GPU 0: Zeroshot+COT Evaluations                         ║"
echo "║  GPU 1: Fine-tuned+COT Training                          ║"
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
export CUDA_VISIBLE_DEVICES=0,1

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

# Fine-tuned checkpoints
# Kvasir: Use latest checkpoint from exp1_medgemma4b_instruction
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
    fi
else
    echo "⚠️  Kvasir fine-tuned checkpoint not found, using base model"
    KVASIR_FT_CHECKPOINT="$BASE_MODEL"
fi

# EndoVis: Use exp1_medgemma4b_instruction_r1
ENDOVIS_FT_CHECKPOINT="corrected_1-5_experiments/endovis2018_experiments/models/exp1_medgemma4b_instruction_r1"
if [ ! -d "$ENDOVIS_FT_CHECKPOINT" ]; then
    echo "⚠️  EndoVis fine-tuned checkpoint not found, using base model"
    ENDOVIS_FT_CHECKPOINT="$BASE_MODEL"
else
    echo "✓ Found EndoVis fine-tuned checkpoint: $ENDOVIS_FT_CHECKPOINT"
fi

# COT checkpoints (Zeroshot+COT - trained from base model)
KVASIR_COT_CHECKPOINT="results/medgemma_kvasir_cot_5epochs/checkpoint_epoch_5.pt"
if [ ! -f "$KVASIR_COT_CHECKPOINT" ]; then
    # Try to find alternative location
    ALT_CHECKPOINT=$(find results -name "*medgemma*kvasir*cot*checkpoint_epoch_5.pt" -type f 2>/dev/null | head -1)
    if [ -n "$ALT_CHECKPOINT" ]; then
        KVASIR_COT_CHECKPOINT="$ALT_CHECKPOINT"
        echo "✓ Found Kvasir COT checkpoint: $KVASIR_COT_CHECKPOINT"
    else
        echo "⚠️  Kvasir COT checkpoint not found - will skip Kvasir Zeroshot+COT evaluation"
        KVASIR_COT_CHECKPOINT=""
    fi
fi

ENDOVIS_COT_CHECKPOINT="results/medgemma_endovis_cot_5epochs/checkpoint_epoch_5.pt"
if [ ! -f "$ENDOVIS_COT_CHECKPOINT" ]; then
    # Try to find alternative location
    ALT_CHECKPOINT=$(find results -name "*medgemma*endovis*cot*checkpoint_epoch_5.pt" -type f 2>/dev/null | head -1)
    if [ -n "$ALT_CHECKPOINT" ]; then
        ENDOVIS_COT_CHECKPOINT="$ALT_CHECKPOINT"
        echo "✓ Found EndoVis COT checkpoint: $ENDOVIS_COT_CHECKPOINT"
    else
        echo "⚠️  EndoVis COT checkpoint not found - will skip EndoVis Zeroshot+COT evaluation"
        ENDOVIS_COT_CHECKPOINT=""
    fi
fi

# Data paths
KVASIR_DATA_DIR="corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15"
KVASIR_IMAGE_DIR="datasets/Kvasir-VQA/raw/images"
ENDOVIS_DATA_FILE="corrected_1-5_experiments/datasets/endovis2018_vqa/test.jsonl"
ENDOVIS_IMAGE_DIR="datasets/EndoVis2018/raw/images"

# Training data paths
KVASIR_TRAIN_DATA="corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json"
ENDOVIS_TRAIN_DATA="corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl"

# Instruction files
KVASIR_INSTRUCTIONS="corrected_1-5_experiments/datasets/kvasir_ULTRA_CONDENSED/INSTRUCTIONS_PER_CATEGORY.txt"
ENDOVIS_INSTRUCTIONS="corrected_1-5_experiments/datasets/endovis2018_vqa/INSTRUCTIONS_PER_CATEGORY.txt"

# Question categories
QUESTION_CATEGORIES="question_categories.json"

# Output directories
KVASIR_EVAL_OUTPUT="results/medgemma_kvasir_zeroshot_cot_FULL.json"
ENDOVIS_EVAL_OUTPUT="results/medgemma_endovis_zeroshot_cot_FULL.json"
KVASIR_TRAIN_OUTPUT="results/medgemma_kvasir_finetuned_cot"
ENDOVIS_TRAIN_OUTPUT="results/medgemma_endovis_finetuned_cot"

# Log directories
LOG_DIR="slurm/logs/mega_medgemma_4tasks_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

echo "📋 Configuration:"
echo "   GPU 0 Tasks:"
echo "     1. Zeroshot+COT Kvasir evaluation"
echo "     2. Zeroshot+COT EndoVis evaluation"
echo ""
echo "   GPU 1 Tasks:"
echo "     3. Fine-tuned+COT Kvasir training"
echo "     4. Fine-tuned+COT EndoVis training"
echo ""
echo "   Checkpoints:"
echo "     Kvasir FT: $KVASIR_FT_CHECKPOINT"
echo "     EndoVis FT: $ENDOVIS_FT_CHECKPOINT"
if [ -n "$KVASIR_COT_CHECKPOINT" ]; then
    echo "     Kvasir COT: $KVASIR_COT_CHECKPOINT"
fi
if [ -n "$ENDOVIS_COT_CHECKPOINT" ]; then
    echo "     EndoVis COT: $ENDOVIS_COT_CHECKPOINT"
fi
echo ""

# ============================================================================
# GPU 0: TASK 1 - Zeroshot+COT Kvasir Evaluation
# ============================================================================
run_task1_kvasir_eval() {
    if [ -z "$KVASIR_COT_CHECKPOINT" ]; then
        echo "⚠️  Skipping Task 1 (Kvasir Zeroshot+COT) - checkpoint not found"
        TASK1_EXIT=0
        return
    fi
    
    (
        export CUDA_VISIBLE_DEVICES=0
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 GPU 0 - TASK 1: Zeroshot+COT MedGemma Kvasir Evaluation"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Start time: $(date)"
        echo ""
        echo "Checkpoint: $KVASIR_COT_CHECKPOINT"
        echo "Base model: $BASE_MODEL"
        echo "Data path: $KVASIR_DATA_DIR"
        echo "Output: $KVASIR_EVAL_OUTPUT"
        echo ""
        
        python evaluate_multihead_cot_comprehensive.py \
            --checkpoint "$KVASIR_COT_CHECKPOINT" \
            --base_checkpoint "$BASE_MODEL" \
            --model_name medgemma \
            --dataset kvasir \
            --data_path "$KVASIR_DATA_DIR" \
            --image_base_path "$KVASIR_IMAGE_DIR" \
            --question_categories "$QUESTION_CATEGORIES" \
            --output_file "$KVASIR_EVAL_OUTPUT" \
            --batch_size 1 \
            --use_flexible_matching \
            > "$LOG_DIR/task1_kvasir_eval.log" 2>&1
        
        EXIT_CODE=$?
        echo ""
        echo "Task 1 completed at: $(date)"
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Task 1 (Kvasir Zeroshot+COT) completed successfully"
        else
            echo "❌ Task 1 (Kvasir Zeroshot+COT) failed with exit code: $EXIT_CODE"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    TASK1_EXIT=$?
}

# ============================================================================
# GPU 0: TASK 2 - Zeroshot+COT EndoVis Evaluation
# ============================================================================
run_task2_endovis_eval() {
    if [ -z "$ENDOVIS_COT_CHECKPOINT" ]; then
        echo "⚠️  Skipping Task 2 (EndoVis Zeroshot+COT) - checkpoint not found"
        TASK2_EXIT=0
        return
    fi
    
    (
        export CUDA_VISIBLE_DEVICES=0
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 GPU 0 - TASK 2: Zeroshot+COT MedGemma EndoVis Evaluation"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Start time: $(date)"
        echo ""
        echo "Checkpoint: $ENDOVIS_COT_CHECKPOINT"
        echo "Base model: $BASE_MODEL"
        echo "Data file: $ENDOVIS_DATA_FILE"
        echo "Output: $ENDOVIS_EVAL_OUTPUT"
        echo ""
        
        # Convert JSONL to JSON if needed (create temporary directory with test.json)
        ENDOVIS_TEMP_DIR="/tmp/endovis_eval_${SLURM_JOB_ID}"
        mkdir -p "$ENDOVIS_TEMP_DIR"
        
        # Convert JSONL to JSON format
        python3 << PYTHON_EOF
import json
import sys

jsonl_file = "$ENDOVIS_DATA_FILE"
json_file = "$ENDOVIS_TEMP_DIR/test.json"

data = []
with open(jsonl_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

with open(json_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Converted {len(data)} samples from JSONL to JSON")
PYTHON_EOF
        
        python evaluate_multihead_cot_comprehensive.py \
            --checkpoint "$ENDOVIS_COT_CHECKPOINT" \
            --base_checkpoint "$BASE_MODEL" \
            --model_name medgemma \
            --dataset endovis \
            --data_path "$ENDOVIS_TEMP_DIR" \
            --image_base_path "$ENDOVIS_IMAGE_DIR" \
            --question_categories "$QUESTION_CATEGORIES" \
            --output_file "$ENDOVIS_EVAL_OUTPUT" \
            --batch_size 1 \
            --use_flexible_matching \
            > "$LOG_DIR/task2_endovis_eval.log" 2>&1
        
        EXIT_CODE=$?
        # Cleanup temp directory
        rm -rf "$ENDOVIS_TEMP_DIR"
        
        echo ""
        echo "Task 2 completed at: $(date)"
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Task 2 (EndoVis Zeroshot+COT) completed successfully"
        else
            echo "❌ Task 2 (EndoVis Zeroshot+COT) failed with exit code: $EXIT_CODE"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    TASK2_EXIT=$?
}

# ============================================================================
# GPU 1: TASK 3 - Fine-tuned+COT Kvasir Training
# ============================================================================
run_task3_kvasir_train() {
    (
        export CUDA_VISIBLE_DEVICES=1
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 GPU 1 - TASK 3: Fine-tuned+COT MedGemma Kvasir Training"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Start time: $(date)"
        echo ""
        echo "Base checkpoint: $KVASIR_FT_CHECKPOINT"
        echo "Training data: $KVASIR_TRAIN_DATA"
        echo "Output: $KVASIR_TRAIN_OUTPUT"
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
            --batch_size 1 \
            --grad_accum 16 \
            --weight_decay 0.01 \
            --max_seq_len 3072 \
            --bf16 \
            --gradient_checkpointing \
            --lora_r 4 \
            --lora_alpha 8 \
            > "$LOG_DIR/task3_kvasir_train.log" 2>&1
        
        EXIT_CODE=$?
        echo ""
        echo "Task 3 completed at: $(date)"
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Task 3 (Kvasir Fine-tuned+COT) completed successfully"
        else
            echo "❌ Task 3 (Kvasir Fine-tuned+COT) failed with exit code: $EXIT_CODE"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    TASK3_EXIT=$?
}

# ============================================================================
# GPU 1: TASK 4 - Fine-tuned+COT EndoVis Training
# ============================================================================
run_task4_endovis_train() {
    (
        export CUDA_VISIBLE_DEVICES=1
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 GPU 1 - TASK 4: Fine-tuned+COT MedGemma EndoVis Training"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Start time: $(date)"
        echo ""
        echo "Base checkpoint: $ENDOVIS_FT_CHECKPOINT"
        echo "Training data: $ENDOVIS_TRAIN_DATA"
        echo "Output: $ENDOVIS_TRAIN_OUTPUT"
        echo ""
        
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
            --batch_size 1 \
            --grad_accum 16 \
            --weight_decay 0.01 \
            --max_seq_len 3072 \
            --bf16 \
            --gradient_checkpointing \
            --lora_r 4 \
            --lora_alpha 8 \
            > "$LOG_DIR/task4_endovis_train.log" 2>&1
        
        EXIT_CODE=$?
        echo ""
        echo "Task 4 completed at: $(date)"
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Task 4 (EndoVis Fine-tuned+COT) completed successfully"
        else
            echo "❌ Task 4 (EndoVis Fine-tuned+COT) failed with exit code: $EXIT_CODE"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    TASK4_EXIT=$?
}

# ============================================================================
# RUN ALL TASKS IN PARALLEL (2 GPUs, 4 tasks)
# ============================================================================

echo "Starting all tasks..."
echo "GPU 0: Tasks 1 & 2 (Evaluations - sequential)"
echo "GPU 1: Tasks 3 & 4 (Training - sequential)"
echo ""

# Initialize exit codes
TASK1_EXIT=0
TASK2_EXIT=0
TASK3_EXIT=0
TASK4_EXIT=0

# Start GPU 0 and GPU 1 tasks in parallel
(
    # GPU 0: Run evaluations sequentially
    run_task1_kvasir_eval
    TASK1_EXIT=$?
    run_task2_endovis_eval
    TASK2_EXIT=$?
) &
GPU0_PID=$!

(
    # GPU 1: Run training sequentially
    run_task3_kvasir_train
    TASK3_EXIT=$?
    run_task4_endovis_train
    TASK4_EXIT=$?
) &
GPU1_PID=$!

# Wait for both GPU processes to complete
wait $GPU0_PID
wait $GPU1_PID

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MEGA JOB COMPLETED                                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo ""
echo "Task Results:"
echo "  Task 1 (Kvasir Zeroshot+COT Eval):   Exit code: ${TASK1_EXIT:-N/A}"
echo "  Task 2 (EndoVis Zeroshot+COT Eval):   Exit code: ${TASK2_EXIT:-N/A}"
echo "  Task 3 (Kvasir Fine-tuned+COT Train): Exit code: ${TASK3_EXIT:-N/A}"
echo "  Task 4 (EndoVis Fine-tuned+COT Train): Exit code: ${TASK4_EXIT:-N/A}"
echo ""
echo "Outputs:"
if [ -n "$KVASIR_COT_CHECKPOINT" ]; then
    echo "  $KVASIR_EVAL_OUTPUT"
fi
if [ -n "$ENDOVIS_COT_CHECKPOINT" ]; then
    echo "  $ENDOVIS_EVAL_OUTPUT"
fi
echo "  $KVASIR_TRAIN_OUTPUT"
echo "  $ENDOVIS_TRAIN_OUTPUT"
echo ""
echo "Logs: $LOG_DIR/"
echo ""




