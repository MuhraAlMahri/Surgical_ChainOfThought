#!/bin/bash
#SBATCH --job-name=mega_qwen3vl_4gpu
#SBATCH --output=slurm/logs/mega_qwen3vl_4gpu_4tasks_%j.out
#SBATCH --error=slurm/logs/mega_qwen3vl_4gpu_4tasks_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MEGA JOB: Qwen3-VL 4 Tasks (4 GPUs)                     ║"
echo "║  GPU 0: Resume Kvasir Fine-tuned+COT (Epochs 2-5)         ║"
echo "║  GPU 1: Train EndoVis Fine-tuned+COT (Epochs 1-5)         ║"
echo "║  GPU 2: Evaluate Kvasir Zeroshot+COT                      ║"
echo "║  GPU 3: Evaluate EndoVis Zeroshot+COT                     ║"
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
export CUDA_VISIBLE_DEVICES=0,1,2,3

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
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Fine-tuned checkpoints
KVASIR_FT_CHECKPOINT="corrected_1-5_experiments/qlora_experiments/models/exp1_random"
ENDOVIS_FT_CHECKPOINT="corrected_1-5_experiments/endovis2018_experiments/models/exp1_random"

# Resume checkpoint
KVASIR_RESUME_CHECKPOINT="results/qwen3vl_kvasir_finetuned_cot/checkpoint_epoch_1.pt"

# COT checkpoints (Zeroshot+COT)
KVASIR_COT_CHECKPOINT="results/qwen3vl_kvasir_cot_5epochs/checkpoint_epoch_5.pt"
ENDOVIS_COT_CHECKPOINT="results/qwen3vl_endovis_cot_5epochs/checkpoint_epoch_5.pt"

# Data paths
KVASIR_TRAIN_DATA="corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json"
KVASIR_DATA_DIR="corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15"
KVASIR_IMAGE_DIR="datasets/Kvasir-VQA/raw/images"
ENDOVIS_TRAIN_DATA="corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl"
ENDOVIS_DATA_FILE="corrected_1-5_experiments/datasets/endovis2018_vqa/test.jsonl"
ENDOVIS_IMAGE_DIR="datasets/EndoVis2018/raw/images"

# Question categories
QUESTION_CATEGORIES="question_categories.json"

# Output directories
KVASIR_TRAIN_OUTPUT="results/qwen3vl_kvasir_finetuned_cot"
ENDOVIS_TRAIN_OUTPUT="results/qwen3vl_endovis_finetuned_cot"
KVASIR_EVAL_OUTPUT="results/qwen3vl_kvasir_zeroshot_cot_FULL.json"
ENDOVIS_EVAL_OUTPUT="results/qwen3vl_endovis_zeroshot_cot_FULL.json"

# Log directories
LOG_DIR="slurm/logs/mega_qwen3vl_4gpu_4tasks_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

echo "📋 Configuration:"
echo "   GPU 0: Resume Kvasir Fine-tuned+COT (Epochs 2-5)"
echo "   GPU 1: Train EndoVis Fine-tuned+COT (Epochs 1-5)"
echo "   GPU 2: Evaluate Kvasir Zeroshot+COT"
echo "   GPU 3: Evaluate EndoVis Zeroshot+COT"
echo ""

# ============================================================================
# GPU 0: Resume Kvasir Fine-tuned+COT Training (Epochs 2-5)
# ============================================================================
run_gpu0_kvasir_resume() {
    (
        export CUDA_VISIBLE_DEVICES=0
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 GPU 0: Resume Kvasir Fine-tuned+COT (Epochs 2-5)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Start time: $(date)"
        echo ""
        echo "Base checkpoint: $KVASIR_FT_CHECKPOINT"
        echo "Resume from: $KVASIR_RESUME_CHECKPOINT"
        echo "Training data: $KVASIR_TRAIN_DATA"
        echo "Output: $KVASIR_TRAIN_OUTPUT"
        echo ""
        
        python train_multihead_cot.py \
            --model_type qwen3vl \
            --dataset kvasir \
            --base_checkpoint "$KVASIR_FT_CHECKPOINT" \
            --resume_from_checkpoint "$KVASIR_RESUME_CHECKPOINT" \
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
            --dataloader_pin_memory \
            > "$LOG_DIR/gpu0_kvasir_resume.log" 2>&1
        
        EXIT_CODE=$?
        echo ""
        echo "GPU 0 task completed at: $(date)"
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ GPU 0 (Kvasir Resume) completed successfully"
        else
            echo "❌ GPU 0 (Kvasir Resume) failed with exit code: $EXIT_CODE"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    GPU0_EXIT=$?
}

# ============================================================================
# GPU 1: Train EndoVis Fine-tuned+COT (Epochs 1-5)
# ============================================================================
run_gpu1_endovis_train() {
    (
        export CUDA_VISIBLE_DEVICES=1
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 GPU 1: Train EndoVis Fine-tuned+COT (Epochs 1-5)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Start time: $(date)"
        echo ""
        echo "Base checkpoint: $ENDOVIS_FT_CHECKPOINT"
        echo "Training data: $ENDOVIS_TRAIN_DATA"
        echo "Output: $ENDOVIS_TRAIN_OUTPUT"
        echo ""
        
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
            --batch_size 4 \
            --grad_accum 4 \
            --weight_decay 0.01 \
            --max_seq_len 3072 \
            --bf16 \
            --lora_r 4 \
            --lora_alpha 8 \
            --dataloader_num_workers 4 \
            --dataloader_pin_memory \
            > "$LOG_DIR/gpu1_endovis_train.log" 2>&1
        
        EXIT_CODE=$?
        echo ""
        echo "GPU 1 task completed at: $(date)"
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ GPU 1 (EndoVis Train) completed successfully"
        else
            echo "❌ GPU 1 (EndoVis Train) failed with exit code: $EXIT_CODE"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    GPU1_EXIT=$?
}

# ============================================================================
# GPU 2: Evaluate Kvasir Zeroshot+COT
# ============================================================================
run_gpu2_kvasir_eval() {
    (
        export CUDA_VISIBLE_DEVICES=2
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 GPU 2: Evaluate Kvasir Zeroshot+COT"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Start time: $(date)"
        echo ""
        
        if [ ! -f "$KVASIR_COT_CHECKPOINT" ]; then
            echo "⚠️  Skipping GPU 2 (Kvasir Zeroshot+COT) - checkpoint not found: $KVASIR_COT_CHECKPOINT"
            GPU2_EXIT=0
            return
        fi
        
        echo "Checkpoint: $KVASIR_COT_CHECKPOINT"
        echo "Base model: $BASE_MODEL"
        echo "Data path: $KVASIR_DATA_DIR"
        echo "Output: $KVASIR_EVAL_OUTPUT"
        echo ""
        
        python evaluate_multihead_cot_comprehensive.py \
            --checkpoint "$KVASIR_COT_CHECKPOINT" \
            --base_checkpoint "$BASE_MODEL" \
            --model_name qwen3vl \
            --dataset kvasir \
            --data_path "$KVASIR_DATA_DIR" \
            --image_base_path "$KVASIR_IMAGE_DIR" \
            --question_categories "$QUESTION_CATEGORIES" \
            --output_file "$KVASIR_EVAL_OUTPUT" \
            --batch_size 1 \
            --use_flexible_matching \
            > "$LOG_DIR/gpu2_kvasir_eval.log" 2>&1
        
        EXIT_CODE=$?
        echo ""
        echo "GPU 2 task completed at: $(date)"
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ GPU 2 (Kvasir Zeroshot+COT Eval) completed successfully"
        else
            echo "❌ GPU 2 (Kvasir Zeroshot+COT Eval) failed with exit code: $EXIT_CODE"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    GPU2_EXIT=$?
}

# ============================================================================
# GPU 3: Evaluate EndoVis Zeroshot+COT
# ============================================================================
run_gpu3_endovis_eval() {
    (
        export CUDA_VISIBLE_DEVICES=3
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 GPU 3: Evaluate EndoVis Zeroshot+COT"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Start time: $(date)"
        echo ""
        
        if [ ! -f "$ENDOVIS_COT_CHECKPOINT" ]; then
            echo "⚠️  Skipping GPU 3 (EndoVis Zeroshot+COT) - checkpoint not found: $ENDOVIS_COT_CHECKPOINT"
            GPU3_EXIT=0
            return
        fi
        
        # Convert JSONL to JSON if needed
        ENDOVIS_TEMP_DIR="/tmp/endovis_eval_${SLURM_JOB_ID}"
        mkdir -p "$ENDOVIS_TEMP_DIR"
        
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
        
        echo "Checkpoint: $ENDOVIS_COT_CHECKPOINT"
        echo "Base model: $BASE_MODEL"
        echo "Data path: $ENDOVIS_TEMP_DIR"
        echo "Output: $ENDOVIS_EVAL_OUTPUT"
        echo ""
        
        python evaluate_multihead_cot_comprehensive.py \
            --checkpoint "$ENDOVIS_COT_CHECKPOINT" \
            --base_checkpoint "$BASE_MODEL" \
            --model_name qwen3vl \
            --dataset endovis \
            --data_path "$ENDOVIS_TEMP_DIR" \
            --image_base_path "$ENDOVIS_IMAGE_DIR" \
            --question_categories "$QUESTION_CATEGORIES" \
            --output_file "$ENDOVIS_EVAL_OUTPUT" \
            --batch_size 1 \
            --use_flexible_matching \
            > "$LOG_DIR/gpu3_endovis_eval.log" 2>&1
        
        EXIT_CODE=$?
        # Cleanup temp directory
        rm -rf "$ENDOVIS_TEMP_DIR"
        
        echo ""
        echo "GPU 3 task completed at: $(date)"
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ GPU 3 (EndoVis Zeroshot+COT Eval) completed successfully"
        else
            echo "❌ GPU 3 (EndoVis Zeroshot+COT Eval) failed with exit code: $EXIT_CODE"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    GPU3_EXIT=$?
}

# ============================================================================
# RUN ALL 4 TASKS IN PARALLEL
# ============================================================================

echo "Starting all 4 tasks in parallel..."
echo "GPU 0: Resume Kvasir (Epochs 2-5)"
echo "GPU 1: Train EndoVis (Epochs 1-5)"
echo "GPU 2: Evaluate Kvasir Zeroshot+COT"
echo "GPU 3: Evaluate EndoVis Zeroshot+COT"
echo ""

# Initialize exit codes
GPU0_EXIT=0
GPU1_EXIT=0
GPU2_EXIT=0
GPU3_EXIT=0

# Start all tasks in parallel
run_gpu0_kvasir_resume &
GPU0_PID=$!

run_gpu1_endovis_train &
GPU1_PID=$!

run_gpu2_kvasir_eval &
GPU2_PID=$!

run_gpu3_endovis_eval &
GPU3_PID=$!

# Wait for all tasks to complete
wait $GPU0_PID
wait $GPU1_PID
wait $GPU2_PID
wait $GPU3_PID

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MEGA JOB COMPLETED (4 GPUs, 4 Tasks)                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo ""
echo "Task Results:"
echo "  GPU 0 (Kvasir Resume):        Exit code: ${GPU0_EXIT:-N/A}"
echo "  GPU 1 (EndoVis Train):         Exit code: ${GPU1_EXIT:-N/A}"
echo "  GPU 2 (Kvasir Zeroshot+COT):   Exit code: ${GPU2_EXIT:-N/A}"
echo "  GPU 3 (EndoVis Zeroshot+COT):  Exit code: ${GPU3_EXIT:-N/A}"
echo ""
echo "Outputs:"
echo "  $KVASIR_TRAIN_OUTPUT"
echo "  $ENDOVIS_TRAIN_OUTPUT"
echo "  $KVASIR_EVAL_OUTPUT"
echo "  $ENDOVIS_EVAL_OUTPUT"
echo ""
echo "Logs: $LOG_DIR/"
echo ""



