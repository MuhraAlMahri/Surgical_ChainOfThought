#!/bin/bash
#SBATCH --job-name=qwen3vl_2gpu_train
#SBATCH --output=slurm/logs/qwen3vl_2gpu_training_%j.out
#SBATCH --error=slurm/logs/qwen3vl_2gpu_training_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  2-GPU Qwen3-VL Training Job (Optimized)                  ║"
echo "║  GPU 0: Resume Kvasir Fine-tuned+COT (Epochs 2-5)        ║"
echo "║  GPU 1: Train EndoVis Fine-tuned+COT (Epochs 1-5)        ║"
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
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Fine-tuned checkpoints
KVASIR_FT_CHECKPOINT="corrected_1-5_experiments/qlora_experiments/models/exp1_random"
ENDOVIS_FT_CHECKPOINT="corrected_1-5_experiments/endovis2018_experiments/models/exp1_random"

# Resume checkpoint
KVASIR_RESUME_CHECKPOINT="results/qwen3vl_kvasir_finetuned_cot/checkpoint_epoch_1.pt"

# Data paths
KVASIR_TRAIN_DATA="corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json"
KVASIR_IMAGE_DIR="datasets/Kvasir-VQA/raw/images"
ENDOVIS_TRAIN_DATA="corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl"
ENDOVIS_IMAGE_DIR="datasets/EndoVis2018/raw/images"

# Question categories
QUESTION_CATEGORIES="question_categories.json"

# Output directories
KVASIR_TRAIN_OUTPUT="results/qwen3vl_kvasir_finetuned_cot"
ENDOVIS_TRAIN_OUTPUT="results/qwen3vl_endovis_finetuned_cot"

# Log directories
LOG_DIR="slurm/logs/qwen3vl_2gpu_training_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

echo "📋 Configuration:"
echo "   GPU 0: Resume Kvasir Fine-tuned+COT from Epoch 1"
echo "   GPU 1: Train EndoVis Fine-tuned+COT from scratch"
echo ""
echo "   Checkpoints:"
echo "     Kvasir FT Base: $KVASIR_FT_CHECKPOINT"
echo "     Kvasir Resume: $KVASIR_RESUME_CHECKPOINT"
echo "     EndoVis FT Base: $ENDOVIS_FT_CHECKPOINT"
echo ""

# ============================================================================
# GPU 0: RESUME Kvasir Fine-tuned+COT Training (Epochs 2-5)
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
# RUN BOTH TASKS IN PARALLEL
# ============================================================================

echo "Starting both tasks in parallel..."
echo "GPU 0: Resume Kvasir (Epochs 2-5)"
echo "GPU 1: Train EndoVis (Epochs 1-5)"
echo ""

# Initialize exit codes
GPU0_EXIT=0
GPU1_EXIT=0

# Start both tasks in parallel
run_gpu0_kvasir_resume &
GPU0_PID=$!

run_gpu1_endovis_train &
GPU1_PID=$!

# Wait for both tasks to complete
wait $GPU0_PID
wait $GPU1_PID

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  2-GPU JOB COMPLETED                                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo ""
echo "Task Results:"
echo "  GPU 0 (Kvasir Resume):   Exit code: ${GPU0_EXIT:-N/A}"
echo "  GPU 1 (EndoVis Train):    Exit code: ${GPU1_EXIT:-N/A}"
echo ""
echo "Outputs:"
echo "  $KVASIR_TRAIN_OUTPUT"
echo "  $ENDOVIS_TRAIN_OUTPUT"
echo ""
echo "Logs: $LOG_DIR/"
echo ""



