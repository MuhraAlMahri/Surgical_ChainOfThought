#!/bin/bash
#SBATCH --job-name=qwen3vl_2gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/qwen3vl_parallel_2gpu_%j.out
#SBATCH --error=slurm/logs/qwen3vl_parallel_2gpu_%j.err

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Qwen3-VL Parallel Job (2 GPUs, 2 Tasks)                 ║"
echo "║  GPU 0: Evaluate Kvasir Zeroshot+COT                     ║"
echo "║  GPU 1: Train Kvasir Fine-tuned+COT (batch_size=2)       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

# Install peft if needed (to user directory to avoid disk quota)
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PIP_TARGET="/l/users/muhra.almahri/.local/lib/python${PYTHON_VERSION}/site-packages"
export PYTHONPATH="${PIP_TARGET}:${PYTHONPATH}"
mkdir -p ${PIP_TARGET}
pip install --target ${PIP_TARGET} -q peft 2>/dev/null || echo "Note: peft may already be installed"

# Environment
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use /tmp for HuggingFace cache
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Load HuggingFace token
if [ -f ~/.hf_token ]; then
    export HF_TOKEN=$(cat ~/.hf_token)
elif [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
else
    # Set default token for gated repos
    export HF_TOKEN="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"
    echo "⚠️  Using default HF_TOKEN for gated repos"
fi

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create directories
mkdir -p slurm/logs
mkdir -p results

# Log directory for this job
LOG_DIR="slurm/logs/qwen3vl_parallel_2gpu_$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 GPU 0: Evaluate Qwen3-VL Kvasir Zeroshot+COT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Start time: $(date)"
echo ""

CUDA_VISIBLE_DEVICES=0 python evaluate_multihead_cot_comprehensive.py \
    --checkpoint results/qwen3vl_kvasir_cot_5epochs/checkpoint_epoch_5.pt \
    --base_checkpoint Qwen/Qwen3-VL-8B-Instruct \
    --model_name qwen3vl \
    --dataset kvasir \
    --data_path corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15 \
    --image_base_path datasets/Kvasir-VQA/raw/images \
    --question_categories question_categories.json \
    --output_file results/qwen3vl_kvasir_zeroshot_cot_eval.json \
    --use_flexible_matching \
    > "$LOG_DIR/gpu0_kvasir_eval.log" 2>&1 &

GPU0_PID=$!
echo "GPU 0 task started (PID: $GPU0_PID)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 GPU 1: Train Qwen3-VL Kvasir Fine-tuned+COT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  - Base checkpoint: corrected_1-5_experiments/qlora_experiments/models/exp1_random"
echo "  - Batch size: 2 (reduced from 4 to avoid OOM)"
echo "  - Gradient checkpointing: True"
echo "  - Epochs: 5"
echo "  - Output: results/qwen3vl_kvasir_finetuned_cot/"
echo ""

CUDA_VISIBLE_DEVICES=1 python train_multihead_cot.py \
    --model_type qwen3vl \
    --dataset kvasir \
    --base_checkpoint corrected_1-5_experiments/qlora_experiments/models/exp1_random \
    --question_categories question_categories.json \
    --data_path corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json \
    --image_base_path datasets/Kvasir-VQA/raw/images \
    --output_dir results/qwen3vl_kvasir_finetuned_cot \
    --learning_rate 5e-5 \
    --epochs 5 \
    --batch_size 2 \
    --grad_accum 4 \
    --bf16 \
    --gradient_checkpointing \
    --lora_r 4 \
    --lora_alpha 8 \
    --weight_decay 0.01 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    > "$LOG_DIR/gpu1_kvasir_train.log" 2>&1 &

GPU1_PID=$!
echo "GPU 1 task started (PID: $GPU1_PID)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ Waiting for both tasks to complete..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Wait for both tasks
wait $GPU0_PID
GPU0_EXIT=$?

wait $GPU1_PID
GPU1_EXIT=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Task Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $GPU0_EXIT -eq 0 ]; then
    echo "✅ GPU 0 (Kvasir Zeroshot+COT Eval) completed successfully"
else
    echo "❌ GPU 0 (Kvasir Zeroshot+COT Eval) failed with exit code: $GPU0_EXIT"
fi

if [ $GPU1_EXIT -eq 0 ]; then
    echo "✅ GPU 1 (Kvasir Fine-tuned+COT Train) completed successfully"
else
    echo "❌ GPU 1 (Kvasir Fine-tuned+COT Train) failed with exit code: $GPU1_EXIT"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "End time: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

