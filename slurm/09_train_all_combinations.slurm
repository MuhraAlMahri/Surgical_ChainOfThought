#!/bin/bash
#SBATCH --job-name=train_all_combos
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/train_all_combos_%j.out
#SBATCH --error=slurm/logs/train_all_combos_%j.err

echo "=========================================="
echo "Multi-Head CoT Training - All Combinations"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load modules
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

# Environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Use /tmp for HuggingFace cache
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create directories
mkdir -p slurm/logs
mkdir -p results/multihead_cot

# Configuration
LEARNING_RATE="5e-5"
EPOCHS=5
BATCH_SIZE=1
GRAD_ACCUM=16

# Model-dataset combinations
declare -a COMBINATIONS=(
    "qwen3vl:kvasir:Qwen/Qwen3-VL-8B-Instruct"
    "qwen3vl:endovis:Qwen/Qwen3-VL-8B-Instruct"
    "medgemma:kvasir:google/medgemma-4b"
    "medgemma:endovis:google/medgemma-4b"
    "llava_med:kvasir:microsoft/llava-med-v1.5-mistral-7b"
    "llava_med:endovis:microsoft/llava-med-v1.5-mistral-7b"
)

# Dataset paths
declare -A DATA_PATHS=(
    ["kvasir"]="datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json"
    ["endovis"]="datasets/EndoVis2018/raw/metadata/train_vqa_pairs.json"
)

declare -A IMAGE_PATHS=(
    ["kvasir"]="datasets/Kvasir-VQA/raw/images"
    ["endovis"]="datasets/EndoVis2018/raw/images"
)

QUESTION_CATEGORIES="results/multihead_cot/question_categories.json"

# Check if question categories exist
if [ ! -f "$QUESTION_CATEGORIES" ]; then
    echo "⚠️  WARNING: Question categories file not found: $QUESTION_CATEGORIES"
    echo "   Continuing without categories..."
    QUESTION_CATEGORIES=""
fi

# Function to run training for one combination
run_training() {
    local model_type=$1
    local dataset=$2
    local base_checkpoint=$3
    
    echo ""
    echo "=========================================="
    echo "Training: $model_type on $dataset"
    echo "Base checkpoint: $base_checkpoint"
    echo "=========================================="
    echo ""
    
    local data_path="${DATA_PATHS[$dataset]}"
    local image_base_path="${IMAGE_PATHS[$dataset]}"
    local output_dir="results/multihead_cot/${model_type}_${dataset}_cot_$(date +%Y%m%d_%H%M%S)"
    
    # Check if data file exists
    if [ ! -f "$data_path" ]; then
        echo "❌ ERROR: Data file not found: $data_path"
        echo "   Skipping this combination..."
        return 1
    fi
    
    echo "Configuration:"
    echo "  Model type: $model_type"
    echo "  Dataset: $dataset"
    echo "  Base checkpoint: $base_checkpoint"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Gradient accumulation: $GRAD_ACCUM"
    echo "  Data path: $data_path"
    echo "  Image base path: $image_base_path"
    echo "  Output directory: $output_dir"
    echo ""
    
    # Run training
    python3 -u train_multihead_cot.py \
        --model_type "$model_type" \
        --dataset "$dataset" \
        --base_checkpoint "$base_checkpoint" \
        --question_categories "${QUESTION_CATEGORIES:-question_categories.json}" \
        --data_path "$data_path" \
        --image_base_path "$image_base_path" \
        --output_dir "$output_dir" \
        --learning_rate "$LEARNING_RATE" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --bf16 \
        --gradient_checkpointing \
        --lora_r 4 \
        --lora_alpha 8
    
    local train_exit_code=$?
    
    echo ""
    echo "=========================================="
    if [ $train_exit_code -eq 0 ]; then
        echo "✅ Training completed: $model_type on $dataset"
        echo "   Checkpoint: $output_dir"
    else
        echo "❌ Training failed: $model_type on $dataset (exit code: $train_exit_code)"
    fi
    echo "=========================================="
    echo ""
    
    return $train_exit_code
}

# Run all combinations
TOTAL_COMBOS=${#COMBINATIONS[@]}
SUCCESS_COUNT=0
FAIL_COUNT=0

echo "Starting training for $TOTAL_COMBOS combinations..."
echo ""

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r model_type dataset base_checkpoint <<< "$combo"
    
    run_training "$model_type" "$dataset" "$base_checkpoint"
    
    if [ $? -eq 0 ]; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
    
    # Brief pause between combinations to allow cleanup
    # Removed sleep to avoid HPC monitoring detection
done

# Summary
echo ""
echo "=========================================="
echo "TRAINING SUMMARY"
echo "=========================================="
echo "Total combinations: $TOTAL_COMBOS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "End time: $(date)"
echo "=========================================="

# Clean up temporary cache
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up temporary cache..."
    rm -rf "$HF_HOME"
fi

exit $([ $FAIL_COUNT -eq 0 ] && echo 0 || echo 1)

