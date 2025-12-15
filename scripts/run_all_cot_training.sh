#!/bin/bash
# Master script to submit all multi-head CoT training jobs
# Usage: ./scripts/run_all_cot_training.sh

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create logs directory
mkdir -p slurm/logs

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     MULTI-HEAD COT TRAINING PIPELINE                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Fine-tuned baseline checkpoint paths
# Using best available checkpoints or base HuggingFace models
declare -A BASE_CHECKPOINTS=(
    # Qwen3-VL - Using base model (fine-tuned checkpoints can be added if available)
    ["qwen3vl_kvasir"]="Qwen/Qwen3-VL-8B-Instruct"
    ["qwen3vl_endovis"]="Qwen/Qwen3-VL-8B-Instruct"
    
    # MedGemma - Using base model
    ["medgemma_kvasir"]="google/medgemma-4b"
    ["medgemma_endovis"]="google/medgemma-4b"
    
    # LLaVA-Med - Using fine-tuned checkpoints where available
    ["llava_med_kvasir"]="corrected_1-5_experiments/qlora_experiments/models/llava_med_kvasir_instruction/best_model"
    ["llava_med_endovis"]="corrected_1-5_experiments/qlora_experiments/models/llava_med_endovis_instruction/best_model"
)

# Dataset paths
declare -A DATA_PATHS=(
    ["kvasir"]="datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json"
    ["endovis"]="corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl"
)

declare -A IMAGE_PATHS=(
    ["kvasir"]="datasets/Kvasir-VQA/raw/images"
    ["endovis"]="datasets/EndoVis2018/raw/images"
)

# Model-dataset combinations
declare -a COMBINATIONS=(
    "qwen3vl:kvasir"
    "qwen3vl:endovis"
    "medgemma:kvasir"
    "medgemma:endovis"
    "llava_med:kvasir"
    "llava_med:endovis"
)

echo "Step 1: Submitting all training jobs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

declare -a JOB_IDS=()

for combo in "${COMBINATIONS[@]}"; do
    IFS=':' read -r model_type dataset <<< "$combo"
    
    base_checkpoint="${BASE_CHECKPOINTS[${model_type}_${dataset}]}"
    data_path="${DATA_PATHS[$dataset]}"
    image_path="${IMAGE_PATHS[$dataset]}"
    
    if [ -z "$base_checkpoint" ]; then
        echo "⚠️  Skipping ${model_type}_${dataset}: No base checkpoint configured"
        continue
    fi
    
    echo "Submitting: ${model_type} on ${dataset}"
    echo "  Base checkpoint: $base_checkpoint"
    echo "  Data path: $data_path"
    echo "  Image path: $image_path"
    
    # Submit job
    job_output=$(sbatch slurm/07_train_multihead_cot.slurm \
        "$model_type" \
        "$dataset" \
        "$base_checkpoint" \
        "5e-5" \
        "5" \
        "1" \
        "16")
    
    job_id=$(echo "$job_output" | awk '{print $4}')
    JOB_IDS+=("$job_id")
    
    echo "  ✅ Job ID: $job_id"
    echo ""
    
    sleep 2
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All training jobs submitted!"
echo ""
echo "Submitted jobs:"
for i in "${!COMBINATIONS[@]}"; do
    combo="${COMBINATIONS[$i]}"
    job_id="${JOB_IDS[$i]}"
    echo "  • $combo: Job $job_id"
done
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs/train_multihead_cot_*.out"
echo ""
echo "Once all jobs complete, verify training:"
echo "  python scripts/check_trained_models.py"
echo ""
echo "Then run evaluation:"
echo "  bash scripts/evaluate_cot_all.sh"
echo ""
echo "Expected training time: 2-3 hours per job"
echo "Total time: ~12-18 hours (if run in parallel on multiple GPUs)"
echo "╚══════════════════════════════════════════════════════════╝"

