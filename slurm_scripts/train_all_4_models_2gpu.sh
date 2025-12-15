#!/bin/bash
#SBATCH --job-name=all_4_cot_2gpu
#SBATCH --output=slurm/logs/all_4_cot_2gpu_%j.out
#SBATCH --error=slurm/logs/all_4_cot_2gpu_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MULTI-HEAD COT TRAINING - ALL 4 MODEL-DATASET COMBOS   ║"
echo "║  Using 2 GPUs                                            ║"
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

# Function to run training job on a specific GPU
run_training() {
    local gpu_id=$1
    local model=$2
    local dataset=$3
    local checkpoint=$4
    local output_dir=$5
    local job_name="${model}_${dataset}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Starting: $job_name on GPU $gpu_id"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start time: $(date)"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python train_multihead_cot_v3.py \
        --model_name $model \
        --dataset $dataset \
        --checkpoint_path $checkpoint \
        --output_dir $output_dir \
        --device cuda
    
    local exit_code=$?
    echo ""
    echo "End time: $(date)"
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: $job_name"
    else
        echo "❌ FAILED: $job_name (exit code: $exit_code)"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    return $exit_code
}

# Run jobs in parallel on 2 GPUs
# GPU 0: Qwen3-VL + Kvasir and Qwen3-VL + EndoVis (sequential)
# GPU 1: LLaVA-Med + Kvasir and LLaVA-Med + EndoVis (sequential)

echo "📋 Job Plan:"
echo "   GPU 0: Qwen3-VL (Kvasir → EndoVis)"
echo "   GPU 1: LLaVA-Med (Kvasir → EndoVis)"
echo ""

# Start GPU 0 jobs in background
(
    run_training 0 qwen3vl kvasir \
        "corrected_1-5_experiments/qlora_experiments/models/qwen3vl_kvasir_instruction/best_model" \
        "results/qwen3vl_kvasir_cot"
    
    run_training 0 qwen3vl endovis \
        "corrected_1-5_experiments/qlora_experiments/models/qwen3vl_endovis_instruction/best_model" \
        "results/qwen3vl_endovis_cot"
) > slurm/logs/gpu0_training.log 2>&1 &
GPU0_PID=$!

# Start GPU 1 jobs in background
(
    run_training 1 llava_med kvasir \
        "corrected_1-5_experiments/qlora_experiments/models/llava_med_kvasir_instruction/best_model" \
        "results/llava_kvasir_cot"
    
    run_training 1 llava_med endovis \
        "corrected_1-5_experiments/qlora_experiments/models/llava_med_endovis_instruction/best_model" \
        "results/llava_endovis_cot"
) > slurm/logs/gpu1_training.log 2>&1 &
GPU1_PID=$!

# Wait for both GPU processes to complete
echo "⏳ Waiting for all training jobs to complete..."
echo "   GPU 0 PID: $GPU0_PID"
echo "   GPU 1 PID: $GPU1_PID"
echo ""

wait $GPU0_PID
GPU0_EXIT=$?

wait $GPU1_PID
GPU1_EXIT=$?

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALL TRAINING JOBS COMPLETED                              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "GPU 0 (Qwen3-VL) exit code: $GPU0_EXIT"
echo "GPU 1 (LLaVA-Med) exit code: $GPU1_EXIT"
echo ""
echo "End: $(date)"
echo ""

# Check final status
if [ $GPU0_EXIT -eq 0 ] && [ $GPU1_EXIT -eq 0 ]; then
    echo "✅ ALL JOBS COMPLETED SUCCESSFULLY!"
    exit 0
else
    echo "⚠️  SOME JOBS FAILED - Check logs for details"
    exit 1
fi


