#!/bin/bash
#SBATCH --job-name=eval_exp5
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_exp5_%j.out
#SBATCH --error=logs/eval_exp5_%j.err

echo "========================================================================="
echo "🔧 EXP5 EVALUATION - Sequential Chain-of-Thought"
echo "========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load modules and activate environment
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1

# Use /tmp for HF cache
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"

echo "Using HF_HOME: $HF_HOME"
echo ""

# Set base paths
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
SCRIPT_DIR="${BASE_DIR}/corrected 1-5 experiments/scripts/evaluation"
MODEL_DIR="${BASE_DIR}/corrected 1-5 experiments/models"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets"
RESULTS_DIR="${BASE_DIR}/corrected 1-5 experiments/results"

echo "========================================================================="
echo "📊 Exp5: Sequential Chain-of-Thought"
echo "========================================================================="
echo ""
echo "This evaluation uses 3-stage sequential reasoning:"
echo "  Stage 1: Q -> model -> pred1"
echo "  Stage 2: Q + pred1 -> model -> pred2"
echo "  Stage 3: Q + pred1 + pred2 -> model -> pred3 (final answer)"
echo ""

python3 "$SCRIPT_DIR/evaluate_exp5.py" \
    --model_path "$MODEL_DIR/exp5_sequential_cot" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "$RESULTS_DIR/exp5_evaluation_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct"

echo ""
echo "✅ Exp5 Complete: $(date)"
echo ""
echo "Results saved to: $RESULTS_DIR/exp5_evaluation_results.json"
echo ""

# Cleanup temp cache
rm -rf "$HF_HOME" "/tmp/triton_cache_$SLURM_JOB_ID"
echo "Cleaned up temp caches"


