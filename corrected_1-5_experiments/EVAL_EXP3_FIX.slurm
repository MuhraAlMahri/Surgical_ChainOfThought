#!/bin/bash
#SBATCH --job-name=eval_exp3
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_exp3_fix_%j.out
#SBATCH --error=logs/eval_exp3_fix_%j.err

echo "========================================================================="
echo "🔧 EXP3 EVALUATION (FIXED ARGUMENTS)"
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
echo "📊 Exp3: CXR-TREK Sequential (3 Separate Models)"
echo "========================================================================="
echo ""

python3 "$SCRIPT_DIR/evaluate_exp3.py" \
    --model_stage1 "$MODEL_DIR/exp3_cxrtrek_seq/stage1" \
    --model_stage2 "$MODEL_DIR/exp3_cxrtrek_seq/stage2" \
    --model_stage3 "$MODEL_DIR/exp3_cxrtrek_seq/stage3" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "$RESULTS_DIR/exp3_evaluation_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct"

echo ""
echo "✅ Exp3 Complete: $(date)"
echo ""
echo "Results saved to: $RESULTS_DIR/exp3_evaluation_results.json"
echo ""

# Cleanup temp cache
rm -rf "$HF_HOME" "/tmp/triton_cache_$SLURM_JOB_ID"
echo "Cleaned up temp caches"


