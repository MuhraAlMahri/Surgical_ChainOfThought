#!/bin/bash
#SBATCH --job-name=exp1_predict_642
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/predict_642_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/predict_642_%j.err

echo "========================================================================="
echo "EXP1 PREDICTION - Checkpoint 642"
echo "========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Load CUDA
module load nvidia/cuda/12.0

# Activate environment
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"

# Run prediction
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "Generating predictions for Exp1 (checkpoint-642)..."
echo "Model: Qwen3-VL-8B-Instruct + LoRA (768×768, 1 epoch)"
echo "Test set: 8,984 samples"
echo ""

python3 exp1/predict_checkpoint642.py

EXIT_CODE=$?

echo ""
echo "========================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Prediction completed successfully"
    echo "Output: exp1/outputs/predictions_checkpoint642.jsonl"
else
    echo "❌ Prediction failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "========================================================================="

# Cleanup
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"

exit $EXIT_CODE





