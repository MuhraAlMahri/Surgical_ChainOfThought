#!/bin/bash
#SBATCH --job-name=exp1_train
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/train_%j.err

echo "EXP1 REFACTORED - FULL TRAINING"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

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

# Copy model cache to /tmp
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct"
if [ -d "$SOURCE_MODEL_CACHE" ]; then
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/"
fi

# Run training
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

python3 exp1/train_exp1.py exp1/config_exp1.yaml

echo "Training completed at: $(date)"

# Cleanup
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"


















