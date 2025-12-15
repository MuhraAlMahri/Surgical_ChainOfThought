#!/bin/bash
#SBATCH --job-name=llava_FINAL
#SBATCH --output=logs/llava_final_%j.out
#SBATCH --error=logs/llava_final_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=cscc-gpu

echo "=========================================="
echo "LLaVA-Med Training - FINAL VERSION"
echo "Uses config.image_token_id (32000)"
echo "=========================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Load modules
module purge
module load cuda/11.8
module load anaconda3

# Activate environment
source activate llm_env  # CHANGE TO YOUR ENV NAME

# Shared cache configuration
export SHARED_CACHE="/l/users/muhra.almahri/.cache/hf_shared"
export HF_HOME=$SHARED_CACHE
export TRANSFORMERS_CACHE=$SHARED_CACHE/transformers
export HF_DATASETS_CACHE=$SHARED_CACHE/datasets
export HF_HUB_CACHE=$SHARED_CACHE
export TORCH_HOME=$SHARED_CACHE/torch

# CRITICAL: Aggressively disable DeepSpeed completely
# Remove all DeepSpeed-related environment variables
unset DS_SKIP_CUDA_CHECK
unset DS_ACCELERATOR
unset DEEPSPEED_CONFIG_FILE
unset DEEPSPEED_AUTOTUNING
unset DEEPSPEED_MULTINODE_LAUNCHER
unset DEEPSPEED_HOSTFILE
unset DEEPSPEED_INIT_METHOD

# Set explicit environment variables to disable DeepSpeed
export ACCELERATE_USE_DEEPSPEED=false
export DEEPSPEED_DISABLED=true
export ACCELERATE_USE_CUDA=true
export ACCELERATE_USE_CPU=false
export ACCELERATE_USE_XPU=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export CUDNN_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Dataset configuration (adjust as needed)
export DATASET_NAME=kvasir
export DATA_ROOT=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_ULTRA_CONDENSED

echo "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  Cache: $SHARED_CACHE"
echo ""

# Create directories
mkdir -p logs outputs

# Change to script directory
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments

# Run training (adjust config path as needed)
echo "Starting training with config.image_token_id (32000)..."
python train_llava_FINAL.py --config ../exp1/config_exp1_llavamed_v15.yaml

echo ""
echo "Completed at $(date)"
echo "=========================================="

