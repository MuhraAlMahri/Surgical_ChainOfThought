#!/bin/bash
#SBATCH --job-name=endovis_train_exp1
#SBATCH --output=slurm/logs/train_exp1_%j.out
#SBATCH --error=slurm/logs/train_exp1_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Training Script for EndoVis2018 - Experiment 1: Random Baseline
# IMPORTANT: This runs on compute nodes, NOT login nodes (MBZUAI HPC policy)

set -e

echo "=========================================="
echo "TRAINING: EndoVis2018 - Exp1 Random Baseline"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Started: $(date)"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
QLORA_SCRIPT="${BASE_DIR}/qlora_experiments/train_qlora_qwen3vl.py"
CONFIG_FILE="${EXPERIMENTS_DIR}/configs/exp1_random.yaml"

# Create logs directory
mkdir -p "${EXPERIMENTS_DIR}/slurm/logs"

# Load modules (if needed)
# module load nvidia/cuda/12.0
# source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1

# Run training
echo ""
echo "Starting training..."
echo "Config: ${CONFIG_FILE}"
echo ""

python3 "${QLORA_SCRIPT}" \
    --config "${CONFIG_FILE}"

echo ""
echo "=========================================="
echo "TRAINING COMPLETED"
echo "=========================================="
echo "Finished: $(date)"
echo "Model saved to: ${EXPERIMENTS_DIR}/models/exp1_random"
echo "=========================================="





















