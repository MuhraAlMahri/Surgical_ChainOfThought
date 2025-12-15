#!/bin/bash
#SBATCH --job-name=test_single_sample
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/logs/test_single_sample_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/logs/test_single_sample_%j.err"

echo "=========================================="
echo "Testing Instructed Model on Single Sample"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

echo "GPU Info:"
nvidia-smi
echo ""

python3 /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/test_model_single_sample.py

echo ""
echo "Test completed at: $(date)"




