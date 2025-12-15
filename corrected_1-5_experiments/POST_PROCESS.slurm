#!/bin/bash
#SBATCH --job-name=post_process
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/post_process_%j.out
#SBATCH --error=logs/post_process_%j.err

echo "========================================================================="
echo "📊 POST-PROCESSING: Compute Concise Accuracy"
echo "========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Activate environment
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1

# Change to correct directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments"

echo "Working directory: $(pwd)"
echo ""

echo "========================================================================="
echo "Running concise answer extraction..."
echo "========================================================================="
echo ""

python3 scripts/post_processing/compute_concise_accuracy.py

echo ""
echo "========================================================================="
echo "✅ Post-processing complete: $(date)"
echo "========================================================================="
echo ""




