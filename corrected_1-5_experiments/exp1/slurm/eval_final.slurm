#!/bin/bash
#SBATCH --job-name=exp1_eval_ckpt6000
#SBATCH --partition=cscc-gpu-p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/eval_ckpt6000_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/eval_ckpt6000_%j.err

# Evaluation for Exp1 Checkpoint-6000 (2.34 epochs, full resolution)
# Training was interrupted at 95% - evaluating existing checkpoint

echo "=============================================="
echo "EXP1 CHECKPOINT-6000 EVALUATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

# Setup
export PYTHONUNBUFFERED=1
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1

# Activate environment
source /l/users/muhra.almahri/miniconda3/etc/profile.d/conda.sh
conda activate surgical_cot

# Step 1: Generate predictions
echo ""
echo "STEP 1: Generating predictions on test set..."
echo "----------------------------------------------"
python3 predict_final.py

if [ $? -ne 0 ]; then
    echo "❌ Prediction failed!"
    exit 1
fi

# Step 2: Evaluate predictions
echo ""
echo "STEP 2: Evaluating predictions..."
echo "----------------------------------------------"
python3 eval_final.py

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ EVALUATION COMPLETE!"
echo "End Time: $(date)"
echo "=============================================="


