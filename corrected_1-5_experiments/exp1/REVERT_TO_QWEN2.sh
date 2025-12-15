#!/bin/bash
# Quick script to revert back to Qwen2-VL-7B if Qwen3-VL doesn't fit

cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1

echo "Reverting to Qwen2-VL-7B..."

# Update configs
sed -i 's/Qwen3-VL-8B-Instruct/Qwen2-VL-7B-Instruct/g' config_exp1*.yaml

# Update train_exp1.py
sed -i 's/AutoModelForImageTextToText/AutoModelForVision2Seq/g' train_exp1.py
sed -i 's/load_in_8bit=True.*# Use 8-bit.*/trust_remote_code=True/g' train_exp1.py

echo "✅ Reverted to Qwen2-VL-7B"
echo "You can now train with: sbatch slurm/train_exp1_category_based.slurm"






