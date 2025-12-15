#!/bin/bash
#SBATCH --job-name=category_instructions
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/logs/category_instructions_%A.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=00:30:00

echo "=========================================="
echo "Creating Category-Based Instructions"
echo "=========================================="
echo "Start time: $(date)"
echo ""

cd /l/users/muhra.almahri/Surgical_COT

# Create output directory
mkdir -p corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED

# Run category-based instruction builder
python scripts/create_category_based_instructions.py \
    --train_file corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json \
    --val_file corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/val.json \
    --test_file corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15/test.json \
    --output_dir corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED

echo ""
echo "=========================================="
echo "Category-Based Instructions Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Output files created:"
ls -lh corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED/
echo ""
echo "IMPORTANT: Send this file to your advisor:"
echo "  corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED/INSTRUCTIONS_PER_CATEGORY.txt"
echo ""
echo "This file contains ONE instruction template per category"
echo "as requested by your advisor."
