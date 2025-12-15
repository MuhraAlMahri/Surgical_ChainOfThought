#!/bin/bash
#SBATCH --job-name=preprocess_revised
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/logs/preprocessing_%A.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=00:30:00

echo "=========================================="
echo "Creating Revised Instruction Templates"
echo "=========================================="
echo "Start time: $(date)"
echo ""

cd /l/users/muhra.almahri/Surgical_COT

# Create output directory
mkdir -p corrected_1-5_experiments/datasets/kvasir_REVISED_test

# Run preprocessing
python scripts/create_revised_instructions.py \
    --input_dir corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15 \
    --output_dir corrected_1-5_experiments/datasets/kvasir_REVISED_test

echo ""
echo "=========================================="
echo "Preprocessing Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh corrected_1-5_experiments/datasets/kvasir_REVISED_test/
echo ""
echo "Next: Run verification job (sbatch scripts/verify_instructions.sh)"
