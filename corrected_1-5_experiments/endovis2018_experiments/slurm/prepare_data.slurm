#!/bin/bash
#SBATCH --job-name=prep_endovis_data
#SBATCH --output=slurm/logs/prepare_data_%j.out
#SBATCH --error=slurm/logs/prepare_data_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Data Preparation for EndoVis2018
# IMPORTANT: Runs on compute nodes (MBZUAI HPC policy)

set -e

echo "=========================================="
echo "EndoVis2018 Data Preparation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "=========================================="

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "${BASE_DIR}"

# Step 1: Organize images
echo ""
echo "Step 1: Organizing images..."
python scripts/organize_endovis2018.py

# Step 2: Prepare VQA format
echo ""
echo "Step 2: Preparing VQA format..."
python scripts/prepare_endovis2018_for_vqa.py \
    --output_dir corrected_1-5_experiments/datasets/endovis2018_vqa \
    --image_dir datasets/EndoVis2018/raw/images \
    --use_proper_split

echo ""
echo "=========================================="
echo "Data Preparation Completed"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Output directory: corrected_1-5_experiments/datasets/endovis2018_vqa"
echo "Files created:"
echo "  - train.jsonl"
echo "  - validation.jsonl"
echo "  - test.jsonl"
echo "=========================================="




















