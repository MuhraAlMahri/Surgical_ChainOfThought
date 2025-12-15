#!/bin/bash
#SBATCH --job-name=regenerate_endovis_vqa
#SBATCH --output=logs/regenerate_endovis_vqa_%j.out
#SBATCH --error=logs/regenerate_endovis_vqa_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Change to project directory
cd /l/users/muhra.almahri/Surgical_COT

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate Python environment if needed (adjust path as necessary)
# source /path/to/venv/bin/activate

# Run the script
echo "Starting EndoVis2018 VQA dataset regeneration..."
echo ""

python3 scripts/prepare_endovis2018_for_vqa.py \
    --output_dir corrected_1-5_experiments/datasets/endovis2018_vqa \
    --image_dir datasets/EndoVis2018/raw/images \
    --use_proper_split

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Job completed!"
echo "=========================================="

