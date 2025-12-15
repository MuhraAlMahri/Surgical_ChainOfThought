#!/bin/bash
#SBATCH --job-name=categorize_questions
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/categorize_questions_%j.out
#SBATCH --error=slurm/logs/categorize_questions_%j.err

echo "=========================================="
echo "Question Categorization Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load modules
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

# Environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create output directory
mkdir -p data/categorized
mkdir -p slurm/logs

# Dataset configuration
DATASET="${1:-kvasir}"  # kvasir or endovis
TRAIN_FILE="${2:-datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json}"
OUTPUT_DIR="data/categorized"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Input file: $TRAIN_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Model: $MODEL_NAME"
echo ""

# Check input file exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ ERROR: Input file not found: $TRAIN_FILE"
    exit 1
fi

# GPU info
nvidia-smi
echo ""

# Run categorization
echo "Starting question categorization..."
python3 data/question_categorizer.py \
    --input "$TRAIN_FILE" \
    --output "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --model "$MODEL_NAME"

echo ""
echo "=========================================="
echo "Categorization completed!"
echo "End time: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="














