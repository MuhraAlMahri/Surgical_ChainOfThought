#!/bin/bash
#SBATCH --job-name=create_temporal
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/create_temporal_%j.out
#SBATCH --error=slurm/logs/create_temporal_%j.err

echo "=========================================="
echo "Temporal Structure Creation Job"
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

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create output directory
mkdir -p data/temporal
mkdir -p slurm/logs

# Configuration
SEQUENCE_DIR="${1:-datasets/EndoVis2018/data/images}"
QA_FILE="${2:-datasets/EndoVis2018/train.json}"
OUTPUT_FILE="data/temporal/temporal_structure.json"
SEQUENCE_ID="${3:-seq_1}"

echo "Configuration:"
echo "  Sequence directory: $SEQUENCE_DIR"
echo "  QA file: $QA_FILE"
echo "  Output file: $OUTPUT_FILE"
echo "  Sequence ID: $SEQUENCE_ID"
echo ""

# Check files exist
if [ ! -d "$SEQUENCE_DIR" ]; then
    echo "❌ ERROR: Sequence directory not found: $SEQUENCE_DIR"
    exit 1
fi

if [ ! -f "$QA_FILE" ]; then
    echo "❌ ERROR: QA file not found: $QA_FILE"
    exit 1
fi

# Run temporal structure creation
echo "Creating temporal structure..."
python3 data/temporal_linker.py \
    --sequence-dir "$SEQUENCE_DIR" \
    --qa-file "$QA_FILE" \
    --output "$OUTPUT_FILE" \
    --sequence-id "$SEQUENCE_ID"

echo ""
echo "=========================================="
echo "Temporal structure creation completed!"
echo "End time: $(date)"
echo "Output: $OUTPUT_FILE"
echo "=========================================="














