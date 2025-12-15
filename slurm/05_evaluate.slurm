#!/bin/bash
#SBATCH --job-name=evaluate_cot
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/evaluate_%j.out
#SBATCH --error=slurm/logs/evaluate_%j.err

echo "=========================================="
echo "Model Evaluation Job"
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

# Create directories
mkdir -p slurm/logs
mkdir -p results

# Configuration
MODEL_PATH="${1:-checkpoints/kvasir_unified/best_model.pt}"
TEST_DATA="${2:-data/categorized/test_categorized.json}"
OUTPUT_DIR="results/evaluation_$(date +%Y%m%d_%H%M%S)"
BASELINE_RESULTS="${3:-}"  # Optional baseline results JSON

echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Test data: $TEST_DATA"
echo "  Output directory: $OUTPUT_DIR"
if [ ! -z "$BASELINE_RESULTS" ]; then
    echo "  Baseline results: $BASELINE_RESULTS"
fi
echo ""

# Check files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "❌ ERROR: Test data file not found: $TEST_DATA"
    exit 1
fi

# GPU info
nvidia-smi
echo ""

# Run evaluation
echo "Starting evaluation..."
python3 evaluation/baseline_comparison.py \
    --model-path "$MODEL_PATH" \
    --test-data "$TEST_DATA" \
    --output "$OUTPUT_DIR" \
    ${BASELINE_RESULTS:+--baseline-results "$BASELINE_RESULTS"}

EVAL_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EVAL_EXIT_CODE














