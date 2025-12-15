#!/bin/bash
#SBATCH --job-name=eval_cot_all_2gpu
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/eval_cot_all_2gpu_%j.out
#SBATCH --error=slurm/logs/eval_cot_all_2gpu_%j.err

echo "=========================================="
echo "Evaluate All CoT Configurations (2 GPUs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load modules
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

# Environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# HuggingFace token (if available)
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo "Using HF_TOKEN from environment"
elif [ -f ~/.hf_token ]; then
    export HF_TOKEN=$(cat ~/.hf_token)
    echo "Loaded HF_TOKEN from ~/.hf_token"
fi

# Use /tmp for HuggingFace cache
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create directories
mkdir -p slurm/logs
mkdir -p results/cot_evaluation

# Configuration
BASELINE_RESULTS="${1:-results/baseline_results.json}"
OUTPUT_DIR="results/cot_evaluation/all_combined_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Baseline results: $BASELINE_RESULTS"
echo "  Output directory: $OUTPUT_DIR"
echo "  GPUs: 0, 1"
echo ""

# Check baseline results exist
if [ ! -f "$BASELINE_RESULTS" ]; then
    echo "❌ ERROR: Baseline results file not found: $BASELINE_RESULTS"
    exit 1
fi

# Run combined evaluation
echo "Starting combined evaluation on 2 GPUs..."
python3 -u evaluate_cot_all_combined.py \
    --baseline-results "$BASELINE_RESULTS" \
    --output-dir "$OUTPUT_DIR" \
    --gpu-0 0 \
    --gpu-1 1

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Combined evaluation completed!"
    echo "   Results: $OUTPUT_DIR"
else
    echo "❌ Combined evaluation failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

# Clean up temporary cache
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up temporary cache..."
    rm -rf "$HF_HOME"
fi

exit $EXIT_CODE

