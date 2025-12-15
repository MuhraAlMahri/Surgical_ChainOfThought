#!/bin/bash
#SBATCH --job-name=res_test
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/resolution_test_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/resolution_test_%j.err

echo "=========================================="
echo "RESOLUTION COMPARISON TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Load CUDA
echo "Loading CUDA module..."
module load nvidia/cuda/12.0

# Activate environment
echo "Activating conda environment..."
source ~/miniconda3/bin/activate base

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME"

echo "Cache directories created:"
echo "  HF_HOME: $HF_HOME"
echo "  TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo ""

# Copy model cache to /tmp for faster access
SOURCE_MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct"
if [ -d "$SOURCE_MODEL_CACHE" ]; then
    echo "Copying model cache to /tmp..."
    cp -r "$SOURCE_MODEL_CACHE" "$HF_HOME/hub/"
    echo "Model cache copied."
else
    echo "WARNING: Source model cache not found. Will download on first use."
fi
echo ""

# Change to working directory
cd "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"

echo "=========================================="
echo "TEST CONFIGURATION"
echo "=========================================="
echo "Testing 4 resolutions:"
echo "  1. Low (448x448)     - Fast, like previous experiments"
echo "  2. Medium (768x768)  - Balanced speed/quality"
echo "  3. High (1024x1024)  - Better quality, slower"
echo "  4. Full (~1036x1288) - Current setting, highest quality"
echo ""
echo "Each test runs 50 training steps on 100 samples"
echo "Estimated time: 3-5 hours total"
echo "=========================================="
echo ""

# Run resolution comparison
python3 exp1/test_resolutions.py \
    --config exp1/config_exp1_category_based.yaml \
    --max_steps 50

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ RESOLUTION TEST COMPLETED"
    echo ""
    echo "Results saved to:"
    echo "  exp1/resolution_tests/comparison_summary.json"
    echo ""
    echo "To view results:"
    echo "  cat exp1/resolution_tests/comparison_summary.json | python3 -m json.tool"
else
    echo "❌ RESOLUTION TEST FAILED with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

# Cleanup
echo ""
echo "Cleaning up temporary files..."
rm -rf "$HF_HOME"
rm -rf "$TRITON_CACHE_DIR"
echo "Cleanup complete."

exit $EXIT_CODE







