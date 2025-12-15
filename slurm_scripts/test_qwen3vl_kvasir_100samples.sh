#!/bin/bash
#SBATCH --job-name=test_qwen3vl_100
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/test_qwen3vl_kvasir_100samples_%j.out
#SBATCH --error=slurm/logs/test_qwen3vl_kvasir_100samples_%j.err

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TEST: Qwen3-VL Kvasir Zeroshot+COT (100 samples)        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

# Install peft if needed (to user directory to avoid disk quota)
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PIP_TARGET="/l/users/muhra.almahri/.local/lib/python${PYTHON_VERSION}/site-packages"
export PYTHONPATH="${PIP_TARGET}:${PYTHONPATH}"
mkdir -p ${PIP_TARGET}
pip install --target ${PIP_TARGET} -q peft 2>/dev/null || echo "Note: peft may already be installed"

# Environment
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use /tmp for HuggingFace cache
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Load HuggingFace token
if [ -f ~/.hf_token ]; then
    export HF_TOKEN=$(cat ~/.hf_token)
elif [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
else
    export HF_TOKEN="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"
    echo "⚠️  Using default HF_TOKEN for gated repos"
fi

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create directories
mkdir -p slurm/logs
mkdir -p results

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 TEST: Evaluate Qwen3-VL Kvasir Zeroshot+COT (100 samples)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  - Checkpoint: results/qwen3vl_kvasir_cot_5epochs/checkpoint_epoch_5.pt"
echo "  - Base model: Qwen/Qwen3-VL-8B-Instruct"
echo "  - Dataset: Kvasir test (100 samples only)"
echo "  - Batch size: 1 (conservative)"
echo "  - Output: results/test_qwen3vl_kvasir_100samples.json"
echo ""

# Run evaluation with max_samples=100
python evaluate_multihead_cot_comprehensive.py \
    --checkpoint results/qwen3vl_kvasir_cot_5epochs/checkpoint_epoch_5.pt \
    --base_checkpoint Qwen/Qwen3-VL-8B-Instruct \
    --model_name qwen3vl \
    --dataset kvasir \
    --data_path corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15 \
    --image_base_path datasets/Kvasir-VQA/raw/images \
    --question_categories question_categories.json \
    --output_file results/test_qwen3vl_kvasir_100samples.json \
    --batch_size 1 \
    --use_flexible_matching \
    --max_samples 100

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST PASSED - Evaluation completed successfully"
    echo ""
    echo "Results saved to: results/test_qwen3vl_kvasir_100samples.json"
    echo ""
    echo "Next step: Run full evaluation if this test passed"
else
    echo "❌ TEST FAILED - Exit code: $EXIT_CODE"
    echo ""
    echo "Check logs for errors and fix before running full evaluation"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $EXIT_CODE

