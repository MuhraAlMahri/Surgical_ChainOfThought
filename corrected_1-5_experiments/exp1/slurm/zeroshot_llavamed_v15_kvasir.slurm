#!/bin/bash
#SBATCH --job-name=zeroshot_llavamed_kvasir
#SBATCH --output=slurm/logs/zeroshot_llavamed_v15_kvasir_%j.out
#SBATCH --error=slurm/logs/zeroshot_llavamed_v15_kvasir_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Zero-Shot Evaluation for Kvasir-VQA using LLaVA-Med v1.5 (Mistral-7B)
# Evaluates the base LLaVA-Med v1.5 model WITHOUT any fine-tuning

set -e

echo "=========================================="
echo "ZERO-SHOT EVALUATION: Kvasir-VQA (LLaVA-Med v1.5)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
SCRIPT_DIR="${BASE_DIR}/scripts/evaluation"
RESULTS_DIR="${BASE_DIR}/exp1/results"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"

# Base model
BASE_MODEL="microsoft/llava-med-v1.5-mistral-7b"

# Test data paths
TEST_DATA_JSONL="${BASE_DIR}/datasets/kvasir_ULTRA_CONDENSED/test_CATEGORY_BASED.jsonl"
TEST_DATA_JSON="/tmp/zeroshot_kvasir_llavamed_${SLURM_JOB_ID}.json"

# Create results directory
mkdir -p "${RESULTS_DIR}"
mkdir -p "${BASE_DIR}/exp1/slurm/logs"

# Load modules
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

# CRITICAL: Upgrade dependencies BEFORE Python imports anything
# This must happen before the Python script runs, as transformers checks versions on import
echo "=========================================="
echo "Upgrading dependencies for LLaVA-Med..."
echo "=========================================="
echo "Step 1: Upgrading huggingface-hub..."
pip install --upgrade "huggingface-hub>=1.0.0" --quiet
echo "✓ huggingface-hub upgraded"

echo "Step 2: Installing transformers from source..."
pip install --upgrade git+https://github.com/huggingface/transformers.git --quiet
echo "✓ Transformers updated from source"

echo "Step 3: Verifying versions..."
python3 -c "import huggingface_hub; import transformers; print(f'huggingface-hub: {huggingface_hub.__version__}'); print(f'transformers: {transformers.__version__}')"
echo "=========================================="

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Use user directory instead of /tmp to avoid disk space issues
export HF_HOME=/l/users/muhra.almahri/.cache/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_cache_${SLURM_JOB_ID}
mkdir -p ${HF_HOME} ${TRITON_CACHE_DIR}

# Hugging Face authentication (if needed for gated models)
export HF_TOKEN="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"

# CUDA environment
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH

# Convert JSONL to JSON and verify
echo "Using test dataset: ${TEST_DATA_JSONL}"
python3 << PYTHON_EOF
import json
import sys
import os

jsonl_path = "${TEST_DATA_JSONL}"
json_path = "${TEST_DATA_JSON}"

try:
    # Read JSONL and convert to JSON
    with open(jsonl_path, 'r') as f_in:
        data = [json.loads(line) for line in f_in]
    
    # Write as JSON
    with open(json_path, 'w') as f_out:
        json.dump(data, f_out, indent=2)
    
    print(f"✓ Converted {len(data)} samples from JSONL to JSON")
    print(f"✓ Loaded {len(data)} test samples")
    
except Exception as e:
    print(f"Error reading test data: {e}")
    sys.exit(1)
PYTHON_EOF

# Run zero-shot evaluation
echo ""
echo "=========================================="
echo "Running Zero-Shot Evaluation (LLaVA-Med v1.5)"
echo "=========================================="

python3 "${SCRIPT_DIR}/evaluate_zeroshot.py" \
    --base_model "${BASE_MODEL}" \
    --test_data "${TEST_DATA_JSON}" \
    --image_dir "${IMAGE_DIR}" \
    --output "${RESULTS_DIR}/kvasir_zeroshot_llavamed_v15.json" \
    --use_instruction

echo ""
echo "=========================================="
echo "ZERO-SHOT EVALUATION COMPLETED"
echo "=========================================="
echo "Results saved to: ${RESULTS_DIR}/kvasir_zeroshot_llavamed_v15.json"
echo "=========================================="

# Clean up temporary file
rm -f "${TEST_DATA_JSON}"

echo "Zero-shot evaluation completed successfully!"

