#!/bin/bash
#SBATCH --job-name=eval_endovis_ft
#SBATCH --output=slurm/logs/eval_endovis_finetuned_llava_med_%j.out
#SBATCH --error=slurm/logs/eval_endovis_finetuned_llava_med_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Evaluation Job: Fine-tuned EndoVis2018 LLaVA-Med v1.5 Model
# Evaluates EndoVis2018 fine-tuned model on test set
#
# Model: microsoft/llava-med-v1.5-mistral-7b (with LoRA adapter)

set -e

echo "=========================================="
echo "FINE-TUNED MODEL EVALUATION: EndoVis2018 LLaVA-Med v1.5"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: 1"
echo "Started: $(date)"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
SCRIPT_DIR="${BASE_DIR}/scripts/evaluation"
QLORA_DIR="${BASE_DIR}/qlora_experiments"
RESULTS_DIR="${QLORA_DIR}/results"
ENDOVIS_IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"

# Base model
BASE_MODEL="microsoft/llava-med-v1.5-mistral-7b"

# Fine-tuned model path
ENDOVIS_ADAPTER="${QLORA_DIR}/models/llava_med_endovis_instruction/best_model"

# Test data paths
ENDOVIS_TEST_JSONL="${BASE_DIR}/datasets/endovis2018_vqa/test.jsonl"
ENDOVIS_TEST_JSON="/tmp/test_endovis_llava_${SLURM_JOB_ID}.json"

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${QLORA_DIR}/slurm/logs"

# Load modules
module purge
module load nvidia/cuda/12.0 2>/dev/null || true

# Activate environment
source ~/miniconda3/bin/activate base

# Install only peft to user-writable location (avoid home quota)
# Note: torch, transformers, etc. should already be in conda environment
# Only install peft which is a pure Python package
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PIP_TARGET="/l/users/muhra.almahri/.local/lib/python${PYTHON_VERSION}/site-packages"
export PYTHONPATH="${PIP_TARGET}:${PYTHONPATH}"
mkdir -p ${PIP_TARGET}
pip install --target ${PIP_TARGET} -q peft 2>/dev/null || echo "Note: peft may already be installed"

export PYTHONUNBUFFERED=1

# Cache configuration
export HF_HOME=/l/users/muhra.almahri/.cache/hf_shared
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/l/users/muhra.almahri/.cache/triton_shared
mkdir -p ${HF_HOME} ${TRANSFORMERS_CACHE} ${HF_DATASETS_CACHE} ${HF_HUB_CACHE} ${TRITON_CACHE_DIR}

# Disable DeepSpeed
export ACCELERATE_USE_DEEPSPEED=false
export DEEPSPEED_DISABLED=true
export ACCELERATE_USE_CUDA=true
export ACCELERATE_USE_CPU=false

# CUDA settings
export CUDNN_DETERMINISTIC=0
export CUDA_LAUNCH_BLOCKING=0
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH

# Hugging Face authentication
export HF_TOKEN="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"

# ============================================================================
# EndoVis2018 Evaluation
# ============================================================================
echo ""
echo "=========================================="
echo "Evaluating EndoVis2018 Fine-Tuned Model"
echo "=========================================="
echo "Started: $(date)"
echo "Model: ${ENDOVIS_ADAPTER}"

# Convert JSONL to JSON
python3 << PYTHON_EOF
import json
import sys

jsonl_path = "${ENDOVIS_TEST_JSONL}"
json_path = "${ENDOVIS_TEST_JSON}"

try:
    with open(jsonl_path, 'r') as f_in:
        data = [json.loads(line) for line in f_in]
    
    with open(json_path, 'w') as f_out:
        json.dump(data, f_out, indent=2)
    
    print(f"✓ Converted {len(data)} samples from JSONL to JSON")
except Exception as e:
    print(f"Error reading test data: {e}")
    sys.exit(1)
PYTHON_EOF

# Run evaluation
python3 "${SCRIPT_DIR}/evaluate_finetuned_llava.py" \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ENDOVIS_ADAPTER}" \
    --test_data "${ENDOVIS_TEST_JSON}" \
    --image_dir "${ENDOVIS_IMAGE_DIR}" \
    --output "${RESULTS_DIR}/endovis_finetuned_llava_med_v15_fixed_v2.json" \
    --use_instruction

ENDOVIS_STATUS=$?
if [ $ENDOVIS_STATUS -eq 0 ]; then
    echo "✓ EndoVis2018 evaluation completed: $(date)"
    echo "  Results: ${RESULTS_DIR}/endovis_finetuned_llava_med_v15_fixed_v2.json"
else
    echo "✗ EndoVis2018 evaluation failed: $(date) (exit code: $ENDOVIS_STATUS)"
    exit $ENDOVIS_STATUS
fi

# Clean up temporary file
rm -f "${ENDOVIS_TEST_JSON}"

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "ENDOVIS2018 EVALUATION COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""
if [ $ENDOVIS_STATUS -eq 0 ]; then
    echo "  ✓ EndoVis2018 Fine-tuned Evaluation: SUCCESS"
else
    echo "  ✗ EndoVis2018 Fine-tuned Evaluation: FAILED (exit code: $ENDOVIS_STATUS)"
fi

echo ""
echo "Results saved to:"
echo "  EndoVis2018: ${RESULTS_DIR}/endovis_finetuned_llava_med_v15_fixed_v2.json"
echo "=========================================="

exit $ENDOVIS_STATUS





