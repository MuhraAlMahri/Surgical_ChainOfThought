#!/bin/bash
#SBATCH --job-name=exp1_llava_manual
#SBATCH --output=slurm/logs/train_exp1_llava_v15_manual_%j.out
#SBATCH --error=slurm/logs/train_exp1_llava_v15_manual_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Kvasir-VQA Exp1 Instruction Fine-tuning with LLaVA-Med v1.5 (Manual Training Loop)
# Using manual training loop to bypass DeepSpeed auto-detection

set -e

echo "=========================================="
echo "Kvasir-VQA Exp1 Instruction Fine-tuning (LLaVA-Med v1.5 - Manual Loop)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: 1"
echo "Started: $(date)"
echo ""
echo "⚡ CONFIGURATION:"
echo "   - Model: LLaVA-Med v1.5 (microsoft/llava-med-v1.5-mistral-7b)"
echo "   - Dataset: Kvasir-VQA ULTRA-CONDENSED"
echo "   - Training: Manual loop (NO DeepSpeed)"
echo "   - train_bs: 1"
echo "   - grad_accum: 8 (effective batch = 8)"
echo "   - lr: 2.0e-4"
echo "   - epochs: 5"
echo "   - max_seq_len: 2048"
echo "   - LoRA: r=8, alpha=16"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
EXPERIMENTS_DIR="${BASE_DIR}/exp1"
QLORA_DIR="${BASE_DIR}/qlora_experiments"

# Create directories
mkdir -p "${EXPERIMENTS_DIR}/slurm/logs"
mkdir -p "${EXPERIMENTS_DIR}/models"

# Load modules
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

# CRITICAL: Upgrade dependencies BEFORE Python imports anything
echo "=========================================="
echo "Upgrading dependencies for LLaVA..."
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Enable CUDA error checking for better debugging
export CUDA_LAUNCH_BLOCKING=1

# CRITICAL: Use shared cache to avoid disk quota issues
# All jobs share the same cache, so models are downloaded once and reused
SHARED_CACHE="/l/users/muhra.almahri/.cache/hf_shared"
export HF_HOME=${SHARED_CACHE}
export TRANSFORMERS_CACHE=${SHARED_CACHE}/transformers
export HF_HUB_CACHE=${SHARED_CACHE}
export HF_DATASETS_CACHE=${SHARED_CACHE}/datasets
export TRITON_CACHE_DIR=${SHARED_CACHE}/triton
mkdir -p ${SHARED_CACHE} ${TRITON_CACHE_DIR}

echo "Using shared cache: ${SHARED_CACHE}"
echo "Cache size: $(du -sh ${SHARED_CACHE} 2>/dev/null | cut -f1 || echo '0')"

# CRITICAL: Disable DeepSpeed completely
unset DS_SKIP_CUDA_CHECK
unset DS_ACCELERATOR
unset DEEPSPEED_CONFIG_FILE
export ACCELERATE_USE_DEEPSPEED=false
export DEEPSPEED_DISABLED=true
export ACCELERATE_USE_CUDA=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
# Note: CUDNN_DETERMINISTIC=1 can cause CUDNN_STATUS_INTERNAL_ERROR
# Using benchmark mode instead for stability
export CUDNN_DETERMINISTIC=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Hugging Face authentication (if needed for gated models)
export HF_TOKEN="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"

# CUDA environment
export CUDA_HOME=/apps/local/nvidia/cuda-12.0
export LD_LIBRARY_PATH=/apps/local/nvidia/cuda-12.0/lib64:$LD_LIBRARY_PATH

# ============================================================================
# Exp1 Instruction Fine-tuning Training (Manual Loop)
# ============================================================================
echo ""
echo "=========================================="
echo "Exp1 Instruction Fine-tuning (LLaVA-Med v1.5 - Manual Loop)"
echo "=========================================="
echo "Started: $(date)"
echo ""

CONFIG_FILE="${EXPERIMENTS_DIR}/config_exp1_llavamed_v15.yaml"

# Verify config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Verify dataset files exist
TRAIN_DATA="${BASE_DIR}/datasets/kvasir_ULTRA_CONDENSED/train_CATEGORY_BASED.jsonl"
VAL_DATA="${BASE_DIR}/datasets/kvasir_ULTRA_CONDENSED/val_CATEGORY_BASED.jsonl"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: Train data not found: ${TRAIN_DATA}"
    exit 1
fi

if [ ! -f "${VAL_DATA}" ]; then
    echo "ERROR: Val data not found: ${VAL_DATA}"
    exit 1
fi

echo "✓ Config file: ${CONFIG_FILE}"
echo "✓ Train data: ${TRAIN_DATA}"
echo "✓ Val data: ${VAL_DATA}"
echo ""

# Run training with manual loop
python3 "${QLORA_DIR}/train_llava_manual.py" \
    --config "${CONFIG_FILE}"

STATUS=$?

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=========================================="
echo "EXP1 LLAVA-MED V1.5 MANUAL TRAINING COMPLETED"
echo "=========================================="
echo "Completed: $(date)"
echo ""

if [ $STATUS -eq 0 ]; then
    echo "  ✓ Exp1 LLaVA-Med v1.5 Manual Training: SUCCESS"
    echo ""
    echo "Model output:"
    echo "  ${EXPERIMENTS_DIR}/models/exp1_llava_v15_manual"
    echo ""
    echo "Configuration:"
    echo "  - Model: LLaVA-Med v1.5 (microsoft/llava-med-v1.5-mistral-7b)"
    echo "  - Dataset: Kvasir-VQA ULTRA-CONDENSED"
    echo "  - Training: Manual loop (NO DeepSpeed)"
else
    echo "  ✗ Exp1 LLaVA-Med v1.5 Manual Training: FAILED (exit code: $STATUS)"
    exit $STATUS
fi

echo "=========================================="

exit 0


