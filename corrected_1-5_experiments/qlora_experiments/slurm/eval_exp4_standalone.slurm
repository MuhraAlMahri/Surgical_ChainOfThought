#!/bin/bash
#SBATCH --job-name=exp4_eval
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp4_eval_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp4_eval_%j.err

# ============================================================================
# Exp4 Evaluation (Standalone)
# ============================================================================
# Evaluates the retrained Exp4 model (with fixed curriculum learning)
# ============================================================================

echo "=========================================="
echo "Exp4 Evaluation (Retrained Model)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Setup
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p ${HF_HOME}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"
SCRIPTS_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/scripts/evaluation"
IMAGE_ROOT="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
RESULTS_DIR="${BASE_DIR}/results"

# Create results directory
mkdir -p ${RESULTS_DIR}

# Model path (use stage2 latest checkpoint since stage3 doesn't have adapter)
EXP4_MODEL="${BASE_DIR}/models/exp4_curriculum/stage2/checkpoint-3294"
EXP4_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl"
EXP4_OUTPUT="${RESULTS_DIR}/exp4_evaluation.json"

# Verify model exists
if [ ! -d "${EXP4_MODEL}" ]; then
    echo "❌ ERROR: Exp4 model not found at ${EXP4_MODEL}"
    exit 1
fi

echo "Model: ${EXP4_MODEL}"
echo "Test data: ${EXP4_TEST_DATA}"
echo "Output: ${EXP4_OUTPUT}"
echo ""

# Convert JSONL to JSON for evaluation script
EXP4_TEST_JSON="/tmp/exp4_test_${SLURM_JOB_ID}.json"
if [ -f "${EXP4_TEST_DATA}" ]; then
    echo "Converting test data from JSONL to JSON..."
    python3 -c "
import json
with open('${EXP4_TEST_DATA}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${EXP4_TEST_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"
else
    echo "❌ ERROR: Test data not found at ${EXP4_TEST_DATA}"
    exit 1
fi

# Show GPU status
nvidia-smi
echo ""

# Run evaluation
echo "Starting Exp4 evaluation..."
echo "=========================================="

python3 ${SCRIPTS_DIR}/evaluate_exp4.py \
    --model_path ${EXP4_MODEL} \
    --test_data ${EXP4_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP4_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct

EXIT_CODE=$?

echo "=========================================="
echo "Exp4 evaluation completed with exit code: ${EXIT_CODE}"
echo "Results saved to: ${EXP4_OUTPUT}"
echo "End time: $(date)"
echo "=========================================="

# Cleanup
rm -f ${EXP4_TEST_JSON}
rm -rf ${HF_HOME}
rm -rf ${TRITON_CACHE_DIR}

exit ${EXIT_CODE}


