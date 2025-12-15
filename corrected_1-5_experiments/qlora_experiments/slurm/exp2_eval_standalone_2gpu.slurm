#!/bin/bash
#SBATCH --job-name=exp2_eval_standalone
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp2_eval_standalone_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp2_eval_standalone_%j.err

# ============================================================================
# Exp2 Evaluation - Standalone (Independent, no dependencies)
# ============================================================================

echo "=========================================="
echo "Exp2 Evaluation - Standalone"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Setup
module load nvidia/cuda/12.0 2>/dev/null || true
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"
SCRIPTS_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/scripts/evaluation"
IMAGE_ROOT="/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
RESULTS_DIR="${BASE_DIR}/results"

# Create results directory
mkdir -p ${RESULTS_DIR}

# Set environment
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p ${HF_HOME}

# Show GPU status
nvidia-smi
echo ""

# Exp2 Evaluation
EXP2_MODEL="${BASE_DIR}/models/exp2_qwen_reordered"
EXP2_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp2_qwen_reordered/test.jsonl"
EXP2_OUTPUT="${RESULTS_DIR}/exp2_evaluation.json"

# Convert JSONL to JSON for evaluation script
EXP2_TEST_JSON="/tmp/exp2_test_${SLURM_JOB_ID}.json"
if [ -f "${EXP2_TEST_DATA}" ]; then
    echo "Converting Exp2 test data from JSONL to JSON..."
    python3 -c "
import json
with open('${EXP2_TEST_DATA}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${EXP2_TEST_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"
fi

echo "Running Exp2 evaluation..."
python3 ${SCRIPTS_DIR}/evaluate_exp2.py \
    --model_path ${EXP2_MODEL} \
    --test_data ${EXP2_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP2_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct

EXIT_CODE=$?

# Cleanup
rm -f ${EXP2_TEST_JSON}
rm -rf ${HF_HOME}
rm -rf ${TRITON_CACHE_DIR}

echo ""
echo "=========================================="
echo "Exp2 Evaluation - COMPLETED"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}







