#!/bin/bash
#SBATCH --job-name=exp2_eval_rerun
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp2_eval_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp2_eval_%j.err

# ============================================================================
# Exp2 Evaluation Re-run (Standalone)
# ============================================================================
# This job runs Exp2 evaluation with the fixed script that includes
# instruction and question_type tracking
# ============================================================================

echo "=========================================="
echo "Exp2 Evaluation Re-run"
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

# Show GPU status
nvidia-smi
echo ""

# ============================================================================
# Exp2 Evaluation
# ============================================================================
EXP2_MODEL="${BASE_DIR}/models/exp2_qwen_reordered"
EXP2_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp2_qwen_reordered/test.jsonl"
EXP2_OUTPUT="${RESULTS_DIR}/exp2_evaluation.json"

# Verify model exists
if [ ! -d "${EXP2_MODEL}" ]; then
    echo "ERROR: Exp2 model not found at ${EXP2_MODEL}"
    exit 1
fi
echo "✓ Exp2 model found: ${EXP2_MODEL}"

# Verify test data exists
if [ ! -f "${EXP2_TEST_DATA}" ]; then
    echo "ERROR: Exp2 test data not found at ${EXP2_TEST_DATA}"
    exit 1
fi
echo "✓ Exp2 test data found: ${EXP2_TEST_DATA}"

# Convert JSONL to JSON for evaluation script
EXP2_TEST_JSON="/tmp/exp2_test_${SLURM_JOB_ID}.json"
echo "Converting Exp2 test data from JSONL to JSON..."
python3 -c "
import json
with open('${EXP2_TEST_DATA}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${EXP2_TEST_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"

if [ ! -f "${EXP2_TEST_JSON}" ]; then
    echo "ERROR: Failed to convert test data"
    exit 1
fi

# Run evaluation
echo ""
echo "=========================================="
echo "Running Exp2 Evaluation"
echo "=========================================="
echo "Model: ${EXP2_MODEL}"
echo "Test data: ${EXP2_TEST_JSON}"
echo "Output: ${EXP2_OUTPUT}"
echo ""

python3 ${SCRIPTS_DIR}/evaluate_exp2.py \
    --model_path ${EXP2_MODEL} \
    --test_data ${EXP2_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP2_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct

EXIT_CODE=$?

# Cleanup
rm -f ${EXP2_TEST_JSON}

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Exp2 Evaluation completed successfully!"
    echo "Results saved to: ${EXP2_OUTPUT}"
else
    echo "✗ Exp2 Evaluation failed with exit code: ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}









