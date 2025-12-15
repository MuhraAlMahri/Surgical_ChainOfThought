#!/bin/bash
#SBATCH --job-name=exp1_eval_no_instr
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp1_eval_no_instr_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/exp1_eval_no_instr_%j.err

# ============================================================================
# Exp1 Evaluation WITHOUT Instructions
# ============================================================================
# This job runs Exp1 evaluation using only the question field (no instructions)
# ============================================================================

echo "=========================================="
echo "Exp1 Evaluation (WITHOUT INSTRUCTIONS)"
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
# Exp1 Evaluation (WITHOUT INSTRUCTIONS)
# ============================================================================
EXP1_MODEL="${BASE_DIR}/models/exp1_random"
EXP1_TEST_DATA="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl"
EXP1_OUTPUT="${RESULTS_DIR}/exp1_evaluation_no_instruction.json"

# Verify model exists
if [ ! -d "${EXP1_MODEL}" ]; then
    echo "ERROR: Exp1 model not found at ${EXP1_MODEL}"
    exit 1
fi
echo "✓ Exp1 model found: ${EXP1_MODEL}"

# Verify test data exists
if [ ! -f "${EXP1_TEST_DATA}" ]; then
    echo "ERROR: Exp1 test data not found at ${EXP1_TEST_DATA}"
    exit 1
fi
echo "✓ Exp1 test data found: ${EXP1_TEST_DATA}"

# Convert JSONL to JSON for evaluation script
EXP1_TEST_JSON="/tmp/exp1_test_no_instr_${SLURM_JOB_ID}.json"
echo "Converting Exp1 test data from JSONL to JSON..."
python3 -c "
import json
with open('${EXP1_TEST_DATA}', 'r') as f:
    data = [json.loads(line) for line in f]
with open('${EXP1_TEST_JSON}', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Converted {len(data)} samples')
"

if [ ! -f "${EXP1_TEST_JSON}" ]; then
    echo "ERROR: Failed to convert test data"
    exit 1
fi

# Run evaluation
echo ""
echo "=========================================="
echo "Running Exp1 Evaluation (WITHOUT INSTRUCTIONS)"
echo "=========================================="
echo "Model: ${EXP1_MODEL}"
echo "Test data: ${EXP1_TEST_JSON}"
echo "Output: ${EXP1_OUTPUT}"
echo "Mode: Using question field only (no instructions)"
echo ""

python3 ${SCRIPTS_DIR}/evaluate_exp1.py \
    --model_path ${EXP1_MODEL} \
    --test_data ${EXP1_TEST_JSON} \
    --image_dir ${IMAGE_ROOT} \
    --output ${EXP1_OUTPUT} \
    --base_model Qwen/Qwen3-VL-8B-Instruct \
    --use_question_only

EXIT_CODE=$?

# Cleanup
rm -f ${EXP1_TEST_JSON}

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Exp1 Evaluation (WITHOUT INSTRUCTIONS) completed successfully!"
    echo "Results saved to: ${EXP1_OUTPUT}"
    
    # Display summary
    echo ""
    echo "Summary:"
    python3 -c "
import json
with open('${EXP1_OUTPUT}', 'r') as f:
    results = json.load(f)
print(f\"Total: {results['total']}\")
print(f\"Correct: {results['correct']}\")
print(f\"Accuracy: {results['accuracy']:.2f}%\")
print(\"\nBy Stage:\")
for stage in sorted(results['by_stage'].keys()):
    s = results['by_stage'][stage]
    print(f\"  {stage}: {s['accuracy']:.2f}% ({s['correct']}/{s['total']})\")
"
else
    echo "✗ Exp1 Evaluation failed with exit code: ${EXIT_CODE}"
fi
echo "End time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}




