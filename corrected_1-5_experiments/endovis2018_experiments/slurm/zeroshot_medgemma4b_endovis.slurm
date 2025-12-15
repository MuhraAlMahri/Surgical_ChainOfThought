#!/bin/bash
#SBATCH --job-name=zeroshot_medgemma_endovis
#SBATCH --output=slurm/logs/zeroshot_medgemma4b_endovis_%j.out
#SBATCH --error=slurm/logs/zeroshot_medgemma4b_endovis_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos

# Zero-Shot Evaluation for EndoVis2018 using MedGemma-4B
# Evaluates the base MedGemma-4B model WITHOUT any fine-tuning

set -e

echo "=========================================="
echo "ZERO-SHOT EVALUATION: EndoVis2018 (MedGemma-4B)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Base directories
BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
SCRIPT_DIR="${BASE_DIR}/scripts/evaluation"
EXPERIMENTS_DIR="${BASE_DIR}/endovis2018_experiments"
RESULTS_DIR="${EXPERIMENTS_DIR}/results"
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"

# Base model
BASE_MODEL="google/medgemma-4b-it"

# Test data paths - USING CORRECTED DATASET
TEST_DATA_JSONL="${BASE_DIR}/datasets/endovis2018_vqa/test.jsonl"
TEST_DATA_JSON="/tmp/zeroshot_endovis_medgemma_${SLURM_JOB_ID}.json"

# Create results directory
mkdir -p "${RESULTS_DIR}"
mkdir -p "${EXPERIMENTS_DIR}/slurm/logs"

# Hugging Face authentication for gated models (MedGemma-4B)
export HF_TOKEN="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"

# Convert JSONL to JSON and verify
echo "Using CORRECTED dataset: ${TEST_DATA_JSONL}"
python3 << PYTHON_EOF
import json
import sys

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
    
    total = len(data)
    print(f"✓ Loaded {total} test samples from corrected dataset")
    
    # Quick check for real answers
    instrument_samples = [s for s in data if isinstance(s, dict) and s.get('question_type') == 'instrument_detection']
    anatomy_samples = [s for s in data if isinstance(s, dict) and s.get('question_type') == 'anatomy_detection']
    count_samples = [s for s in data if isinstance(s, dict) and s.get('question_type') == 'instrument_count']
    
    instrument_none = sum(1 for s in instrument_samples if s.get('answer', '').lower() == 'none')
    anatomy_none = sum(1 for s in anatomy_samples if s.get('answer', '').lower() == 'none')
    count_zero = sum(1 for s in count_samples if s.get('answer', '') == '0')
    
    print(f"  Instrument Detection: {len(instrument_samples)} samples ({len(instrument_samples) - instrument_none} with real answers)")
    print(f"  Anatomy Detection: {len(anatomy_samples)} samples ({len(anatomy_samples) - anatomy_none} with real answers)")
    print(f"  Instrument Count: {len(count_samples)} samples ({len(count_samples) - count_zero} with real answers)")
    
except Exception as e:
    print(f"Error reading test data: {e}")
    sys.exit(1)
PYTHON_EOF

# Run zero-shot evaluation
echo ""
echo "=========================================="
echo "Running Zero-Shot Evaluation (MedGemma-4B)"
echo "=========================================="

python3 "${SCRIPT_DIR}/evaluate_zeroshot.py" \
    --base_model "${BASE_MODEL}" \
    --test_data "${TEST_DATA_JSON}" \
    --image_dir "${IMAGE_DIR}" \
    --output "${RESULTS_DIR}/endovis2018_zeroshot_medgemma4b.json" \
    --use_instruction

echo ""
echo "=========================================="
echo "ZERO-SHOT EVALUATION COMPLETED"
echo "=========================================="
echo "Results saved to: ${RESULTS_DIR}/endovis2018_zeroshot_medgemma4b.json"
echo "=========================================="

# Clean up temporary file
rm -f "${TEST_DATA_JSON}"

echo "Zero-shot evaluation completed successfully!"
