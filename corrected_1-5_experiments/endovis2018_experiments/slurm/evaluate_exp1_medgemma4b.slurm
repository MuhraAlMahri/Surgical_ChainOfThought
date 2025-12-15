#!/bin/bash
#SBATCH --job-name=eval_exp1_medgemma_endovis
#SBATCH --output=slurm/logs/evaluate_exp1_medgemma4b_endovis_%j.out
#SBATCH --error=slurm/logs/evaluate_exp1_medgemma4b_endovis_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Evaluate EndoVis2018 MedGemma-4B Exp1 Instruction Fine-tuned Model
# Model trained on R1 split (1560 train frames, 228 val frames)

echo "=========================================="
echo "EVALUATING ENDOVIS2018 EXP1 MEDGEMMA-4B INSTRUCTION FINE-TUNED MODEL"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate surgical_cot

# Set paths
SCRIPT_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
cd "$SCRIPT_DIR"

# Model path (final trained model)
MODEL_PATH="endovis2018_experiments/models/exp1_medgemma4b_instruction_r1"

# Test dataset (using the standard test.jsonl)
TEST_DATA="datasets/endovis2018_vqa/test.jsonl"

# Image directory
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"

# Output file
OUTPUT_FILE="endovis2018_experiments/results/endovis2018_exp1_medgemma4b_instruction_test.json"

# Base model
BASE_MODEL="google/medgemma-4b-it"

# Hugging Face token for gated models
export HF_TOKEN="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"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test data: $TEST_DATA"
echo "  Image dir: $IMAGE_DIR"
echo "  Output: $OUTPUT_FILE"
echo "  Base model: $BASE_MODEL"
echo ""

# Check if files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "ERROR: Test data file not found: $TEST_DATA"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image directory not found: $IMAGE_DIR"
    exit 1
fi

# Convert JSONL to JSON for evaluation script
TEMP_JSON="/tmp/test_endovis_medgemma_${SLURM_JOB_ID}.json"
echo "Converting JSONL to JSON..."
python3 << PYTHON_EOF
import json
import sys

jsonl_path = "${TEST_DATA}"
json_path = "${TEMP_JSON}"

with open(jsonl_path, 'r') as f_in:
    data = [json.loads(line) for line in f_in]

with open(json_path, 'w') as f_out:
    json.dump(data, f_out, indent=2)

print(f"✓ Converted {len(data)} samples from JSONL to JSON")
PYTHON_EOF

# Run evaluation
echo "Starting evaluation..."
python3 scripts/evaluation/evaluate_exp1.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEMP_JSON" \
    --image_dir "$IMAGE_DIR" \
    --output "$OUTPUT_FILE" \
    --base_model "$BASE_MODEL"

EXIT_CODE=$?

# Clean up temp file
rm -f "$TEMP_JSON"

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "EVALUATION COMPLETED SUCCESSFULLY"
    echo "Results saved to: $OUTPUT_FILE"
else
    echo "EVALUATION FAILED (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE


