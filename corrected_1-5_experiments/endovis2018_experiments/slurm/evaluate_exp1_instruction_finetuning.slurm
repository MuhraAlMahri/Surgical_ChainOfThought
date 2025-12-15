#!/bin/bash
#SBATCH --job-name=eval_exp1_endovis
#SBATCH --output=slurm/logs/evaluate_exp1_instruction_finetuning_%j.out
#SBATCH --error=slurm/logs/evaluate_exp1_instruction_finetuning_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Evaluate Qwen3-VL EndoVis2018 Exp1 Instruction Fine-tuned Model
# Model trained on R1 split (1560 train frames, 228 val frames)

echo "=========================================="
echo "EVALUATING ENDOVIS2018 EXP1 INSTRUCTION FINE-TUNED MODEL"
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
MODEL_PATH="endovis2018_experiments/models/exp1_random_instruction_r1"

# Test dataset (using the standard test.jsonl)
TEST_DATA="datasets/endovis2018_vqa/test.jsonl"

# Image directory
IMAGE_DIR="/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"

# Output file
OUTPUT_FILE="endovis2018_experiments/results/endovis2018_exp1_instruction_finetuning_test.json"

# Base model
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

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

# Run evaluation
echo "Starting evaluation..."
python3 scripts/evaluation/evaluate_exp1.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_DATA" \
    --image_dir "$IMAGE_DIR" \
    --output "$OUTPUT_FILE" \
    --base_model "$BASE_MODEL"

EXIT_CODE=$?

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

