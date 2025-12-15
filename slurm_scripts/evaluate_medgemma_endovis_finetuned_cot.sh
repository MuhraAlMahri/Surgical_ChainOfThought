#!/bin/bash
#SBATCH --job-name=eval_medgemma_endovis_ft_cot
#SBATCH --output=slurm/logs/eval_medgemma_endovis_finetuned_cot_%j.out
#SBATCH --error=slurm/logs/eval_medgemma_endovis_finetuned_cot_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Evaluate MedGemma EndoVis Fine-tuned+COT              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/bin/activate base

# Install peft if needed
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PIP_TARGET="/l/users/muhra.almahri/.local/lib/python${PYTHON_VERSION}/site-packages"
export PYTHONPATH="${PIP_TARGET}:${PYTHONPATH}"
mkdir -p ${PIP_TARGET}
pip install --target ${PIP_TARGET} -q peft 2>/dev/null || echo "Note: peft may already be installed"

export PYTHONUNBUFFERED=1

# Set HuggingFace cache
export HF_HOME=/l/users/muhra.almahri/.cache/hf_shared
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}

# Set working directory
cd /l/users/muhra.almahri/Surgical_COT

# ============================================================================
# CONFIGURATION
# ============================================================================

# Checkpoint paths
COT_CHECKPOINT="results/medgemma_endovis_finetuned_cot/checkpoint_epoch_5.pt"
BASE_CHECKPOINT="corrected_1-5_experiments/endovis2018_experiments/models/exp1_random"

# Verify checkpoint exists
if [ ! -f "$COT_CHECKPOINT" ]; then
    echo "❌ ERROR: COT checkpoint not found: $COT_CHECKPOINT"
    exit 1
fi

# Verify base checkpoint exists
if [ ! -d "$BASE_CHECKPOINT" ]; then
    echo "⚠️  Base checkpoint not found: $BASE_CHECKPOINT"
    echo "   Using default base model: google/medgemma-4b-it"
    BASE_CHECKPOINT="google/medgemma-4b-it"
fi

# Data paths
ENDOVIS_DATA_FILE="corrected_1-5_experiments/datasets/endovis2018_vqa/test.jsonl"
ENDOVIS_IMAGE_DIR="datasets/EndoVis2018/raw/images"

# Question categories
QUESTION_CATEGORIES="question_categories.json"

# Output file
OUTPUT_FILE="results/medgemma_endovis_finetuned_cot_eval.json"

echo "📋 Configuration:"
echo "   COT Checkpoint: $COT_CHECKPOINT"
echo "   Base Checkpoint: $BASE_CHECKPOINT"
echo "   Test Data: $ENDOVIS_DATA_FILE"
echo "   Image Directory: $ENDOVIS_IMAGE_DIR"
echo "   Output: $OUTPUT_FILE"
echo ""

# ============================================================================
# PREPARE TEST DATA (Convert JSONL to JSON if needed)
# ============================================================================

ENDOVIS_TEMP_DIR="/tmp/endovis_eval_${SLURM_JOB_ID}"
mkdir -p "$ENDOVIS_TEMP_DIR"

echo "Converting test data from JSONL to JSON..."
python3 << PYTHON_EOF
import json
import sys

jsonl_file = "$ENDOVIS_DATA_FILE"
json_file = "$ENDOVIS_TEMP_DIR/test.json"

data = []
with open(jsonl_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

with open(json_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✓ Converted {len(data)} samples from JSONL to JSON")
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to convert test data"
    exit 1
fi

# ============================================================================
# EVALUATION
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting Evaluation: MedGemma EndoVis Fine-tuned+COT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Start time: $(date)"
echo ""

python evaluate_multihead_cot_comprehensive.py \
    --checkpoint "$COT_CHECKPOINT" \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --model_name medgemma \
    --dataset endovis \
    --data_path "$ENDOVIS_TEMP_DIR" \
    --image_base_path "$ENDOVIS_IMAGE_DIR" \
    --question_categories "$QUESTION_CATEGORIES" \
    --output_file "$OUTPUT_FILE" \
    --batch_size 1 \
    --use_flexible_matching

EXIT_CODE=$?

# Cleanup temp directory
rm -rf "$ENDOVIS_TEMP_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Evaluation completed at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
    echo ""
    echo "Results saved to: $OUTPUT_FILE"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "File size: $(ls -lh "$OUTPUT_FILE" | awk '{print $5}')"
        # Show a quick summary if possible
        python3 << PYTHON_EOF
import json
import sys

try:
    with open("$OUTPUT_FILE", 'r') as f:
        results = json.load(f)
    
    if 'accuracy' in results:
        print(f"\n📊 Quick Summary:")
        print(f"   Accuracy: {results['accuracy']:.2f}%")
        print(f"   Correct: {results.get('correct', 'N/A')}/{results.get('total', 'N/A')}")
    elif 'overall' in results:
        overall = results['overall']
        print(f"\n📊 Quick Summary:")
        print(f"   Accuracy: {overall.get('accuracy', 0):.2f}%")
        print(f"   Correct: {overall.get('correct', 'N/A')}/{overall.get('total', 'N/A')}")
except Exception as e:
    print(f"   (Could not parse results: {e})")
PYTHON_EOF
    fi
else
    echo "❌ Evaluation failed with exit code: $EXIT_CODE"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit $EXIT_CODE

