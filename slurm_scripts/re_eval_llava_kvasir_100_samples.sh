#!/bin/bash
#SBATCH --job-name=re_eval_llava_kvasir
#SBATCH --output=slurm/logs/re_eval_llava_kvasir_100_samples_%j.out
#SBATCH --error=slurm/logs/re_eval_llava_kvasir_100_samples_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Re-Evaluate LLaVA-Med Kvasir (100 samples, FIXED logic) ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Set working directory
cd /l/users/muhra.almahri/Surgical_COT

# Result file to re-evaluate
RESULT_FILE="corrected_1-5_experiments/qlora_experiments/results/kvasir_finetuned_llava_med_v15.json"
MAX_SAMPLES=100

echo "📋 Configuration:"
echo "   Result file: $RESULT_FILE"
echo "   Max samples: $MAX_SAMPLES"
echo "   Purpose: Verify evaluation bug fix"
echo ""

# Check file exists
if [ ! -f "$RESULT_FILE" ]; then
    echo "❌ ERROR: Result file not found: $RESULT_FILE"
    echo ""
    echo "Available result files:"
    ls -lh corrected_1-5_experiments/qlora_experiments/results/kvasir*.json 2>/dev/null | head -5
    exit 1
fi

# Run re-evaluation
echo "Starting re-evaluation with FIXED logic..."
echo ""

python3 re_eval_llava_kvasir_100_samples.py "$RESULT_FILE" "$MAX_SAMPLES"

EXIT_CODE=$?

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║  RE-EVALUATION COMPLETED SUCCESSFULLY                  ║"
else
    echo "║  RE-EVALUATION FAILED                                  ║"
fi
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

# Check if summary file was created
if [ -f "re_eval_llava_kvasir_summary.json" ]; then
    echo "✓ Summary saved to: re_eval_llava_kvasir_summary.json"
    echo ""
    echo "Summary:"
    python3 -c "import json; d=json.load(open('re_eval_llava_kvasir_summary.json')); print(f\"  Old Accuracy: {d['accuracy_old']:.2f}%\"); print(f\"  New Accuracy: {d['accuracy_new']:.2f}%\"); print(f\"  Difference: {d['difference']:.2f}%\"); print(f\"  Empty Predictions: {d['empty_predictions']}/{d['total_samples']}\")"
fi

echo ""



