#!/bin/bash
#SBATCH --job-name=eval_epoch5_full
#SBATCH --output=slurm/logs/eval_epoch5_full_%j.out
#SBATCH --error=slurm/logs/eval_epoch5_full_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --partition=cscc-gpu-p

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  FULL EVALUATION: Zeroshot+COT (Epoch 5)                ║"
echo "║  Evaluating on all 8,984 test samples                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Start time: $(date)"
echo ""

# Activate conda environment
source activate surgical_cot

# Set working directory
cd /l/users/muhra.almahri/Surgical_COT

# Run comprehensive evaluation
echo "Running comprehensive evaluation..."
echo ""

# Epoch 5 was trained from BASE Qwen3-VL model (Zeroshot+COT experiment)
# NOT from exp1_random checkpoint
BASE_CHECKPOINT="Qwen/Qwen3-VL-8B-Instruct"
echo "Using base checkpoint: $BASE_CHECKPOINT"
echo "Note: Epoch 5 checkpoint was trained from base model (Zeroshot+COT)"
echo ""

python evaluate_multihead_cot_comprehensive.py \
    --checkpoint results/qwen3vl_kvasir_cot_5epochs/checkpoint_epoch_5.pt \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --model_name qwen3vl \
    --dataset kvasir \
    --data_path /l/users/muhra.almahri/datasets/kvasir-vqa \
    --question_categories question_categories.json \
    --output_file results/qwen3vl_kvasir_zeroshot_cot_epoch5_FULL.json \
    --batch_size 1 \
    --use_flexible_matching

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo ""
    echo "📊 Final Results:"
    echo ""
    python3 -c "
import json
with open('results/qwen3vl_kvasir_zeroshot_cot_epoch5_FULL.json', 'r') as f:
    data = json.load(f)
    print(f\"Overall Accuracy: {data['overall_accuracy']:.2%} ({data['correct']}/{data['total']})\")
    print(f\"\nPer-Category Accuracies:\")
    for cat, acc in data['category_accuracies'].items():
        print(f\"  {cat}: {acc:.2%} ({data['category_correct'][cat]}/{data['category_total'][cat]})\")
"
    echo ""
    echo "📁 Results saved to: results/qwen3vl_kvasir_zeroshot_cot_epoch5_FULL.json"
else
    echo "❌ Evaluation failed with exit code: $EXIT_CODE"
    echo "Check logs for details: slurm/logs/eval_epoch5_full_${SLURM_JOB_ID}.err"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  JOB COMPLETED                                           ║"
echo "╚══════════════════════════════════════════════════════════╝"

