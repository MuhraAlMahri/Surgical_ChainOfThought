#!/bin/bash
#SBATCH --job-name=EVAL_MEGA
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_mega_%j.out
#SBATCH --error=logs/eval_mega_%j.err

echo "========================================================================="
echo "🚀 MEGA EVALUATION JOB - ALL 4 EXPERIMENTS IN PARALLEL"
echo "========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Load modules and activate environment
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1

# Use /tmp for HF cache to avoid home directory quota issues
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"

echo "Using HF_HOME: $HF_HOME"
echo ""

# Set base paths
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
SCRIPT_DIR="${BASE_DIR}/corrected 1-5 experiments/scripts/evaluation"
MODEL_DIR="${BASE_DIR}/corrected 1-5 experiments/models"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets"
RESULTS_DIR="${BASE_DIR}/corrected 1-5 experiments/results"

mkdir -p "$RESULTS_DIR"

echo "========================================================================="
echo "📊 RUNNING ALL 4 EVALUATIONS IN PARALLEL"
echo "========================================================================="
echo ""

# GPU 0: Exp1 - Random Baseline
(
export CUDA_VISIBLE_DEVICES=0
echo "[GPU 0] Starting Exp1: Random Baseline"
python3 "$SCRIPT_DIR/evaluate_exp1.py" \
    --model_path "$MODEL_DIR/exp1_random_baseline" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "$RESULTS_DIR/exp1_evaluation_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    2>&1 | sed 's/^/[GPU 0 - Exp1] /'
echo "[GPU 0] ✅ Exp1 Complete: $(date)"
) &

# GPU 1: Exp2 - Qwen Reordered
(
export CUDA_VISIBLE_DEVICES=1
echo "[GPU 1] Starting Exp2: Qwen Reordered"
python3 "$SCRIPT_DIR/evaluate_exp2.py" \
    --model_path "$MODEL_DIR/exp2_qwen_reordered" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "$RESULTS_DIR/exp2_evaluation_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    2>&1 | sed 's/^/[GPU 1 - Exp2] /'
echo "[GPU 1] ✅ Exp2 Complete: $(date)"
) &

# GPU 2: Exp3 - CXR-TREK Sequential
(
export CUDA_VISIBLE_DEVICES=2
echo "[GPU 2] Starting Exp3: CXR-TREK Sequential"
python3 "$SCRIPT_DIR/evaluate_exp3.py" \
    --model_path "$MODEL_DIR/exp3_cxrtrek_seq/stage3" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "$RESULTS_DIR/exp3_evaluation_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    2>&1 | sed 's/^/[GPU 2 - Exp3] /'
echo "[GPU 2] ✅ Exp3 Complete: $(date)"
) &

# GPU 3: Exp4 - Curriculum Learning
(
export CUDA_VISIBLE_DEVICES=3
echo "[GPU 3] Starting Exp4: Curriculum Learning"
python3 "$SCRIPT_DIR/evaluate_exp4.py" \
    --model_path "$MODEL_DIR/exp4_curriculum/stage3" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "$RESULTS_DIR/exp4_evaluation_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    2>&1 | sed 's/^/[GPU 3 - Exp4] /'
echo "[GPU 3] ✅ Exp4 Complete: $(date)"
) &

# Wait for all background jobs to complete
echo ""
echo "⏳ Waiting for all 4 evaluations to complete..."
echo ""
wait

echo ""
echo "========================================================================="
echo "🎉 ALL EVALUATIONS COMPLETE!"
echo "========================================================================="
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  - $RESULTS_DIR/exp1_evaluation_results.json"
echo "  - $RESULTS_DIR/exp2_evaluation_results.json"
echo "  - $RESULTS_DIR/exp3_evaluation_results.json"
echo "  - $RESULTS_DIR/exp4_evaluation_results.json"
echo ""

# Show result summaries
echo "========================================================================="
echo "📊 RESULT SUMMARIES"
echo "========================================================================="
echo ""

for exp in exp1 exp2 exp3 exp4; do
    result_file="$RESULTS_DIR/${exp}_evaluation_results.json"
    if [ -f "$result_file" ]; then
        echo "--- $exp ---"
        python3 -c "
import json
with open('$result_file') as f:
    data = json.load(f)
    if 'accuracy' in data:
        print(f\"  Accuracy: {data['accuracy']:.2f}%\")
    elif 'correct' in data and 'total' in data:
        acc = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f\"  Accuracy: {acc:.2f}% ({data['correct']}/{data['total']})\")
    else:
        print(f\"  Data: {list(data.keys())}\")
" 2>/dev/null || echo "  Could not parse results"
        echo ""
    fi
done

# Cleanup temp cache
rm -rf "$HF_HOME" "/tmp/triton_cache_$SLURM_JOB_ID"
echo "Cleaned up temp caches"
echo ""

echo "========================================================================="
echo "✅ MEGA JOB COMPLETE - ALL RESULTS READY!"
echo "========================================================================="

