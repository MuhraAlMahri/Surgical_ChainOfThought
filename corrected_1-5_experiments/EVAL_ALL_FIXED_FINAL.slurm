#!/bin/bash
#SBATCH --job-name=EVAL_FIX
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_all_fixed_final_%j.out
#SBATCH --error=logs/eval_all_fixed_final_%j.err

echo "========================================================================="
echo "🔧 FIXED EVALUATION JOB - ALL EXPERIMENTS (FINAL)"
echo "========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load modules and activate environment (same as successful Job 153088)
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1

# Set paths
SCRIPT_DIR="/l/users/muhra.almahri/Surgical_COT/evaluation"
MODEL_DIR="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/models"
DATASET_DIR="/l/users/muhra.almahri/Surgical_COT/corrected experiments/datasets"
RESULTS_DIR="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "========================================================================="
echo "📊 Exp1: Random Baseline"
echo "========================================================================="
python3 "$SCRIPT_DIR/evaluate_vl_model.py" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --model_path "$MODEL_DIR/exp1_random_baseline" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15" \
    --output "$RESULTS_DIR/exp1_random_baseline_results_FIXED.json" \
    --max_new_tokens 128

echo ""
echo "✅ Exp1 Complete: $(date)"
echo ""

echo "========================================================================="
echo "📊 Exp2: Qwen Reordered"
echo "========================================================================="
python3 "$SCRIPT_DIR/evaluate_vl_model.py" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --model_path "$MODEL_DIR/exp2_qwen_reordered" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15" \
    --output "$RESULTS_DIR/exp2_qwen_reordered_results_FIXED.json" \
    --max_new_tokens 128

echo ""
echo "✅ Exp2 Complete: $(date)"
echo ""

echo "========================================================================="
echo "📊 Exp3: CXR-TREK Sequential (Stage 3)"
echo "========================================================================="
python3 "$SCRIPT_DIR/evaluate_vl_model.py" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --model_path "$MODEL_DIR/exp3_cxrtrek_seq/stage3" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15" \
    --output "$RESULTS_DIR/exp3_cxrtrek_sequential_results_FIXED.json" \
    --max_new_tokens 128

echo ""
echo "✅ Exp3 Complete: $(date)"
echo ""

echo "========================================================================="
echo "📊 Exp4: Curriculum Learning (Stage 3)"
echo "========================================================================="
python3 "$SCRIPT_DIR/evaluate_vl_model.py" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --model_path "$MODEL_DIR/exp4_curriculum/stage3" \
    --test_data "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "$DATASET_DIR/kvasir_raw_6500_image_level_70_15_15" \
    --output "$RESULTS_DIR/exp4_curriculum_results_FIXED.json" \
    --max_new_tokens 128

echo ""
echo "✅ Exp4 Complete: $(date)"
echo ""

echo "========================================================================="
echo "🎉 ALL EVALUATIONS COMPLETE!"
echo "========================================================================="
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  - $RESULTS_DIR/exp1_random_baseline_results_FIXED.json"
echo "  - $RESULTS_DIR/exp2_qwen_reordered_results_FIXED.json"
echo "  - $RESULTS_DIR/exp3_cxrtrek_sequential_results_FIXED.json"
echo "  - $RESULTS_DIR/exp4_curriculum_results_FIXED.json"
echo ""

