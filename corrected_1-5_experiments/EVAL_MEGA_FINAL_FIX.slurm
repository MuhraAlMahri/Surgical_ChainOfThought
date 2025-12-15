#!/bin/bash
#SBATCH --job-name=eval_mega_final
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_mega_final_%j.out
#SBATCH --error=logs/eval_mega_final_%j.err

echo "========================================================================="
echo "🔧 MEGA EVALUATION JOB - FINAL FIX (PRE-COPY MODELS TO /tmp)"
echo "========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $SLURM_NODELIST"
echo ""

# Load modules and activate environment
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1

# Set up /tmp cache directory
export HF_HOME="/tmp/hf_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRITON_CACHE_DIR="/tmp/triton_cache_$SLURM_JOB_ID"
mkdir -p "$HF_HOME"

echo "========================================================================="
echo "📦 STEP 1: Copy Qwen2-VL Model Cache to /tmp"
echo "========================================================================="
echo "Source: ~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct"
echo "Destination: $HF_HOME/hub/"
echo ""

# Copy the cached model to /tmp (this avoids both download AND quota issues)
mkdir -p "$HF_HOME/hub"
if [ -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct" ]; then
    echo "Copying Qwen model cache..."
    cp -r "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct" "$HF_HOME/hub/"
    echo "✅ Model cache copied successfully"
else
    echo "❌ ERROR: Source model cache not found!"
    exit 1
fi

# Also copy the XET cache if it exists
if [ -d "$HOME/.cache/huggingface/xet" ]; then
    cp -r "$HOME/.cache/huggingface/xet" "$HF_HOME/"
    echo "✅ XET cache copied"
fi

echo ""
echo "Cache size in /tmp:"
du -sh "$HF_HOME"
echo ""

# Set base paths
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
SCRIPT_DIR="${BASE_DIR}/corrected 1-5 experiments/scripts/evaluation"
MODEL_DIR="${BASE_DIR}/corrected 1-5 experiments/models"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets"
RESULTS_DIR="${BASE_DIR}/corrected 1-5 experiments/results"
IMAGE_DIR="${BASE_DIR}/datasets/Kvasir-VQA/raw/images"

echo "========================================================================="
echo "🚀 STEP 2: Run Evaluations (Models already in /tmp)"
echo "========================================================================="
echo ""
echo "Using CORRECT stage-specific test datasets:"
echo "  • Stage 1: kvasir_stage_splits_stage1/test.json (3,275 samples)"
echo "  • Stage 2: kvasir_stage_splits_stage2/test.json (5,703 samples)"
echo "  • Stage 3: kvasir_stage_splits_stage3/test.json (6 samples)"
echo ""

# Run all 3 evaluations in parallel on different GPUs
(
    echo "[GPU 0] Starting Exp3 (CXR-TREK Sequential)"
    CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT_DIR/evaluate_exp3_corrected.py" \
        --model_stage1 "$MODEL_DIR/exp3_cxrtrek_seq/stage1" \
        --model_stage2 "$MODEL_DIR/exp3_cxrtrek_seq/stage2" \
        --model_stage3 "$MODEL_DIR/exp3_cxrtrek_seq/stage3" \
        --test_stage1 "$DATASET_DIR/kvasir_stage_splits_stage1/test.json" \
        --test_stage2 "$DATASET_DIR/kvasir_stage_splits_stage2/test.json" \
        --test_stage3 "$DATASET_DIR/kvasir_stage_splits_stage3/test.json" \
        --image_dir "$IMAGE_DIR" \
        --output "$RESULTS_DIR/exp3_corrected_evaluation_results.json" \
        --base_model "Qwen/Qwen2-VL-7B-Instruct"
    
    echo "[GPU 0] ✅ Exp3 Complete: $(date)"
) &

(
    echo "[GPU 1] Starting Exp4 (Curriculum Learning) - Re-evaluation"
    CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT_DIR/evaluate_exp3_corrected.py" \
        --model_stage1 "$MODEL_DIR/exp4_curriculum/stage3" \
        --model_stage2 "$MODEL_DIR/exp4_curriculum/stage3" \
        --model_stage3 "$MODEL_DIR/exp4_curriculum/stage3" \
        --test_stage1 "$DATASET_DIR/kvasir_stage_splits_stage1/test.json" \
        --test_stage2 "$DATASET_DIR/kvasir_stage_splits_stage2/test.json" \
        --test_stage3 "$DATASET_DIR/kvasir_stage_splits_stage3/test.json" \
        --image_dir "$IMAGE_DIR" \
        --output "$RESULTS_DIR/exp4_corrected_evaluation_results.json" \
        --base_model "Qwen/Qwen2-VL-7B-Instruct"
    
    echo "[GPU 1] ✅ Exp4 Complete: $(date)"
) &

(
    echo "[GPU 2] Starting Exp5 (Sequential CoT)"
    CUDA_VISIBLE_DEVICES=2 python3 "$SCRIPT_DIR/evaluate_exp5.py" \
        --model_path "$MODEL_DIR/exp5_sequential_cot" \
        --test_data "$DATASET_DIR/kvasir_stage_splits_stage1/test.json" \
        --image_dir "$IMAGE_DIR" \
        --output "$RESULTS_DIR/exp5_stage1_evaluation_results.json" \
        --base_model "Qwen/Qwen2-VL-7B-Instruct"
    
    echo "[GPU 2] ✅ Exp5 Stage 1 Complete"
    
    CUDA_VISIBLE_DEVICES=2 python3 "$SCRIPT_DIR/evaluate_exp5.py" \
        --model_path "$MODEL_DIR/exp5_sequential_cot" \
        --test_data "$DATASET_DIR/kvasir_stage_splits_stage2/test.json" \
        --image_dir "$IMAGE_DIR" \
        --output "$RESULTS_DIR/exp5_stage2_evaluation_results.json" \
        --base_model "Qwen/Qwen2-VL-7B-Instruct"
    
    echo "[GPU 2] ✅ Exp5 Stage 2 Complete"
    
    CUDA_VISIBLE_DEVICES=2 python3 "$SCRIPT_DIR/evaluate_exp5.py" \
        --model_path "$MODEL_DIR/exp5_sequential_cot" \
        --test_data "$DATASET_DIR/kvasir_stage_splits_stage3/test.json" \
        --image_dir "$IMAGE_DIR" \
        --output "$RESULTS_DIR/exp5_stage3_evaluation_results.json" \
        --base_model "Qwen/Qwen2-VL-7B-Instruct"
    
    echo "[GPU 2] ✅ Exp5 Complete: $(date)"
) &

# Wait for all background jobs to finish
wait

echo ""
echo "========================================================================="
echo "✅ ALL EVALUATIONS COMPLETE!"
echo "========================================================================="
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  • $RESULTS_DIR/exp3_corrected_evaluation_results.json"
echo "  • $RESULTS_DIR/exp4_corrected_evaluation_results.json"
echo "  • $RESULTS_DIR/exp5_stage1_evaluation_results.json"
echo "  • $RESULTS_DIR/exp5_stage2_evaluation_results.json"
echo "  • $RESULTS_DIR/exp5_stage3_evaluation_results.json"
echo ""

# Cleanup temp cache
echo "Cleaning up /tmp cache..."
rm -rf "$HF_HOME" "/tmp/triton_cache_$SLURM_JOB_ID"
echo "✅ Cleanup complete"



