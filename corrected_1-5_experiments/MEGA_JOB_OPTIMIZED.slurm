#!/bin/bash
#SBATCH --job-name=MEGA_OPT
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/MEGA_OPT_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/MEGA_OPT_%j.err"

echo "════════════════════════════════════════════════════════════════════"
echo "MEGA-JOB OPTIMIZED: Only run what's needed!"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "════════════════════════════════════════════════════════════════════"
echo ""

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets"

echo "GPUs allocated:"
nvidia-smi --list-gpus
echo ""

# Create sync directory
mkdir -p /tmp/mega_job_opt_${SLURM_JOB_ID}
SYNC_DIR="/tmp/mega_job_opt_${SLURM_JOB_ID}"

echo "════════════════════════════════════════════════════════════════════"
echo "SKIPPING already completed tasks:"
echo "  ✅ Exp3 Stage 1 (saved Nov 04 22:48)"
echo "  ✅ Exp3 Stage 3 (saved Nov 04 19:11)"
echo "  ✅ Exp4 Stage 1 (saved Nov 04 22:48)"
echo ""
echo "RUNNING only what's needed:"
echo "  🔄 Exp2 Evaluation (FIXED)"
echo "  🔄 Exp4 Stage 2"
echo "  🔄 Exp4 Stage 3"
echo "  🔄 All final evaluations"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# GPU 0: Exp2 Eval (2h) → then Exp3 Eval (2h)
# ============================================================================
(
  export CUDA_VISIBLE_DEVICES=0
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
  export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
  export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu0
  mkdir -p $HF_HOME
  
  echo "[GPU 0] Starting Exp2 Evaluation (FIXED - using test.json)..."
  
  python3 ${BASE_DIR}/evaluation/evaluate_vl_model.py \
    --model_path "${BASE_DIR}/corrected 1-5 experiments/models/exp2_qwen_reordered" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp2_qwen_reordered_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 0 - Exp2 Eval] /'
  
  echo "[GPU 0] Exp2 Evaluation COMPLETED at $(date)"
  
  echo "[GPU 0] Starting Exp3 Evaluation..."
  
  python3 ${BASE_DIR}/evaluation/evaluate_vl_model.py \
    --model_path "${BASE_DIR}/corrected 1-5 experiments/models/exp3_cxrtrek_seq/stage3" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp3_cxrtrek_sequential_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 0 - Exp3 Eval] /'
  
  echo "[GPU 0] Exp3 Evaluation COMPLETED at $(date)"
) &
PID_GPU0=$!

# ============================================================================
# GPU 1: Exp4 Stage 2 (12h)
# ============================================================================
(
  export CUDA_VISIBLE_DEVICES=1
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
  export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
  export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu1
  mkdir -p $HF_HOME
  
  echo "[GPU 1] Starting Exp4 Stage 2 (uses saved Stage 1 model)..."
  
  python3 ${BASE_DIR}/training/train_qwen_lora.py \
    --train_file "${DATASET_DIR}/kvasir_stage_splits_stage2/train.json" \
    --val_file "${DATASET_DIR}/kvasir_stage_splits_stage2/val.json" \
    --output_dir "${BASE_DIR}/corrected 1-5 experiments/models/exp4_curriculum/stage2" \
    --prev_checkpoint "${BASE_DIR}/corrected 1-5 experiments/models/exp4_curriculum/stage1" \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --num_train_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --dataset_type "reordered" \
    --max_length 512 \
    2>&1 | sed 's/^/[GPU 1 - Exp4 S2] /'
  
  echo "[GPU 1] Exp4 Stage 2 COMPLETED at $(date)"
  touch ${SYNC_DIR}/exp4_s2_done
) &
PID_GPU1=$!

# ============================================================================
# GPU 2: Exp4 Stage 3 (12h, waits for Stage 2) → then Exp4 Eval
# ============================================================================
(
  export CUDA_VISIBLE_DEVICES=2
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
  export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
  export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu2
  mkdir -p $HF_HOME
  
  echo "[GPU 2] Waiting for Exp4 Stage 2 to complete..."
  wait $PID_GPU1
  
  echo "[GPU 2] Starting Exp4 Stage 3..."
  
  python3 ${BASE_DIR}/training/train_qwen_lora.py \
    --train_file "${DATASET_DIR}/kvasir_stage_splits_stage3/train.json" \
    --val_file "${DATASET_DIR}/kvasir_stage_splits_stage3/val.json" \
    --output_dir "${BASE_DIR}/corrected 1-5 experiments/models/exp4_curriculum/stage3" \
    --prev_checkpoint "${BASE_DIR}/corrected 1-5 experiments/models/exp4_curriculum/stage2" \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --num_train_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --dataset_type "reordered" \
    --max_length 512 \
    2>&1 | sed 's/^/[GPU 2 - Exp4 S3] /'
  
  echo "[GPU 2] Exp4 Stage 3 COMPLETED at $(date)"
  
  echo "[GPU 2] Starting Exp4 Evaluation..."
  
  python3 ${BASE_DIR}/evaluation/evaluate_vl_model.py \
    --model_path "${BASE_DIR}/corrected 1-5 experiments/models/exp4_curriculum/stage3" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp4_curriculum_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 2 - Exp4 Eval] /'
  
  echo "[GPU 2] Exp4 Evaluation COMPLETED at $(date)"
) &
PID_GPU2=$!

# ============================================================================
# GPU 3: Exp5 Evaluation (2h)
# ============================================================================
(
  export CUDA_VISIBLE_DEVICES=3
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
  export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
  export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu3
  mkdir -p $HF_HOME
  
  echo "[GPU 3] Starting Exp5 Evaluation..."
  
  python3 "${BASE_DIR}/corrected 1-5 experiments/experiments/exp5_sequential_cot/evaluate_sequential_cot.py" \
    --model_path "${BASE_DIR}/corrected 1-5 experiments/models/exp5_sequential_cot" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp5_sequential_cot_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 3 - Exp5 Eval] /'
  
  echo "[GPU 3] Exp5 Evaluation COMPLETED at $(date)"
) &
PID_GPU3=$!

echo ""
echo "All processes started!"
echo "  GPU 0 (PID $PID_GPU0): Exp2 Eval → Exp3 Eval"
echo "  GPU 1 (PID $PID_GPU1): Exp4 Stage 2"
echo "  GPU 2 (PID $PID_GPU2): Exp4 Stage 3 → Exp4 Eval"
echo "  GPU 3 (PID $PID_GPU3): Exp5 Eval"
echo ""
echo "Waiting for all processes to complete..."
echo ""

wait

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "ALL TASKS COMPLETED!"
echo "End time: $(date)"
echo "════════════════════════════════════════════════════════════════════"


