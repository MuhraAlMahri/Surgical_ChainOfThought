#!/bin/bash
#SBATCH --job-name=MEGA_JOB_2
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --dependency=afterok:153084
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/MEGA_JOB_2_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/MEGA_JOB_2_%j.err"

echo "════════════════════════════════════════════════════════════════════"
echo "MEGA-JOB 2: Final Training + Evaluations"
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
mkdir -p /tmp/mega_job_2_${SLURM_JOB_ID}
SYNC_DIR="/tmp/mega_job_2_${SLURM_JOB_ID}"

echo "════════════════════════════════════════════════════════════════════"
echo "Starting 4 parallel processes..."
echo "════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# GPU 0: Exp4 Stage 3 (10h, waits for Stage 2 from Mega-Job 1)
# ============================================================================
(
  export CUDA_VISIBLE_DEVICES=0
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
  export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
  export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu0
  mkdir -p $HF_HOME
  
  echo "[GPU 0] Starting Exp4 Stage 3..."
  
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
    2>&1 | sed 's/^/[GPU 0 - Exp4 S3] /'
  
  echo "[GPU 0] Exp4 Stage 3 COMPLETED at $(date)"
  touch ${SYNC_DIR}/exp4_s3_done
) &
PID_GPU0=$!

# ============================================================================
# GPU 1: Exp3 Evaluation (2h, waits for Exp3 S3)
# ============================================================================
(
  export CUDA_VISIBLE_DEVICES=1
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
  export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
  export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu1
  mkdir -p $HF_HOME
  
  echo "[GPU 1] Starting Exp3 Evaluation (will wait for model to be available)..."
  EXP3_S3_MODEL="${BASE_DIR}/corrected 1-5 experiments/models/exp3_cxrtrek_seq/stage3"
  # Model should be available from Mega-Job 1
  
  echo "[GPU 1] Starting Exp3 Evaluation..."
  
  python3 ${BASE_DIR}/evaluation/evaluate_vl_model.py \
    --model_path "$EXP3_S3_MODEL" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp3_cxrtrek_sequential_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 1 - Exp3 Eval] /'
  
  echo "[GPU 1] Exp3 Evaluation COMPLETED at $(date)"
) &
PID_GPU1=$!

# ============================================================================
# GPU 2: Exp4 Evaluation (2h, waits for Exp4 S3)
# ============================================================================
(
  export CUDA_VISIBLE_DEVICES=2
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
  export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
  export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu2
  mkdir -p $HF_HOME
  
  echo "[GPU 2] Waiting for Exp4 Stage 3 to complete..."
  wait $PID_GPU0
  
  echo "[GPU 2] Starting Exp4 Evaluation..."
  
  python3 ${BASE_DIR}/evaluation/evaluate_vl_model.py \
    --model_path "${BASE_DIR}/corrected 1-5 experiments/models/exp4_curriculum/stage3" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp4_curriculum_learning_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 2 - Exp4 Eval] /'
  
  echo "[GPU 2] Exp4 Evaluation COMPLETED at $(date)"
) &
PID_GPU2=$!

# ============================================================================
# GPU 3: Exp5 Evaluation (2h, waits for Exp5 training from separate job)
# ============================================================================
(
  export CUDA_VISIBLE_DEVICES=3
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
  export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
  export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_gpu3
  mkdir -p $HF_HOME
  
  echo "[GPU 3] Waiting for Exp5 Training to complete (from separate job)..."
  EXP5_MODEL="${BASE_DIR}/corrected 1-5 experiments/models/exp5_sequential_cot"
  # Exp5 training is running separately, will complete before this job starts
  
  echo "[GPU 3] Starting Exp5 Evaluation..."
  
  python3 ${BASE_DIR}/corrected\ 1-5\ experiments/experiments/exp5_sequential_cot/evaluate_sequential_cot.py \
    --model_path "$EXP5_MODEL" \
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
echo "All 4 processes started!"
echo "  GPU 0 (PID $PID_GPU0): Exp4 Stage 3"
echo "  GPU 1 (PID $PID_GPU1): Exp3 Evaluation"
echo "  GPU 2 (PID $PID_GPU2): Exp4 Evaluation"
echo "  GPU 3 (PID $PID_GPU3): Exp5 Evaluation"
echo ""
echo "Waiting for all processes to complete..."
echo ""

# Wait for all background processes
wait $PID_GPU0
wait $PID_GPU1
wait $PID_GPU2
wait $PID_GPU3

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "MEGA-JOB 2 COMPLETED!"
echo "End time: $(date)"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ Exp4 Stage 3"
echo "  ✅ Exp3 Evaluation"
echo "  ✅ Exp4 Evaluation"
echo "  ✅ Exp5 Evaluation"
echo ""
echo "🎉 ALL EXPERIMENTS COMPLETE! 🎉"
echo ""

# Cleanup
rm -rf ${SYNC_DIR}
