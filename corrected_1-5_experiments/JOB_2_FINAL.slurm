#!/bin/bash
#SBATCH --job-name=JOB2_FIN
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --dependency=afterok:153088
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/JOB2_FIN_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/JOB2_FIN_%j.err"

echo "════════════════════════════════════════════════════════════════════"
echo "JOB 2: Final tasks (Exp4 Stage 3 + Eval)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "════════════════════════════════════════════════════════════════════"

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets"

nvidia-smi --list-gpus

# GPU 0: Exp4 Stage 3
(
  export CUDA_VISIBLE_DEVICES=0
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
  mkdir -p $HF_HOME
  
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
    2>&1 | sed 's/^/[GPU 0] /'
) &
PID_GPU0=$!

wait $PID_GPU0

# GPU 1: Exp4 Eval
(
  export CUDA_VISIBLE_DEVICES=1
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
  mkdir -p $HF_HOME
  
  python3 ${BASE_DIR}/evaluation/evaluate_vl_model.py \
    --model_path "${BASE_DIR}/corrected 1-5 experiments/models/exp4_curriculum/stage3" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp4_curriculum_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 1] /'
) &

wait
echo "JOB 2 COMPLETE at $(date)"

