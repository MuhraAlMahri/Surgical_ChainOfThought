#!/bin/bash
#SBATCH --job-name=JOB1_ESS
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/JOB1_ESS_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/JOB1_ESS_%j.err"

echo "════════════════════════════════════════════════════════════════════"
echo "JOB 1: Essential tasks (no dependencies)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "════════════════════════════════════════════════════════════════════"

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets"

nvidia-smi --list-gpus

# GPU 0: Exp2 Eval (FIXED)
(
  export CUDA_VISIBLE_DEVICES=0
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu0
  mkdir -p $HF_HOME
  
  python3 ${BASE_DIR}/evaluation/evaluate_vl_model.py \
    --model_path "${BASE_DIR}/corrected 1-5 experiments/models/exp2_qwen_reordered" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp2_qwen_reordered_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 0] /'
) &

# GPU 1: Exp4 Stage 2
(
  export CUDA_VISIBLE_DEVICES=1
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu1
  mkdir -p $HF_HOME
  
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
    2>&1 | sed 's/^/[GPU 1] /'
) &

# GPU 2: Exp3 Eval (uses saved model)
(
  export CUDA_VISIBLE_DEVICES=2
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu2
  mkdir -p $HF_HOME
  
  python3 ${BASE_DIR}/evaluation/evaluate_vl_model.py \
    --model_path "${BASE_DIR}/corrected 1-5 experiments/models/exp3_cxrtrek_seq/stage3" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp3_cxrtrek_sequential_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 2] /'
) &

# GPU 3: Exp5 Eval
(
  export CUDA_VISIBLE_DEVICES=3
  export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}_gpu3
  mkdir -p $HF_HOME
  
  python3 "${BASE_DIR}/corrected 1-5 experiments/experiments/exp5_sequential_cot/evaluate_sequential_cot.py" \
    --model_path "${BASE_DIR}/corrected 1-5 experiments/models/exp5_sequential_cot" \
    --test_data "${DATASET_DIR}/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "${BASE_DIR}/datasets/Kvasir-VQA/raw/images" \
    --output "${BASE_DIR}/corrected 1-5 experiments/results/exp5_sequential_cot_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct" \
    --max_new_tokens 128 \
    2>&1 | sed 's/^/[GPU 3] /'
) &

wait
echo "JOB 1 COMPLETE at $(date)"


