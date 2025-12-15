
#!/bin/bash

#SBATCH --job-name=exp3_s1

#SBATCH -p cscc-gpu-p

#SBATCH -q cscc-gpu-qos

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --mem=128G

#SBATCH --time=24:00:00

#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp3_s1_%j.out"

#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/exp3_s1_%j.err"

echo "=========================================="

echo "EXPERIMENT 3: Stage 1 (FIXED)"

echo "Start time: $(date)"

echo "=========================================="

module load nvidia/cuda/12.0

source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=0

export PYTHONUNBUFFERED=1

export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}

export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}

mkdir -p $HF_HOME

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"

DATASET="${BASE_DIR}/corrected 1-5 experiments/datasets/kvasir_stage_splits_stage1/train.json"

OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/models/exp3_stage1"

nvidia-smi

echo ""

python3 ${BASE_DIR}/training/train_qwen_lora.py \

    --data_path "$DATASET" \

    --output_dir "$OUTPUT_DIR" \

    --model_name "Qwen/Qwen2-VL-7B-Instruct" \

    --num_train_epochs 3 \

    --per_device_train_batch_size 1 \

    --gradient_accumulation_steps 16 \

    --learning_rate 5e-6 \

    --max_seq_length 512 \

    --logging_steps 50 \

    --save_strategy epoch \

    --save_total_limit 2

echo "Stage 1 completed: $(date)"

