#!/bin/bash
#SBATCH --job-name=eval_surgical
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/Kvasir-pilot/logs/eval_auto_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=03:00:00
#SBATCH -q cscc-gpu-qos
#SBATCH --dependency=afterok:144796

source /l/users/muhra.almahri/miniconda3/bin/activate surgical_cot

cd /l/users/muhra.almahri/Surgical_COT/Kvasir-pilot

python scripts/compare_models.py \
  --simple_model runs/kvasir_mllm/qwen2vl_simple/checkpoints/qwen2vl_simple_final.pt \
  --surgical_model runs/kvasir_mllm/qwen2vl_surgical/checkpoints/best_model.pt \
  --data_path mllm_training_data/kvasir_test.json \
  --output_dir runs/kvasir_mllm/surgical_eval \
  --num_samples 200
