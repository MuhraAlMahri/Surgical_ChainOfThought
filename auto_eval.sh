#!/bin/bash
#SBATCH --job-name=eval_after_train
#SBATCH --dependency=afterok:144796
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_auto_%j.out
#SBATCH --error=logs/eval_auto_%j.err

# Your evaluation command here
python scripts/evaluate_models.py \
  --simple_model path/to/simple_model.pt \
  --surgical_model runs/kvasir_mllm/qwen2vl_surgical/checkpoints/best_model.pt \
  --output_dir runs/kvasir_mllm/surgical_eval
