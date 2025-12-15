#!/bin/bash
#SBATCH --job-name=medgemma_endovis_cot
#SBATCH --output=slurm/logs/medgemma_endovis_cot_%j.out
#SBATCH --error=slurm/logs/medgemma_endovis_cot_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00

echo "================================================"
echo "Training: MedGemma on EndoVis with Multi-Head CoT"
echo "Start: $(date)"
echo "================================================"

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export HF_TOKEN=${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}

python train_multihead_cot_v3.py \
    --model_name medgemma \
    --dataset endovis \
    --checkpoint_path corrected_1-5_experiments/qlora_experiments/models/medgemma_endovis_instruction/best_model \
    --output_dir results/medgemma_endovis_cot \
    --device cuda

echo "================================================"
echo "End: $(date)"
echo "================================================"




