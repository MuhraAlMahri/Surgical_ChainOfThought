#!/bin/bash
#SBATCH --job-name=qwen3vl_endovis_cot
#SBATCH --output=slurm/logs/qwen3vl_endovis_cot_%j.out
#SBATCH --error=slurm/logs/qwen3vl_endovis_cot_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

echo "================================================"
echo "Training: Qwen3-VL on EndoVis with Multi-Head CoT"
echo "Start: $(date)"
echo "================================================"

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export HF_TOKEN=${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}

# Set HuggingFace cache to workspace to avoid home quota issues
export HF_HOME=$SLURM_SUBMIT_DIR/.hf_cache
export TRANSFORMERS_CACHE=$SLURM_SUBMIT_DIR/.hf_cache/transformers
export HF_HUB_CACHE=$SLURM_SUBMIT_DIR/.hf_cache/hub
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_HUB_CACHE

python train_multihead_cot_v3.py \
    --model_name qwen3vl \
    --dataset endovis \
    --checkpoint_path corrected_1-5_experiments/qlora_experiments/models/qwen3vl_endovis_instruction/best_model \
    --output_dir results/qwen3vl_endovis_cot \
    --device cuda

echo "================================================"
echo "End: $(date)"
echo "================================================"




