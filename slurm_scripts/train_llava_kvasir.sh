#!/bin/bash
#SBATCH --job-name=llava_kvasir_cot
#SBATCH --output=slurm/logs/llava_kvasir_cot_%j.out
#SBATCH --error=slurm/logs/llava_kvasir_cot_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

echo "================================================"
echo "Training: LLaVA-Med on Kvasir with Multi-Head CoT"
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
    --model_name llava_med \
    --dataset kvasir \
    --checkpoint_path corrected_1-5_experiments/qlora_experiments/models/llava_med_kvasir_instruction/best_model \
    --output_dir results/llava_kvasir_cot \
    --device cuda

echo "================================================"
echo "End: $(date)"
echo "================================================"




