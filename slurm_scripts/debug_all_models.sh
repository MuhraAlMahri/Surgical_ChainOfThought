#!/bin/bash
#SBATCH --job-name=debug_all_models
#SBATCH --output=slurm/logs/debug_all_models_%j.out
#SBATCH --error=slurm/logs/debug_all_models_%j.err
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00

echo "================================================"
echo "TESTING ALL MODEL LOADERS"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================"

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export HF_TOKEN=${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}

# Set HuggingFace cache to workspace to avoid home quota issues
export HF_HOME=$SLURM_SUBMIT_DIR/.hf_cache
export TRANSFORMERS_CACHE=$SLURM_SUBMIT_DIR/.hf_cache/transformers
export HF_HUB_CACHE=$SLURM_SUBMIT_DIR/.hf_cache/hub
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_HUB_CACHE
echo "Using HuggingFace cache: $HF_HOME"

# Test 1: Qwen3-VL
echo ""
echo "TEST 1: Qwen3-VL + Kvasir (dry run)"
echo "------------------------------------------------"
python train_multihead_cot_v3.py \
    --model_name qwen3vl \
    --dataset kvasir \
    --checkpoint_path corrected_1-5_experiments/qlora_experiments/models/qwen3vl_kvasir_instruction/best_model \
    --output_dir results/test \
    --dry_run

if [ $? -eq 0 ]; then
    echo "✓ Qwen3-VL PASSED"
    qwen3vl_status="PASS"
else
    echo "✗ Qwen3-VL FAILED"
    qwen3vl_status="FAIL"
fi

# Test 2: MedGemma (with corrected path)
echo ""
echo "TEST 2: MedGemma + Kvasir (dry run)"
echo "------------------------------------------------"
python train_multihead_cot_v3.py \
    --model_name medgemma \
    --dataset kvasir \
    --checkpoint_path corrected_1-5_experiments/qlora_experiments/models/medgemma_kvasir_instruction/best_model \
    --output_dir results/test \
    --dry_run

if [ $? -eq 0 ]; then
    echo "✓ MedGemma PASSED"
    medgemma_status="PASS"
else
    echo "✗ MedGemma FAILED"
    medgemma_status="FAIL"
fi

# Test 3: LLaVA-Med
echo ""
echo "TEST 3: LLaVA-Med + Kvasir (dry run)"
echo "------------------------------------------------"
python train_multihead_cot_v3.py \
    --model_name llava_med \
    --dataset kvasir \
    --checkpoint_path corrected_1-5_experiments/qlora_experiments/models/llava_med_kvasir_instruction/best_model \
    --output_dir results/test \
    --dry_run

if [ $? -eq 0 ]; then
    echo "✓ LLaVA-Med PASSED"
    llava_status="PASS"
else
    echo "✗ LLaVA-Med FAILED"
    llava_status="FAIL"
fi

echo ""
echo "================================================"
echo "SUMMARY"
echo "================================================"
echo "Qwen3-VL:  $qwen3vl_status"
echo "MedGemma:  $medgemma_status"
echo "LLaVA-Med: $llava_status"
echo ""
echo "End time: $(date)"
echo "================================================"
echo "Submit training jobs for passed models only"
echo "================================================"



