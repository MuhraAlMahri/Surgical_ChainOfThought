#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/logs/test_model_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/logs/test_model_%j.err"

echo "=========================================="
echo "Testing Instructed Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

echo "GPU Info:"
nvidia-smi
echo ""

python3 << 'EOF'
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import torch
import json
import os

print("="*60)
print("LOADING MODEL...")
print("="*60)

# Load base model and adapter
base_model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model_path = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/models/exp1_random_baseline_instructed/checkpoint-7704"
print(f"Loading adapter from: {model_path}")

model = PeftModel.from_pretrained(base_model, model_path)
model = model.merge_and_unload()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

print("✅ Model loaded successfully")
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

print("\n" + "="*60)
print("TESTING WITH SAMPLE FROM TEST SET")
print("="*60)

# Load a test sample
test_file = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_instructed/test_instructed.json"
with open(test_file) as f:
    test_data = json.load(f)

# Test with first 3 samples
for i in range(min(3, len(test_data))):
    sample = test_data[i]
    print(f"\n--- Test Sample {i+1} ---")
    print(f"Question Type: {sample.get('question_type', 'N/A')}")
    print(f"Question: {sample['question'][:80]}...")
    print(f"Ground Truth: {sample['answer']}")
    
    # Check if image exists
    image_path = sample.get('image_path') or sample.get('image_filename')
    if image_path and not image_path.startswith('/'):
        # Try to construct full path
        image_base = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_raw_6500_image_level_70_15_15"
        full_path = os.path.join(image_base, sample.get('image_filename', ''))
        print(f"Image path: {full_path}")
        print(f"Image exists: {os.path.exists(full_path) if full_path else 'N/A'}")
    else:
        print(f"Image path: {image_path}")
        print(f"Image exists: {os.path.exists(image_path) if image_path else False}")
    
    print(f"Instruction (first 100 chars): {sample.get('instruction', 'N/A')[:100]}...")

print("\n" + "="*60)
print("MODEL TESTING COMPLETE")
print("="*60)

EOF

echo ""
echo "Testing completed at: $(date)"




