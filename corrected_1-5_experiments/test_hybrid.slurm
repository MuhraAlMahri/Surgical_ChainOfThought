#!/bin/bash
#SBATCH --job-name=test_hybrid
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_hybrid_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_hybrid_%j.err"

echo "=========================================="
echo "TESTING HYBRID SCRIPT (Custom Training Loop)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "Approach: Custom loop (no HuggingFace Trainer)"
echo "Based on: train_progressive_stage_PROPER.py"
echo "Expected: Loss > 0 and decreasing"
echo "=========================================="

module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export HF_HUB_CACHE=/tmp/hf_cache_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets/kvasir_raw_6500_image_level_70_15_15"
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/test_hybrid_output"
TRAIN_FILE="${DATASET_DIR}/train.json"
IMAGE_DIR="${BASE_DIR}/datasets/Kvasir-VQA/raw/images"

echo "Train file: $TRAIN_FILE"
echo "Image dir: $IMAGE_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

nvidia-smi
echo ""

echo "Running HYBRID script with custom training loop..."
echo ""

# Run hybrid script with custom loop
python3 ${BASE_DIR}/training/train_qwen_lora_hybrid.py \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --train_file "$TRAIN_FILE" \
    --val_file "$TRAIN_FILE" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --max_steps 20 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --lora_r 32 \
    --lora_alpha 64 \
    --logging_steps 5

echo ""
echo "=========================================="
echo "VERIFICATION"
echo "=========================================="

if [ -f "$OUTPUT_DIR/checkpoint-20/trainer_state.json" ]; then
    echo "✓ Checkpoint created"
    echo ""
    python3 << 'PYEOF'
import json
try:
    with open("/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/test_hybrid_output/checkpoint-20/trainer_state.json") as f:
        state = json.load(f)
    
    step = state.get('step', 'N/A')
    loss = state.get('loss', 'N/A')
    
    print(f"Final step: {step}")
    print(f"Final loss: {loss}")
    
    if isinstance(loss, (int, float)):
        if loss > 0.0:
            print(f"\n✅ SUCCESS: Loss = {loss:.6f} (> 0.0!)")
            print("🎉 TRAINING WORKS WITH CUSTOM LOOP!")
        else:
            print(f"\n❌ FAILED: Loss still 0.0")
    else:
        print("\n⚠️  Could not determine loss")
except Exception as e:
    print(f"Error: {e}")
PYEOF
fi

echo ""
echo "Test completed at: $(date)"
echo "=========================================="
