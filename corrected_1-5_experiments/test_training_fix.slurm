#!/bin/bash
#SBATCH --job-name=test_training_fix
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_training_fix_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_training_fix_%j.err"

echo "=========================================="
echo "TEST: Verify Training Fix (Loss > 0)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
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
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/test_fix_output"
TRAIN_FILE="${DATASET_DIR}/train.json"
VAL_FILE="${DATASET_DIR}/val.json"

echo "Train: $TRAIN_FILE"
echo "Val: $VAL_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

nvidia-smi
echo ""

echo "Running 50-step test to verify loss > 0 and decreasing..."
echo ""

python3 ${BASE_DIR}/training/train_qwen_lora.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --max_steps 50 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --dataset_type "reordered" \
    --max_length 512

echo ""
echo "=========================================="
echo "VERIFICATION: Checking Training Loss"
echo "=========================================="
echo ""

# Check trainer_state.json if it exists
if [ -f "$OUTPUT_DIR/checkpoint-50/trainer_state.json" ]; then
    echo "✓ Checkpoint created successfully"
    echo ""
    echo "Loss values from trainer_state.json:"
    python3 << 'EOF'
import json
import sys

try:
    with open("$OUTPUT_DIR/checkpoint-50/trainer_state.json") as f:
        state = json.load(f)
    
    if 'log_history' in state:
        loss_entries = [log for log in state['log_history'] if 'loss' in log]
        if loss_entries:
            print(f"\n{'Step':<10} {'Loss':<15}")
            print("-" * 25)
            for entry in loss_entries[:10]:  # Show first 10
                print(f"{entry.get('step', 'N/A'):<10} {entry.get('loss', 0.0):<15.6f}")
            
            first_loss = loss_entries[0].get('loss', 0.0)
            last_loss = loss_entries[-1].get('loss', 0.0)
            
            print("\n" + "="*40)
            if first_loss == 0.0 and last_loss == 0.0:
                print("❌ FAIL: Loss is still 0.0!")
                print("   The fix didn't work!")
                sys.exit(1)
            elif last_loss < first_loss:
                print("✅ PASS: Loss is > 0 and DECREASING!")
                print(f"   First loss: {first_loss:.6f}")
                print(f"   Last loss: {last_loss:.6f}")
                print(f"   Improvement: {((first_loss - last_loss) / first_loss * 100):.2f}%")
            elif last_loss > 0:
                print("⚠️  PARTIAL: Loss > 0 but not decreasing much")
                print(f"   First loss: {first_loss:.6f}")
                print(f"   Last loss: {last_loss:.6f}")
            else:
                print("❓ UNCLEAR: Check logs manually")
        else:
            print("❌ No loss entries found")
    else:
        print("❌ No log_history in trainer_state.json")
except Exception as e:
    print(f"❌ Error reading trainer_state.json: {e}")
EOF
else
    echo "⚠️  trainer_state.json not found, checking output logs..."
fi

echo ""
echo "Test completed at: $(date)"
echo "=========================================="

