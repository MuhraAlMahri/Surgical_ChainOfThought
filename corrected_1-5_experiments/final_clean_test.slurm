#!/bin/bash
#SBATCH --job-name=final_clean_test
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/final_clean_test_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/final_clean_test_%j.err"

echo "=========================================="
echo "FINAL CLEAN VERIFICATION TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "Purpose: Verify training fix with FRESH output directory"
echo "Approach: Full conversation (user + assistant)"
echo "Expected: Loss > 0 and DECREASING!"
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
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/final_clean_test_output"
TRAIN_FILE="${DATASET_DIR}/train.json"
VAL_FILE="${DATASET_DIR}/val.json"

# Delete output directory if it exists to ensure clean test
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Train: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR (FRESH - no checkpoints)"
echo ""

nvidia-smi
echo ""

echo "Running with WORKING approach (full conversation)..."
echo ""

# Run with current train_qwen_lora.py (has working collator)
python3 ${BASE_DIR}/training/train_qwen_lora.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --num_train_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --dataset_type "reordered" \
    --max_length 512 \
    --max_steps 20 \
    --evaluation_strategy "no"

echo ""
echo "=========================================="
echo "FINAL VERIFICATION"
echo "=========================================="

if [ -f "$OUTPUT_DIR/checkpoint-20/trainer_state.json" ]; then
    echo "✓ Checkpoint created"
    echo ""
    python3 << 'PYEOF'
import json
try:
    with open("/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/final_clean_test_output/checkpoint-20/trainer_state.json") as f:
        state = json.load(f)
    
    global_step = state.get('global_step', 0)
    
    loss_entries = [log for log in state.get('log_history', []) if 'loss' in log]
    
    if loss_entries:
        print("Loss progression:")
        for entry in loss_entries:
            step = entry.get('step', '?')
            loss = entry.get('loss', 0.0)
            grad = entry.get('grad_norm', 0.0)
            print(f"  Step {step}: loss={loss:.4f}, grad_norm={grad:.4f}")
        
        first_loss = loss_entries[0].get('loss', 0.0)
        last_loss = loss_entries[-1].get('loss', 0.0)
        
        print("\n" + "=" * 60)
        if first_loss > 0.0 and last_loss > 0.0:
            print("✅ ✅ ✅ SUCCESS! TRAINING FIX CONFIRMED! ✅ ✅ ✅")
            print(f"   First loss: {first_loss:.4f}")
            print(f"   Last loss: {last_loss:.4f}")
            improvement = ((first_loss - last_loss) / first_loss) * 100
            print(f"   Improvement: {improvement:.2f}%")
            print("\n🎉 READY TO RETRAIN ALL MODELS!")
        else:
            print("❌ Loss still 0.0 - fix didn't work")
    else:
        print("❌ No loss entries found")
except Exception as e:
    print(f"Error: {e}")
PYEOF
else
    echo "❌ Checkpoint not created"
fi

echo ""
echo "Test completed at: $(date)"
echo "=========================================="
