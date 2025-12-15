#!/bin/bash
#SBATCH --job-name=test_grad_fix
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_grad_fix_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_grad_fix_%j.err"

echo "=========================================="
echo "TESTING GRADIENT CHECKPOINTING FIX"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "Fixes applied:"
echo "  ✓ Gradient checkpointing DISABLED"
echo "  ✓ Truncation DISABLED (answers preserved)"
echo "  ✓ A100 GPU requested"
echo ""
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
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/test_gradient_fix_output"
TRAIN_FILE="${DATASET_DIR}/train.json"
VAL_FILE="${DATASET_DIR}/val.json"

echo "Train: $TRAIN_FILE"
echo "Val: $VAL_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

nvidia-smi
echo ""

echo "Starting 20-step test..."
echo ""

# Run with FIXED training script
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
echo "CHECKING RESULTS"
echo "=========================================="

if [ -f "$OUTPUT_DIR/checkpoint-20/trainer_state.json" ]; then
    echo "✓ Checkpoint created"
    echo ""
    python3 << 'PYEOF'
import json
import sys

try:
    with open("/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/test_gradient_fix_output/checkpoint-20/trainer_state.json") as f:
        state = json.load(f)
    
    if 'log_history' in state:
        loss_entries = [log for log in state['log_history'] if 'loss' in log]
        
        if not loss_entries:
            print("❌ No loss entries found")
            sys.exit(1)
        
        print("Loss progression:")
        print("-" * 50)
        for entry in loss_entries[:10]:  # Show up to 10 entries
            step = entry.get('step', '?')
            loss = entry.get('loss', 0.0)
            grad_norm = entry.get('grad_norm', 0.0)
            print(f"  Step {step:3d}: loss={loss:.6f}, grad_norm={grad_norm:.6f}")
        
        first_loss = loss_entries[0].get('loss', 0.0)
        last_loss = loss_entries[-1].get('loss', 0.0)
        
        print("\n" + "=" * 50)
        if first_loss == 0.0 and last_loss == 0.0:
            print("❌ FAILED: Loss is still 0.0!")
            print("   The fix did not work.")
            sys.exit(1)
        elif first_loss > 0.0:
            print("✅ SUCCESS: Loss > 0.0!")
            print(f"   First loss: {first_loss:.6f}")
            print(f"   Last loss: {last_loss:.6f}")
            if last_loss < first_loss:
                improvement = ((first_loss - last_loss) / first_loss * 100)
                print(f"   Improvement: {improvement:.2f}%")
                print("\n🎉 TRAINING FIX CONFIRMED!")
                print("   → Ready to retrain all models")
            else:
                print(f"   ⚠️  Loss not decreasing much")
        else:
            print("❓ Unclear result")
    else:
        print("❌ No log_history found")
        sys.exit(1)
        
except FileNotFoundError:
    print("❌ trainer_state.json not found")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
PYEOF
else
    echo "❌ Checkpoint not created"
fi

echo ""
echo "Test completed at: $(date)"
echo "=========================================="
