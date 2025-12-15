#!/bin/bash
#SBATCH --job-name=test_final_fix
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_final_fix_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_final_fix_%j.err"

echo "=========================================="
echo "FINAL TEST: Gradient Checkpointing Fix"
echo "Testing: Loss > 0 and decreasing"
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
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p $HF_HOME

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
DATASET_DIR="${BASE_DIR}/corrected 1-5 experiments/datasets/kvasir_raw_6500_image_level_70_15_15"
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/test_final_fix_output"
TRAIN_FILE="${DATASET_DIR}/train.json"
VAL_FILE="${DATASET_DIR}/val.json"

echo "Using FIXED train_qwen_lora.py with:"
echo "  ✓ Truncation disabled (answers preserved)"
echo "  ✓ Gradient checkpointing disabled (gradients flow)"
echo ""
echo "Train: $TRAIN_FILE"
echo "Val: $VAL_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

nvidia-smi
echo ""

echo "Running 20-step verification test..."
echo ""

# Run training for 20 steps to verify fix
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
echo "VERIFICATION:"
echo "=========================================="

if [ -f "$OUTPUT_DIR/checkpoint-20/trainer_state.json" ]; then
    echo "✓ Checkpoint created"
    echo ""
    echo "Checking loss values..."
    python3 << 'PYEOF'
import json
try:
    with open("${OUTPUT_DIR}/checkpoint-20/trainer_state.json") as f:
        state = json.load(f)
    
    if 'log_history' in state:
        loss_entries = [log for log in state['log_history'] if 'loss' in log]
        if loss_entries:
            first_loss = loss_entries[0].get('loss', 0.0)
            last_loss = loss_entries[-1].get('loss', 0.0)
            
            print(f"First loss: {first_loss:.6f}")
            print(f"Last loss: {last_loss:.6f}")
            
            if first_loss == 0.0:
                print("\n❌ FAIL: Loss is still 0.0!")
            elif first_loss > 0.0 and last_loss < first_loss:
                print(f"\n✅ PASS: Loss > 0 and decreasing!")
                print(f"Improvement: {((first_loss - last_loss) / first_loss * 100):.2f}%")
            else:
                print(f"\n⚠️  Loss > 0 but not decreasing much")
except Exception as e:
    print(f"Error: {e}")
PYEOF
fi

echo ""
echo "Test completed at: $(date)"
echo "=========================================="
