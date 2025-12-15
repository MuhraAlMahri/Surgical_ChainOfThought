#!/bin/bash
#SBATCH --job-name=test_proven
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_proven_%j.out"
#SBATCH --error="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/logs/test_proven_%j.err"

echo "=========================================="
echo "TESTING PROVEN APPROACH"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Approach: Full conversation (user + assistant)"
echo "Labels: Clone of input_ids"
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
OUTPUT_DIR="${BASE_DIR}/corrected 1-5 experiments/test_proven_output"
TRAIN_FILE="${DATASET_DIR}/train.json"
VAL_FILE="${DATASET_DIR}/val.json"

echo "Train: $TRAIN_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

nvidia-smi
echo ""

echo "Running with PROVEN approach (full conversation in chat template)..."
echo ""

# Use updated train_qwen_lora.py with correct collator
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
echo "RESULTS ANALYSIS"
echo "=========================================="

if [ -f "$OUTPUT_DIR/checkpoint-20/trainer_state.json" ]; then
    python3 << 'PYEOF'
import json
try:
    with open("/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/test_proven_output/checkpoint-20/trainer_state.json") as f:
        state = json.load(f)
    
    print(f"num_input_tokens_seen: {state.get('num_input_tokens_seen', 0)}")
    print(f"Global step: {state.get('global_step')}")
    
    loss_entries = [log for log in state.get('log_history', []) if 'loss' in log]
    if loss_entries:
        print("\nLoss progression:")
        for entry in loss_entries:
            print(f"  Step {entry['step']}: loss={entry['loss']:.6f}")
        
        first_loss = loss_entries[0]['loss']
        last_loss = loss_entries[-1]['loss']
        
        if first_loss > 0:
            print(f"\n✅ SUCCESS! Loss > 0 and working!")
            print(f"   First: {first_loss:.6f}, Last: {last_loss:.6f}")
        else:
            print(f"\n❌ Loss still 0.0")
except Exception as e:
    print(f"Error: {e}")
PYEOF
fi

echo ""
echo "Test completed at: $(date)"
echo "=========================================="
