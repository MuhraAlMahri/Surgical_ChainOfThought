#!/bin/bash
#SBATCH --job-name=debug_train
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/debug_train_%j.out
#SBATCH --error=slurm/logs/debug_train_%j.err

echo "================================================"
echo "DEBUG TRAINING - Test Model Loading"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================"

# Load modules
module load nvidia/cuda/12.0
source ~/miniconda3/bin/activate base

# Environment
export PYTHONUNBUFFERED=1
export HF_TOKEN="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"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Base directory
BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

# Create logs directory
mkdir -p slurm/logs

# Test 1: Qwen3-VL dry run (just load model)
echo ""
echo "TEST 1: Qwen3-VL + Kvasir (dry run)"
echo "------------------------------------------------"
python3 train_multihead_cot_v2.py \
    --model_type qwen3vl \
    --dataset kvasir \
    --base_checkpoint Qwen/Qwen3-VL-8B-Instruct \
    --dry_run

TEST1_EXIT=$?
if [ $TEST1_EXIT -eq 0 ]; then
    echo "✓ Qwen3-VL loads successfully"
else
    echo "✗ Qwen3-VL failed to load (exit code: $TEST1_EXIT)"
fi

# Test 2: LLaVA-Med dry run
echo ""
echo "TEST 2: LLaVA-Med + Kvasir (dry run)"
echo "------------------------------------------------"
python3 train_multihead_cot_v2.py \
    --model_type llava_med \
    --dataset kvasir \
    --base_checkpoint corrected_1-5_experiments/qlora_experiments/models/llava_med_kvasir_instruction/best_model \
    --dry_run

TEST2_EXIT=$?
if [ $TEST2_EXIT -eq 0 ]; then
    echo "✓ LLaVA-Med loads successfully"
else
    echo "✗ LLaVA-Med failed to load (exit code: $TEST2_EXIT)"
fi

# Test 3: Quick training test (if loading works)
if [ $TEST1_EXIT -eq 0 ]; then
    echo ""
    echo "TEST 3: Quick training test - Qwen3-VL (10 batches)"
    echo "------------------------------------------------"
    python3 train_multihead_cot_v2.py \
        --model_type qwen3vl \
        --dataset kvasir \
        --base_checkpoint Qwen/Qwen3-VL-8B-Instruct
    
    TEST3_EXIT=$?
    if [ $TEST3_EXIT -eq 0 ]; then
        echo "✓ Training loop works"
    else
        echo "✗ Training loop failed (exit code: $TEST3_EXIT)"
    fi
else
    echo ""
    echo "TEST 3: SKIPPED (Qwen3-VL loading failed)"
fi

echo ""
echo "================================================"
echo "Debug tests complete"
echo "End time: $(date)"
echo "================================================"

# Summary
echo ""
echo "SUMMARY:"
echo "--------"
if [ $TEST1_EXIT -eq 0 ]; then
    echo "✓ Qwen3-VL: PASS"
else
    echo "✗ Qwen3-VL: FAIL"
fi

if [ $TEST2_EXIT -eq 0 ]; then
    echo "✓ LLaVA-Med: PASS"
else
    echo "✗ LLaVA-Med: FAIL"
fi

if [ $TEST3_EXIT -eq 0 ]; then
    echo "✓ Training Loop: PASS"
elif [ -z "$TEST3_EXIT" ]; then
    echo "- Training Loop: SKIPPED"
else
    echo "✗ Training Loop: FAIL"
fi

exit 0

