#!/bin/bash
# Submit remaining CoT training jobs (medgemma and llava_med)
# Run this after some jobs complete to avoid QOS limits

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "$BASE_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     SUBMITTING REMAINING COT TRAINING JOBS               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check current job count
CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
echo "Current jobs in queue: $CURRENT_JOBS"
echo ""

if [ "$CURRENT_JOBS" -gt 10 ]; then
    echo "⚠️  WARNING: You still have $CURRENT_JOBS jobs in queue."
    echo "   Consider waiting for some to complete before submitting more."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

echo "Submitting remaining jobs..."
echo ""

# MedGemma jobs
echo "1. Submitting medgemma on kvasir..."
JOB1=$(sbatch slurm/07_train_multihead_cot.slurm medgemma kvasir google/medgemma-4b 5e-5 5 1 16 | awk '{print $4}')
if [ -n "$JOB1" ]; then
    echo "   ✅ Job ID: $JOB1"
else
    echo "   ❌ Failed to submit"
fi
sleep 2

echo "2. Submitting medgemma on endovis..."
JOB2=$(sbatch slurm/07_train_multihead_cot.slurm medgemma endovis google/medgemma-4b 5e-5 5 1 16 | awk '{print $4}')
if [ -n "$JOB2" ]; then
    echo "   ✅ Job ID: $JOB2"
else
    echo "   ❌ Failed to submit"
fi
sleep 2

# LLaVA-Med jobs
echo "3. Submitting llava_med on kvasir..."
JOB3=$(sbatch slurm/07_train_multihead_cot.slurm llava_med kvasir corrected_1-5_experiments/qlora_experiments/models/llava_med_kvasir_instruction/best_model 5e-5 5 1 16 | awk '{print $4}')
if [ -n "$JOB3" ]; then
    echo "   ✅ Job ID: $JOB3"
else
    echo "   ❌ Failed to submit"
fi
sleep 2

echo "4. Submitting llava_med on endovis..."
JOB4=$(sbatch slurm/07_train_multihead_cot.slurm llava_med endovis corrected_1-5_experiments/qlora_experiments/models/llava_med_endovis_instruction/best_model 5e-5 5 1 16 | awk '{print $4}')
if [ -n "$JOB4" ]; then
    echo "   ✅ Job ID: $JOB4"
else
    echo "   ❌ Failed to submit"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Remaining jobs submitted!"
echo ""
echo "Monitor: squeue -u \$USER"
echo "╚══════════════════════════════════════════════════════════╝"





