#!/bin/bash
# Quick status check for QLoRA training jobs

echo "========================================================================"
echo "QLORA TRAINING STATUS"
echo "========================================================================"
echo ""

# Check running jobs
echo ">>> CURRENT JOBS:"
squeue -u muhra.almahri -o "%.18i %.12P %.30j %.8u %.2t %.10M %.10l %.6D"
echo ""

# Check latest progress from logs
echo ">>> BATCH 1 PROGRESS:"
echo ""

# Task 0 (Exp1)
TASK0_LOG=$(ls -t /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/batch1_0_*.err 2>/dev/null | head -1)
if [ -f "$TASK0_LOG" ]; then
    echo "Task 0 (Exp1 - Random):"
    tail -5 "$TASK0_LOG" | grep "it/s" | tail -1
    echo ""
fi

# Task 1 (Exp2)
TASK1_LOG=$(ls -t /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/slurm/logs/batch1_1_*.err 2>/dev/null | head -1)
if [ -f "$TASK1_LOG" ]; then
    echo "Task 1 (Exp2 - Qwen Reordered):"
    tail -5 "$TASK1_LOG" | grep "it/s" | tail -1
    echo ""
fi

echo "========================================================================"
echo "To monitor in real-time:"
echo "  watch -n 60 'bash $(readlink -f $0)'"
echo "========================================================================"

