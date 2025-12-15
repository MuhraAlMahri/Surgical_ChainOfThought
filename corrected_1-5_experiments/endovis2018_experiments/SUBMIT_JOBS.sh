#!/bin/bash
# Submission script for EndoVis2018 experiments
# IMPORTANT: All jobs run on compute nodes via SLURM (MBZUAI HPC policy)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "EndoVis2018 Experiments - Job Submission"
echo "=========================================="
echo ""

# Check if data is prepared
DATA_DIR="../datasets/endovis2018_vqa"
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "⚠️  WARNING: Data not prepared yet!"
    echo "   Please run: python scripts/prepare_endovis2018_for_vqa.py"
    echo "   Then run this script again."
    exit 1
fi

echo "✅ Data files found"
echo ""

# Step 1: Zero-Shot Evaluation
echo "Step 1: Submitting Zero-Shot Evaluation..."
ZEROSHOT_JOB=$(sbatch slurm/zeroshot_endovis2018.slurm | grep -oP '\d+')
echo "  ✓ Zero-shot job submitted: ${ZEROSHOT_JOB}"
echo "  Monitor with: squeue -j ${ZEROSHOT_JOB}"
echo ""

# Step 2: Training (can be submitted after zero-shot or in parallel)
echo "Step 2: Submitting Training (Exp1)..."
TRAIN_JOB=$(sbatch slurm/train_exp1.slurm | grep -oP '\d+')
echo "  ✓ Training job submitted: ${TRAIN_JOB}"
echo "  Monitor with: squeue -j ${TRAIN_JOB}"
echo ""

echo "=========================================="
echo "All Jobs Submitted"
echo "=========================================="
echo "Zero-Shot Job ID: ${ZEROSHOT_JOB}"
echo "Training Job ID: ${TRAIN_JOB}"
echo ""
echo "Monitor all jobs: squeue -u \$USER"
echo "Check logs: tail -f slurm/logs/*.out"
echo "=========================================="




















