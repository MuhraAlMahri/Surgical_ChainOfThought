#!/bin/bash
# Quick submission script for EndoVis2018 mega job
# Submits zero-shot + training in parallel (3 GPUs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting EndoVis2018 Mega Job"
echo "=========================================="
echo ""

# Check if data is prepared
DATA_DIR="../datasets/endovis2018_vqa"
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "⚠️  WARNING: Data not prepared yet!"
    echo "   Please run data preparation first:"
    echo "   python scripts/prepare_endovis2018_for_vqa.py --use_proper_split"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Submit mega job
echo "Submitting mega job (zero-shot + training)..."
JOB_ID=$(sbatch slurm/mega_job_zeroshot_and_training.slurm | grep -oP '\d+')

echo ""
echo "=========================================="
echo "Job Submitted Successfully"
echo "=========================================="
echo "Job ID: ${JOB_ID}"
echo ""
echo "Monitor job:"
echo "  squeue -j ${JOB_ID}"
echo ""
echo "View logs:"
echo "  tail -f slurm/logs/mega_job_${JOB_ID}.out"
echo ""
echo "This job runs:"
echo "  - GPU 0: Zero-shot evaluation"
echo "  - GPU 1: Training Exp1 (Random Baseline)"
echo "  - GPU 2: Training Exp2 (Qwen Reordered)"
echo "  - GPU 3: Training Exp3 (Sequential)"
echo "=========================================="

