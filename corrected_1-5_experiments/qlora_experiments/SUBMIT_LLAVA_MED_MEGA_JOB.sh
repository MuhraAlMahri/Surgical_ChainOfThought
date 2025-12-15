#!/bin/bash
# Quick submission script for LLaVA-Med v1.5 mega job
# Submits zero-shot + instruction fine-tuning in parallel (4 GPUs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting LLaVA-Med v1.5 Mega Job"
echo "=========================================="
echo ""

# Check if data is prepared
KVASIR_DATA="${SCRIPT_DIR}/../datasets/kvasir_ULTRA_CONDENSED"
ENDOVIS_DATA="${SCRIPT_DIR}/../datasets/endovis2018_vqa"

if [ ! -f "${KVASIR_DATA}/train_CATEGORY_BASED.jsonl" ]; then
    echo "⚠️  WARNING: Kvasir data not found!"
    echo "   Expected: ${KVASIR_DATA}/train_CATEGORY_BASED.jsonl"
    echo ""
fi

if [ ! -f "${ENDOVIS_DATA}/train.jsonl" ]; then
    echo "⚠️  WARNING: EndoVis2018 data not found!"
    echo "   Expected: ${ENDOVIS_DATA}/train.jsonl"
    echo ""
fi

# Submit mega job
echo "Submitting mega job (zero-shot + instruction fine-tuning)..."
JOB_ID=$(sbatch slurm/mega_job_llava_med_zeroshot_and_training.slurm | grep -oP '\d+')

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
echo "  tail -f slurm/logs/mega_job_llava_med_${JOB_ID}.out"
echo ""
echo "This job runs (2 GPUs):"
echo "  - GPU 0: Kvasir-VQA (Zero-shot → Instruction fine-tuning)"
echo "  - GPU 1: EndoVis2018 (Zero-shot → Instruction fine-tuning)"
echo ""
echo "Model: microsoft/llava-med-v1.5-mistral-7b"
echo "=========================================="

