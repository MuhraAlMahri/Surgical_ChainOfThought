#!/bin/bash
# Quick submission script for Kvasir-VQA standalone job
# Submits zero-shot + instruction fine-tuning for LLaVA-Med v1.5 on Kvasir-VQA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting Kvasir-VQA Standalone Job"
echo "=========================================="
echo ""

# Submit standalone job
echo "Submitting Kvasir-VQA job (zero-shot + instruction fine-tuning)..."
JOB_ID=$(sbatch slurm/kvasir_llava_med_standalone.slurm | grep -oP '\d+')

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
echo "  tail -f slurm/logs/kvasir_llava_med_${JOB_ID}.out"
echo ""
echo "This job runs:"
echo "  - Zero-shot evaluation on Kvasir-VQA"
echo "  - Instruction fine-tuning on Kvasir-VQA"
echo ""
echo "Model: microsoft/llava-med-v1.5-mistral-7b"
echo "=========================================="






