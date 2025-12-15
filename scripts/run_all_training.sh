#!/bin/bash
# run_all_training.sh - Submit all 6 model-dataset combinations

echo "================================================"
echo "SUBMITTING ALL TRAINING JOBS"
echo "================================================"

mkdir -p slurm/logs results

export HF_TOKEN=${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}

# Qwen3-VL
job1=$(sbatch slurm_scripts/train_qwen3vl_kvasir.sh | awk '{print $4}')
echo "✓ Submitted: Qwen3-VL + Kvasir (Job $job1)"

job2=$(sbatch slurm_scripts/train_qwen3vl_endovis.sh | awk '{print $4}')
echo "✓ Submitted: Qwen3-VL + EndoVis (Job $job2)"

# MedGemma (now with correct path!)
job3=$(sbatch slurm_scripts/train_medgemma_kvasir.sh | awk '{print $4}')
echo "✓ Submitted: MedGemma + Kvasir (Job $job3)"

job4=$(sbatch slurm_scripts/train_medgemma_endovis.sh | awk '{print $4}')
echo "✓ Submitted: MedGemma + EndoVis (Job $job4)"

# LLaVA-Med
job5=$(sbatch slurm_scripts/train_llava_kvasir.sh | awk '{print $4}')
echo "✓ Submitted: LLaVA-Med + Kvasir (Job $job5)"

job6=$(sbatch slurm_scripts/train_llava_endovis.sh | awk '{print $4}')
echo "✓ Submitted: LLaVA-Med + EndoVis (Job $job6)"

echo ""
echo "================================================"
echo "ALL 6 JOBS SUBMITTED"
echo "================================================"
echo "Job IDs: $job1, $job2, $job3, $job4, $job5, $job6"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs/*.out"
echo ""
echo "Expected completion: 3-5 hours"
echo "================================================"




