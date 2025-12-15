#!/bin/bash
# Submit mega job for EndoVis2018 Experiments 2, 3, 4, 5

cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/endovis2018_experiments

echo "=========================================="
echo "Submitting EndoVis2018 Experiments 2,3,4,5"
echo "=========================================="

# Check if data is prepared
REORDERED_DIR="../datasets/endovis2018_vqa_reordered"
if [ ! -f "${REORDERED_DIR}/exp2_qwen_reordered/train.jsonl" ]; then
    echo "ERROR: Reordered data not found!"
    echo "Please run: python3 ../../scripts/reorder_endovis2018_with_qwen.py"
    exit 1
fi

echo "✓ Reordered data found"
echo ""

# Submit job
sbatch slurm/mega_job_exp2_3_4_5.slurm

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs/mega_job_exp2345_*.out"
echo ""
echo "This job will run:"
echo "  - Exp2 (Qwen Reordered) on GPU 0"
echo "  - Exp3 (Sequential: Stage 1 model → Stage 2 model) on GPU 1"
echo "  - Exp4 (Curriculum: Stage 1 → Stage 2) on GPU 2"
echo "  - Exp5 (Sequential CoT) on GPU 3"
echo ""
echo "Note: Exp3 uses separate models per stage (like Kvasir-VQA Exp3)"
echo "      During evaluation, questions are routed to the appropriate stage model"
echo "=========================================="

