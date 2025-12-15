#!/bin/bash
# Submit Batch 2 and 3 after Batch 1 completes
# Run this script when Batch 1 (Job 156135) finishes

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"

echo "========================================================================"
echo "SUBMITTING BATCH 2 & 3 (with dependencies)"
echo "========================================================================"
echo ""

cd ${BASE_DIR}

# Check if Batch 1 completed
BATCH1_STATUS=$(sacct -j 156135 --format=State -n | head -1 | xargs)
echo "Batch 1 (Job 156135) status: ${BATCH1_STATUS}"

if [[ "${BATCH1_STATUS}" != "COMPLETED" ]]; then
    echo "WARNING: Batch 1 has not completed yet (status: ${BATCH1_STATUS})"
    echo "Submitting with dependency anyway..."
fi

echo ""
echo ">>> Submitting Batch 2 (3 parallel GPUs)"
BATCH2_JOB=$(sbatch --parsable --dependency=afterok:156135 slurm/batch2_remaining_experiments.slurm)
echo "✓ Submitted Batch 2: ${BATCH2_JOB}"
echo "  (will start after Batch 1 completes)"
echo ""

echo ">>> Submitting Batch 3 (1 GPU)"
BATCH3_JOB=$(sbatch --parsable --dependency=afterok:${BATCH2_JOB} slurm/batch3_final_stage.slurm)
echo "✓ Submitted Batch 3: ${BATCH3_JOB}"
echo "  (will start after Batch 2 completes)"
echo ""

echo "========================================================================"
echo "✓ BATCH 2 & 3 SUBMITTED!"
echo "========================================================================"
echo ""
echo "Job Chain:"
echo "  Batch 1: 156135 (running now, 4 tasks)"
echo "  Batch 2: ${BATCH2_JOB} (pending, 3 tasks)"
echo "  Batch 3: ${BATCH3_JOB} (pending, 1 task)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "========================================================================"

