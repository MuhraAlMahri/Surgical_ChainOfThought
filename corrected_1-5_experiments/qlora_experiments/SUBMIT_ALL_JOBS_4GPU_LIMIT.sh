#!/bin/bash
# Master script to submit all QLoRA training jobs
# Uses single jobs with multiple GPUs (4 GPUs for Batch 1, 3 GPUs for Batch 2)
# Usage: bash SUBMIT_ALL_JOBS_4GPU_LIMIT.sh

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"

echo "========================================================================"
echo "SUBMITTING ALL QLORA TRAINING JOBS (Multi-GPU Strategy)"
echo "========================================================================"
echo ""
echo "Strategy:"
echo "  Batch 1 (1 job, 4 GPUs): Exp1, Exp2, Exp3-S1, Exp4-S1 (parallel)"
echo "  Batch 2 (1 job, 3 GPUs): Exp3-S2, Exp4-S2, Exp3-S3 (parallel)"
echo "  Batch 3 (1 job, 1 GPU): Exp4-S3"
echo ""

cd ${BASE_DIR}

# ============================================================================
# BATCH 1: Submit priority experiments (1 job with 4 GPUs)
# ============================================================================
echo ">>> Submitting Batch 1: Priority Experiments (1 job, 4 GPUs parallel)"
echo "    GPU 0: Exp1 - Random Baseline (~20h)"
echo "    GPU 1: Exp2 - Qwen Reordered (~20h)"
echo "    GPU 2: Exp3-S1 - CXRTrek Stage 1 (~7h)"
echo "    GPU 3: Exp4-S1 - Curriculum Stage 1 (~7h)"
echo ""

BATCH1_JOB=$(sbatch --parsable slurm/batch1_priority_experiments.slurm)
echo "✓ Submitted Batch 1: ${BATCH1_JOB}"
echo "  (1 job using 4 GPUs, 4 experiments running in parallel)"
echo ""

# ============================================================================
# BATCH 2: Submit remaining experiments (depends on Batch 1)
# ============================================================================
echo ">>> Submitting Batch 2: Remaining Experiments (1 job, 3 GPUs parallel)"
echo "    GPU 0: Exp3-S2 - CXRTrek Stage 2 (~12h)"
echo "    GPU 1: Exp4-S2 - Curriculum Stage 2 (~12h, needs Exp4-S1)"
echo "    GPU 2: Exp3-S3 - CXRTrek Stage 3 (~30min)"
echo ""

BATCH2_JOB=$(sbatch --parsable --dependency=afterok:${BATCH1_JOB} slurm/batch2_remaining_experiments.slurm)
echo "✓ Submitted Batch 2: ${BATCH2_JOB}"
echo "  (will start after Batch 1 completes: ${BATCH1_JOB})"
echo ""

# ============================================================================
# BATCH 3: Submit final stage (depends on Batch 2)
# ============================================================================
echo ">>> Submitting Batch 3: Final Stage (1 job, 1 GPU)"
echo "    Exp4-S3 - Curriculum Stage 3 (~30min, needs Exp4-S2)"
echo ""

BATCH3_JOB=$(sbatch --parsable --dependency=afterok:${BATCH2_JOB} slurm/batch3_final_stage.slurm)
echo "✓ Submitted Batch 3: ${BATCH3_JOB}"
echo "  (will start after Batch 2 completes: ${BATCH2_JOB})"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================================================"
echo "✓ ALL BATCHES SUBMITTED!"
echo "========================================================================"
echo ""
echo "Job Summary:"
echo "------------"
echo "  Batch 1: ${BATCH1_JOB} (1 job, 4 GPUs: Exp1, Exp2, Exp3-S1, Exp4-S1)"
echo "  Batch 2: ${BATCH2_JOB} (1 job, 3 GPUs: Exp3-S2, Exp4-S2, Exp3-S3)"
echo "  Batch 3: ${BATCH3_JOB} (1 job, 1 GPU: Exp4-S3)"
echo ""
echo "Total Experiments: 8"
echo "  - 4 running immediately in parallel (Batch 1, 4 GPUs)"
echo "  - 3 waiting for Batch 1, then running in parallel (Batch 2, 3 GPUs)"
echo "  - 1 waiting for Batch 2 (Batch 3, 1 GPU)"
echo ""
echo "Expected Timeline:"
echo "  - Batch 1 completion: ~20 hours (slowest: Exp1/Exp2)"
echo "  - Batch 2 completion: +12 hours (slowest: Exp3-S2, Exp4-S2)"
echo "  - Batch 3 completion: +30 min (Exp4-S3)"
echo "  - Total wall-clock time: ~32-33 hours"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs/batch*.out"
echo "  tail -f slurm/logs/batch*_gpu*.out  # Individual GPU logs"
echo "========================================================================"

