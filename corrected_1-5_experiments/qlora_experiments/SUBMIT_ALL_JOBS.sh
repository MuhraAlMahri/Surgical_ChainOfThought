#!/bin/bash
# Master script to submit all QLoRA training jobs
# Usage: bash SUBMIT_ALL_JOBS.sh

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"

echo "========================================================================"
echo "SUBMITTING ALL QLORA TRAINING JOBS (Qwen3-VL-8B, 5 epochs, r=4)"
echo "========================================================================"
echo ""

# ============================================================================
# BATCH 1: Submit parallel experiments (Exp1, Exp2, Exp3-all stages)
# ============================================================================
echo ">>> Submitting Batch 1: Parallel Experiments (5 tasks)"
echo "    - Exp1 (Random Baseline)"
echo "    - Exp2 (Qwen Reordered)"
echo "    - Exp3-S1 (CXRTrek Stage 1)"
echo "    - Exp3-S2 (CXRTrek Stage 2)"
echo "    - Exp3-S3 (CXRTrek Stage 3)"
echo ""

cd ${BASE_DIR}
PARALLEL_JOB=$(sbatch --parsable slurm/train_parallel_experiments.slurm)
echo "✓ Submitted job array: ${PARALLEL_JOB}"
echo "  (5 parallel tasks will run)"
echo ""

# ============================================================================
# BATCH 2: Submit Exp4-Stage1 (independent, can run in parallel with batch 1)
# ============================================================================
echo ">>> Submitting Batch 2: Exp4-Stage1 (Curriculum Learning Start)"
echo ""

EXP4_S1_JOB=$(sbatch --parsable slurm/train_exp4_stage1.slurm)
echo "✓ Submitted Exp4-Stage1: ${EXP4_S1_JOB}"
echo ""

# ============================================================================
# BATCH 3: Submit Exp4-Stage2 (depends on Stage1)
# ============================================================================
echo ">>> Submitting Batch 3: Exp4-Stage2 (depends on Stage1)"
echo ""

EXP4_S2_JOB=$(sbatch --parsable --dependency=afterok:${EXP4_S1_JOB} slurm/train_exp4_stage2.slurm)
echo "✓ Submitted Exp4-Stage2: ${EXP4_S2_JOB}"
echo "  (will start after Stage1 completes: ${EXP4_S1_JOB})"
echo ""

# ============================================================================
# BATCH 4: Submit Exp4-Stage3 (depends on Stage2)
# ============================================================================
echo ">>> Submitting Batch 4: Exp4-Stage3 (depends on Stage2)"
echo ""

EXP4_S3_JOB=$(sbatch --parsable --dependency=afterok:${EXP4_S2_JOB} slurm/train_exp4_stage3.slurm)
echo "✓ Submitted Exp4-Stage3: ${EXP4_S3_JOB}"
echo "  (will start after Stage2 completes: ${EXP4_S2_JOB})"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================================================"
echo "✓ ALL JOBS SUBMITTED!"
echo "========================================================================"
echo ""
echo "Job Summary:"
echo "------------"
echo "  Parallel Array: ${PARALLEL_JOB} (5 tasks: Exp1, Exp2, Exp3-S1/S2/S3)"
echo "  Exp4-Stage1:    ${EXP4_S1_JOB}"
echo "  Exp4-Stage2:    ${EXP4_S2_JOB} (depends on ${EXP4_S1_JOB})"
echo "  Exp4-Stage3:    ${EXP4_S3_JOB} (depends on ${EXP4_S2_JOB})"
echo ""
echo "Total Experiments: 8"
echo "  - 6 running in parallel immediately"
echo "  - 2 waiting for dependencies (Exp4-S2, Exp4-S3)"
echo ""
echo "Expected Timeline:"
echo "  - Batch 1 (parallel): ~20 hours max (Exp1, Exp2)"
echo "  - Exp4-S1: ~7 hours (starts now)"
echo "  - Exp4-S2: ~12 hours (starts after S1)"
echo "  - Exp4-S3: ~30 min (starts after S2)"
echo "  - Total Exp4 time: ~20 hours sequential"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  tail -f slurm/logs/*.out"
echo "========================================================================"

