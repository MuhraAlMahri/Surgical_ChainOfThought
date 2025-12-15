#!/bin/bash
# Submit remaining experiments once job slots are available
# This script submits the remaining stages that couldn't be submitted initially

echo "=========================================="
echo "SUBMITTING REMAINING EXPERIMENTS"
echo "=========================================="
echo ""

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments"

cd "$BASE_DIR"

# Experiment 3: Remaining stages (can run in parallel)
echo "Submitting Experiment 3: Remaining stages..."
echo "  Stage 2 (Findings Identification)..."
JOB3_S2=$(sbatch experiments/exp3_cxrtrek_sequential/train_stage2.slurm | grep -o '[0-9]*')
echo "    Job ID: $JOB3_S2"

echo "  Stage 3 (Clinical Context)..."
JOB3_S3=$(sbatch experiments/exp3_cxrtrek_sequential/train_stage3.slurm | grep -o '[0-9]*')
echo "    Job ID: $JOB3_S3"
echo ""

# Experiment 4: Curriculum Learning (sequential with dependencies)
# Note: You may need to check if Stage 1 completed first
echo "Submitting Experiment 4: Curriculum Learning..."
echo "  Stage 1 (Initial Assessment)..."
JOB4_S1=$(sbatch experiments/exp4_curriculum_learning/train_stage1.slurm | grep -o '[0-9]*')
echo "    Job ID: $JOB4_S1"

echo "  Stage 2 (will depend on Stage 1)..."
echo "    Submit manually after Stage 1 completes:"
echo "    JOB4_S2=\$(sbatch --dependency=afterok:\$JOB4_S1 experiments/exp4_curriculum_learning/train_stage2.slurm | grep -o '[0-9]*')"
echo ""
echo "  Stage 3 (will depend on Stage 2)..."
echo "    Submit manually after Stage 2 completes:"
echo "    JOB4_S3=\$(sbatch --dependency=afterok:\$JOB4_S2 experiments/exp4_curriculum_learning/train_stage3.slurm | grep -o '[0-9]*')"
echo ""

echo "=========================================="
echo "SUBMISSION SUMMARY"
echo "=========================================="
echo "Experiment 3 Stage 2: $JOB3_S2"
echo "Experiment 3 Stage 3: $JOB3_S3"
echo "Experiment 4 Stage 1: $JOB4_S1"
echo ""
echo "Note: Experiment 4 Stages 2 & 3 should be submitted"
echo "      after their dependencies complete."
echo ""
echo "Check job status: squeue -u muhra.almahri"
echo "=========================================="

