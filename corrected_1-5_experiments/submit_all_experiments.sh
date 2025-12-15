#!/bin/bash
# Submit all corrected experiments 1-4 for the new dataset split
# 
# Experiments:
# 1. Random Baseline (single job)
# 2. Qwen Ordering (single job)
# 3. CXRTrek Sequential (3 independent jobs - can run in parallel)
# 4. Curriculum Learning (3 sequential jobs with dependencies)

echo "=========================================="
echo "SUBMITTING CORRECTED EXPERIMENTS 1-4"
echo "New Dataset Split (Image-Level, 70/15/15)"
echo "=========================================="
echo ""

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments"

cd "$BASE_DIR"

# Experiment 1: Random Baseline
echo "Submitting Experiment 1: Random Baseline..."
JOB1=$(sbatch experiments/exp1_random/train_random_baseline.slurm | grep -o '[0-9]*')
echo "  Job ID: $JOB1"
echo ""

# Experiment 2: Qwen Reordering
echo "Submitting Experiment 2: Qwen Reordering..."
JOB2=$(sbatch experiments/exp2_qwen_reordered/train_qwen_reordered.slurm | grep -o '[0-9]*')
echo "  Job ID: $JOB2"
echo ""

# Experiment 3: CXRTrek Sequential (all 3 stages can run in parallel)
echo "Submitting Experiment 3: CXRTrek Sequential..."
echo "  Stage 1 (Initial Assessment)..."
JOB3_S1=$(sbatch experiments/exp3_cxrtrek_sequential/train_stage1.slurm | grep -o '[0-9]*')
echo "    Job ID: $JOB3_S1"

echo "  Stage 2 (Findings Identification)..."
JOB3_S2=$(sbatch experiments/exp3_cxrtrek_sequential/train_stage2.slurm | grep -o '[0-9]*')
echo "    Job ID: $JOB3_S2"

echo "  Stage 3 (Clinical Context)..."
JOB3_S3=$(sbatch experiments/exp3_cxrtrek_sequential/train_stage3.slurm | grep -o '[0-9]*')
echo "    Job ID: $JOB3_S3"
echo ""

# Experiment 4: Curriculum Learning (sequential with dependencies)
echo "Submitting Experiment 4: Curriculum Learning..."
echo "  Stage 1 (Initial Assessment)..."
JOB4_S1=$(sbatch experiments/exp4_curriculum_learning/train_stage1.slurm | grep -o '[0-9]*')
echo "    Job ID: $JOB4_S1"

echo "  Stage 2 (depends on Stage 1)..."
JOB4_S2=$(sbatch --dependency=afterok:$JOB4_S1 experiments/exp4_curriculum_learning/train_stage2.slurm | grep -o '[0-9]*')
echo "    Job ID: $JOB4_S2 (depends on $JOB4_S1)"

echo "  Stage 3 (depends on Stage 2)..."
JOB4_S3=$(sbatch --dependency=afterok:$JOB4_S2 experiments/exp4_curriculum_learning/train_stage3.slurm | grep -o '[0-9]*')
echo "    Job ID: $JOB4_S3 (depends on $JOB4_S2)"
echo ""

echo "=========================================="
echo "SUBMISSION SUMMARY"
echo "=========================================="
echo "Experiment 1 (Random Baseline):     $JOB1"
echo "Experiment 2 (Qwen Reordering):      $JOB2"
echo "Experiment 3 Stage 1:                $JOB3_S1"
echo "Experiment 3 Stage 2:                $JOB3_S2"
echo "Experiment 3 Stage 3:                $JOB3_S3"
echo "Experiment 4 Stage 1:                $JOB4_S1"
echo "Experiment 4 Stage 2:                $JOB4_S2 (depends on $JOB4_S1)"
echo "Experiment 4 Stage 3:                 $JOB4_S3 (depends on $JOB4_S2)"
echo ""
echo "Monitor jobs with: squeue -u muhra.almahri"
echo "View logs in: corrected 1-5 experiments/logs/"
echo "=========================================="

