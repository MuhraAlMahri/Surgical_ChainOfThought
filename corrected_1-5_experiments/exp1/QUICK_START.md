# EXP1 - Quick Start Guide

## ✅ Implementation Status

**VERIFIED:** All core components working correctly  
**BLOCKED:** Sanity overfit test due to GPU OOM (zombie process 2068499)

## 🚀 To Run (Once GPU is Available)

### Option 1: Run Everything
```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm
./RUN_ALL.sh
```

### Option 2: Step-by-Step
```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm

# 1. Sanity check (3 minutes)
sbatch 01_sanity_overfit.slurm

# 2. Full training (4 hours)
sbatch 02_train_exp1_clean.slurm

# 3. Generate predictions (10 minutes)
sbatch 03_predict_exp1_clean.slurm

# 4. Evaluate (1 minute)
sbatch 04_evaluate_exp1.slurm
```

## 📊 Check Status
```bash
# Monitor jobs
squeue -u $USER

# Check latest logs
ls -lth slurm/logs/ | head -10

# View specific log
cat slurm/logs/train_JOBID.out
```

## 🎯 Expected Results

**Current Baseline:** 19.56%  
**Expected with New Implementation:** 26-30%

**Per-Type Improvements:**
- yes_no: +22 points (constrained decoding)
- color: +17-22 points (constrained decoding)
- size_numeric: +5-10 points (numeric tolerance)
- Overall: +7-11 points

## 🔧 If GPU OOM Persists

```bash
# Check if zombie process is yours
ps aux | grep 2068499

# If yours, kill it
kill 2068499

# Or use different partition
# Edit SLURM scripts: change --partition=cscc-gpu-p
```

## 📖 Full Documentation

See `IMPLEMENTATION_COMPLETE.md` for:
- Technical details
- Diagnostic test results
- File structure
- Troubleshooting guide
- Publication-ready summary

















