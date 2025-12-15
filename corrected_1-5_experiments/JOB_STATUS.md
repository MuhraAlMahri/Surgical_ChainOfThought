# Experiment Submission Status

## Current Status

### ✅ Running
- **Experiment 2 (Qwen Reordering)**: Job ID **150881** - RUNNING (GPU-03)
- **Experiment 3 Stage 1**: Job ID **150882** - RUNNING (GPU-03)

### ❌ Failed
- **Experiment 1 (Random Baseline)**: Job ID **150880** - FAILED (OOM Error)
  - **Issue**: Model was being loaded twice causing OutOfMemoryError
  - **Fix Applied**: Removed duplicate model loading in training script
  - **Status**: Fixed and ready to resubmit

### ⏳ Pending Submission
- Experiment 1 (Random Baseline) - Ready to resubmit with fix
- Experiment 3 Stages 2 & 3 - Waiting for job slots
- Experiment 4 Stages 1, 2, 3 - Waiting for job slots

## Fix Details

**Problem**: The training script (`train_qwen_lora.py`) was loading the model twice:
1. Line 815: To compute dataset size (unnecessary)
2. Line 662: In the train() function (necessary)

**Solution**: Removed the unnecessary model load at line 815. Now it just reads the JSON file directly to get dataset size, which avoids the OOM error.

## Resubmit Experiment 1

```bash
cd "corrected 1-5 experiments"
sbatch experiments/exp1_random/train_random_baseline.slurm
```

## Monitor Jobs

```bash
# Check all jobs
squeue -u muhra.almahri

# View logs
tail -f "corrected 1-5 experiments/logs/exp1_random_<JOB_ID>.out"
tail -f "corrected 1-5 experiments/logs/exp2_qwen_150881.out"
```

## QoS Limits

- Maximum jobs per user: **4**
- Maximum GPUs per job: **4** (we use 1)
- Time limit: **24 hours** per job
