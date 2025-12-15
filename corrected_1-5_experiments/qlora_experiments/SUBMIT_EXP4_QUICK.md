# Quick Start: Submit Exp4 Retraining

## Option 1: All-in-One Job (Recommended)

Runs all 3 stages sequentially in a single job:

```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments
sbatch slurm/train_exp4_all_stages.slurm
```

**Advantages:**
- ✅ Single job, easier to monitor
- ✅ Automatic sequential execution
- ✅ Built-in verification checks
- ✅ Total time: ~20 hours

**Monitor:**
```bash
squeue -u $USER
tail -f slurm/logs/exp4_all_<JOB_ID>.out
```

---

## Option 2: Separate Jobs with Dependencies

Submit 3 separate jobs with dependencies:

```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments

# Stage 1
JOB1=$(sbatch slurm/train_exp4_stage1.slurm | grep -o '[0-9]*')
echo "Stage 1: Job $JOB1"

# Stage 2 (waits for Stage 1)
JOB2=$(sbatch --dependency=afterok:$JOB1 slurm/train_exp4_stage2.slurm | grep -o '[0-9]*')
echo "Stage 2: Job $JOB2 (depends on $JOB1)"

# Stage 3 (waits for Stage 2)
JOB3=$(sbatch --dependency=afterok:$JOB2 slurm/train_exp4_stage3.slurm | grep -o '[0-9]*')
echo "Stage 3: Job $JOB3 (depends on $JOB2)"
```

**Advantages:**
- ✅ Can monitor each stage separately
- ✅ Can cancel/restart individual stages if needed
- ✅ Better for debugging

---

## Option 3: Interactive Submission Script

Use the interactive submission script:

```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments
./SUBMIT_EXP4_RETRAIN.sh
```

Follow the prompts to choose your option.

---

## What to Watch For

### During Training

**Stage 1:**
- ✅ "Creating new LoRA adapter..."
- ✅ "trainable params: ~2,621,440" (LoRA only)
- ✅ Training completes successfully

**Stage 2:**
- ✅ "*** CURRICULUM LEARNING MODE ***"
- ✅ "Continuing training from previous checkpoint: .../stage1"
- ✅ "Loading existing LoRA adapter (will continue training the same adapter)..."
- ✅ "✓ Loaded previous adapter - continuing training on the same adapter"
- ✅ **First validation loss should be close to Stage 1's final loss** (no big jump)

**Stage 3:**
- ✅ Same checks as Stage 2
- ✅ Smooth loss continuation from Stage 2

### After Training

**Verify final model:**
```bash
python verify_exp4_fix.py models/exp4_curriculum/stage3
```

**Evaluate:**
```bash
python ../scripts/evaluation/evaluate_exp4.py \
  --model_path models/exp4_curriculum/stage3 \
  --test_data ../datasets/qlora_experiments/exp1_random/test.jsonl \
  --image_dir ../../datasets/Kvasir-VQA/raw/images \
  --output results/exp4_evaluation.json
```

**Expected:** ~92% accuracy (matching Exp1-3)

---

## Quick Commands

```bash
# Submit all-in-one
sbatch slurm/train_exp4_all_stages.slurm

# Check job status
squeue -u $USER

# View latest log
tail -f slurm/logs/exp4_all_*.out

# Cancel job (if needed)
scancel <JOB_ID>
```

---

**Ready to submit!** 🚀

