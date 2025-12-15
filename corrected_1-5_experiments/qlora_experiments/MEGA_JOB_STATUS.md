# Mega Job Status: Exp4 Retraining + All Evaluations

## Job Information

**Job ID:** 157551  
**Job Name:** mega_exp4_retrain_evals  
**Status:** Submitted  
**Start Time:** Check with `squeue -j 157551`

## What This Job Does

### Phase 1: Parallel Execution (4 GPUs)
- **GPU 0:** Exp1 Evaluation (rerun with new metrics)
- **GPU 1:** Exp2 Evaluation (rerun with new metrics)
- **GPU 2:** Exp3 Evaluation (rerun with new metrics)
- **GPU 3:** Exp4 Stage 1 Training (starts curriculum learning)

### Phase 2: Exp4 Stage 2 Training
- **GPU 3:** Continues from Stage 1 (curriculum learning)

### Phase 3: Exp4 Stage 3 Training
- **GPU 3:** Continues from Stage 2 (curriculum learning)

### Phase 4: Exp4 Evaluation
- **GPU 3:** Evaluates the retrained Exp4 model

## Expected Timeline

- **Phase 1:** ~7 hours (Exp4 Stage 1 training is the longest)
- **Phase 2:** ~12 hours (Exp4 Stage 2 training)
- **Phase 3:** ~30 minutes (Exp4 Stage 3 training)
- **Phase 4:** ~2 hours (Exp4 evaluation)
- **Total:** ~22 hours

## Monitor Progress

```bash
# Check job status
squeue -j 157551

# View main log
tail -f slurm/logs/mega_exp4_retrain_evals_157551.out

# View individual GPU logs
tail -f slurm/logs/mega_exp4_gpu0_157551.out  # Exp1 eval
tail -f slurm/logs/mega_exp4_gpu1_157551.out  # Exp2 eval
tail -f slurm/logs/mega_exp4_gpu2_157551.out  # Exp3 eval
tail -f slurm/logs/mega_exp4_s1_157551.out    # Exp4 Stage 1
tail -f slurm/logs/mega_exp4_s2_157551.out    # Exp4 Stage 2
tail -f slurm/logs/mega_exp4_s3_157551.out    # Exp4 Stage 3
tail -f slurm/logs/mega_exp4_gpu3_157551.out  # Exp4 eval
```

## Expected Results

### Evaluation Results (with new metrics)
- **Exp1:** `results/exp1_evaluation.json` (Accuracy, Precision, Recall, F1)
- **Exp2:** `results/exp2_evaluation.json` (Accuracy, Precision, Recall, F1)
- **Exp3:** `results/exp3_evaluation.json` (Accuracy, Precision, Recall, F1)

### Exp4 Retraining Results
- **Stage 1:** `models/exp4_curriculum/stage1/`
- **Stage 2:** `models/exp4_curriculum/stage2/`
- **Stage 3:** `models/exp4_curriculum/stage3/`
- **Evaluation:** `results/exp4_evaluation.json`

### Expected Exp4 Accuracy
- **Before fix:** 62.88%
- **After retraining:** ~92% (matching Exp1-3)

## What to Watch For

### During Exp4 Training

**Stage 1:**
- ✅ "Creating new LoRA adapter..."
- ✅ "trainable params: ~2,621,440"

**Stage 2:**
- ✅ "*** CURRICULUM LEARNING MODE ***"
- ✅ "Loading existing LoRA adapter (will continue training the same adapter)..."
- ✅ First validation loss should be close to Stage 1's final loss (no big jump)

**Stage 3:**
- ✅ Same checks as Stage 2
- ✅ Smooth loss continuation

### After Completion

Check the evaluation results:
```bash
# View Exp4 results
cat results/exp4_evaluation.json | python3 -m json.tool | head -20

# Compare with previous (broken) results
cat results/exp4_evaluation.json | grep -A 5 '"accuracy"'
# Should show ~92% instead of 62.88%
```

## Cancel Job (if needed)

```bash
scancel 157551
```

---

**Job submitted:** 2025-01-XX  
**Status:** Running  
**Monitor:** `squeue -j 157551`

