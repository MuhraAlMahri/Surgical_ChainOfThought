# Exp1 Refactored - Job Status

## 🚀 Job Pipeline

### Current Submission Status

| Step | Job ID | Status | Duration | Description |
|------|--------|--------|----------|-------------|
| **1** | 153578 | ⏳ Queued | 30 min | Sanity overfit test (64 samples, 200 steps) |
| **2** | Pending | - | 8 hours | Full training (after sanity passes) |
| **3** | Pending | - | 2 hours | Prediction generation (after training) |
| **4** | Pending | - | 15 min | Evaluation (after prediction) |

**Total Pipeline Time:** ~10.75 hours

---

## 📊 What Each Step Does

### Step 1: Sanity Overfit (Job 153578)

**Purpose:** Validate label masking works before committing to full training

**What it tests:**
- Dataset loads correctly
- Label masking (-100 for prompts)
- Model can learn short answers
- Training loop works

**Expected outcome:**
- ✅ **Loss < 2.0:** Label masking works! Proceed to full training.
- ⚠️ **Loss 2.0-5.0:** Partial success, may need adjustment.
- ❌ **Loss > 7.0:** Label masking bug - DO NOT proceed!

**Log location:** `/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm/logs/sanity_153578.out`

---

### Step 2: Full Training

**What happens:**
1. Auto-enriches JSONL data (adds question_type, candidates)
2. Loads Qwen2-VL-7B-Instruct
3. Applies LoRA (r=16, alpha=32)
4. Freezes vision tower
5. Trains for 1 epoch with instruction SFT

**Configuration:**
- Batch size: 4
- Gradient accumulation: 16 (effective batch = 64)
- Learning rate: 1e-4
- BFloat16 precision
- Max sequence length: 512

**Output:** `exp1/outputs/` (LoRA adapters + checkpoints)

---

### Step 3: Prediction Generation

**What happens:**
1. Loads trained model from `exp1/outputs/`
2. Runs inference on validation set
3. Uses **constrained decoding** for yes/no, color, MCQ
4. Generates short answers (max 4 tokens)

**Features:**
- Type-aware generation
- Forced valid answers for constrained types
- Fast inference (max 4 tokens)

**Output:** `exp1/outputs/predictions.jsonl`

---

### Step 4: Evaluation

**What happens:**
1. Loads predictions.jsonl
2. Computes per-type accuracy
3. Uses numeric tolerance for size/count
4. Generates detailed breakdown

**Metrics computed:**
- Per-type accuracy (yes/no, color, size, count, open)
- Overall micro-average
- Counts and correct predictions per type

**Output:** Printed to SLURM log + console

---

## 📋 Monitoring Commands

### Check job status
```bash
squeue -u muhra.almahri | grep exp1
```

### View live log (training)
```bash
tail -f exp1/slurm/logs/train_JOBID.out
```

### Check sanity overfit result
```bash
tail -30 exp1/slurm/logs/sanity_153578.out
```

### Cancel all jobs
```bash
scancel 153578 TRAIN_JOB_ID PREDICT_JOB_ID EVAL_JOB_ID
```

---

## 🎯 Expected Results

### After Sanity Overfit (30 min):
```
Final training loss: 1.45
✓ PASS - Loss dropped as expected. Label masking is working!
```

### After Full Training (8 hours):
```
Epoch 1/1: 100%
Train loss: ~1.5
Eval loss: ~1.7
Model saved to: exp1/outputs/
```

### After Prediction (2 hours):
```
Generated 8,984 predictions -> exp1/outputs/predictions.jsonl
```

### After Evaluation (15 min):
```
================================================================================
EVALUATION RESULTS - EXP1
================================================================================

Question Type        Count      Correct    Accuracy  
--------------------------------------------------------------------------------
yes_no               3,071      1,853      60.34%
color                491        98         19.96%
count_numeric        1,544      85         5.50%
size_numeric         498        10         2.01%
open_ended           3,380      450        13.31%
--------------------------------------------------------------------------------
OVERALL (micro)      8,984      2,496      27.78%
================================================================================
```

**Improvement over baseline:** 19.56% → 27.78% (+8.22%)

---

## 🔍 Troubleshooting

### Job fails immediately

**Check:** Error log in `slurm/logs/`
**Common issues:**
- Import errors (check Python paths)
- Missing data files (check config paths)
- CUDA out of memory (reduce batch size)

### Loss not dropping

**Check:** `outputs/trainer_state.json`
**Possible causes:**
- Learning rate too high/low
- Label masking not working (run sanity_overfit.py)
- Data preprocessing issues

### Predictions are wrong format

**Check:** `outputs/predictions.jsonl` sample
**Possible issues:**
- Prompt template not working
- Constrained decoding not active
- Max tokens too high

---

## 📁 File Locations

All output files relative to: `/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/`

```
exp1/
├── slurm/logs/          # SLURM job logs
├── outputs/             # Training outputs
│   ├── checkpoint-*/    # Training checkpoints
│   ├── adapter_model.safetensors  # LoRA weights
│   ├── trainer_state.json
│   └── predictions.jsonl
└── ...
```

---

## ⏰ Timeline

| Time | Event |
|------|-------|
| Now | Sanity job queued (153578) |
| +5 min | Sanity job starts |
| +35 min | Sanity completes, check results |
| +35 min | Submit training if sanity passed |
| +8h 35min | Training completes |
| +8h 35min | Prediction starts automatically |
| +10h 35min | Prediction completes |
| +10h 35min | Evaluation starts automatically |
| +10h 50min | **COMPLETE! Final results available** |

---

**Current Status:** Waiting for sanity job to run (Job 153578)

**Next Manual Action:** Review sanity results, then submit full pipeline if passed.


















