# Training & Evaluation Timeline Estimates

## Dataset & Configuration

**Training Samples:** 41,079 (full dataset)  
**Batch Size:** 1 per device  
**Gradient Accumulation:** 16 steps  
**Effective Batch Size:** 16  
**Epochs:** 3  
**Model:** Qwen/Qwen2-VL-7B-Instruct (larger than previous 2B model)

**Steps Calculation:**
- Steps per epoch: ~2,568
- Total training steps: ~7,704

## Per-Experiment Time Estimates

### Full Dataset Experiments (Exp1, Exp2)
- **Optimistic:** 4-6 hours
- **Realistic:** 6-9 hours  
- **Conservative:** 9-12 hours
- **Time limit:** 24 hours (sufficient)

### Stage-Specific Experiments (Exp3, Exp4)

**Stage 1 (14,679 samples):**
- Steps per epoch: ~918
- **Estimated time:** 2-3 hours per epoch → ~6-9 hours total

**Stage 2 (26,357 samples):**
- Steps per epoch: ~1,648
- **Estimated time:** 4-6 hours per epoch → ~12-18 hours total

**Stage 3 (43 samples - very small!):**
- Steps per epoch: ~3
- **Estimated time:** <5 minutes total (negligible)

## Experiment Dependencies & Scheduling

### Current Status (as of now):
- **Exp1:** RUNNING (Job 150885)
- **Exp2:** PENDING (Job 150886) 
- **Exp3 Stage 1:** PENDING (Job 150887)
- **Exp3 Stages 2-3:** Waiting for slots
- **Exp4:** Waiting for slots
- **Exp5:** Will run after Exp1 or Exp2 (evaluation only)

### Timeline Scenarios

#### **Scenario 1: Best Case (All Slots Available)**
If all experiments can run in parallel when slots are available:

| Phase | Experiments | Time | Parallel? |
|-------|-------------|------|-----------|
| Phase 1 | Exp1, Exp2, Exp3 (all 3 stages) | ~12 hours | ✅ Yes |
| Phase 2 | Exp4 (sequential: S1→S2→S3) | ~24 hours | ❌ No |
| Phase 3 | Exp5 (evaluation) | ~2-4 hours | - |
| **Total** | | **~38-40 hours** | **~1.6 days** |

#### **Scenario 2: Realistic (4 Job Limit)**
Account for QoS limits and sequential submissions:

| Timeline | Action | Duration |
|----------|--------|----------|
| Now → 6-9h | Exp1 completes | 6-9 hours |
| Now → 6-9h | Exp2 starts (parallel) | 6-9 hours |
| After Exp1 | Exp3 S1 starts | 6-9 hours |
| After Exp1 | Exp3 S2 starts | 12-18 hours |
| After Exp1 | Exp3 S3 starts | <1 hour |
| After Exp2 | Exp5 evaluation | 2-4 hours |
| After Exp3 S1 | Exp4 S1 starts | 6-9 hours |
| After Exp4 S1 | Exp4 S2 starts | 12-18 hours |
| After Exp4 S2 | Exp4 S3 starts | <1 hour |

**Total Realistic Time:** ~50-70 hours (**2-3 days**)

#### **Scenario 3: Conservative (Maximum Wait Times)**
If jobs hit time limits and queue delays:

- Each job: Up to 24 hours
- Job queue delays: Variable
- **Total Conservative Time:** ~5-7 days

## Evaluation Time (Experiment 5)

- **Dependencies:** Requires Exp1 OR Exp2 to complete
- **Dataset:** Test set (8,984 QA pairs from 975 images)
- **Processing:** Sequential cascading inference (Stage 1→2→3)
- **Estimated time:** 2-4 hours

## Key Factors Affecting Time

### ⚡ Speed Factors:
1. **GPU availability** - GPU 1 now available (not conflicting)
2. **7B model** - Larger than previous 2B, will be slower
3. **Vision-Language** - More compute than text-only
4. **Dataset size** - 41K samples is substantial

### 🐌 Slowing Factors:
1. **QoS limits** - Max 4 jobs (causes queue delays)
2. **Sequential dependencies** - Exp4 must run sequentially
3. **Small batch size** - batch_size=1 is slow but necessary for memory
4. **Gradient accumulation** - 16 steps means slower updates

## Realistic Timeline Summary

**Best Case:** ~1.5-2 days  
**Realistic:** ~2-3 days  
**Worst Case:** ~5-7 days  

**Expected Completion:** ~**2-3 days** from now (accounting for current job queue and dependencies)

## Current Progress Tracking

Monitor with:
```bash
# Check job status
squeue -u muhra.almahri

# Monitor active training
tail -f "corrected 1-5 experiments/logs/exp1_random_150885.out"

# Check for checkpoints
ls -lh "corrected 1-5 experiments/models/"*/checkpoint-*
```

## Notes

- Previous experiments with **2B model** took:
  - Stage 1: 2h 20min
  - Stage 2: 5h 31min
  - Stage 3: 31min
  
- Current **7B model** will be ~2-3x slower due to larger model size
- But dataset processing and setup time should be similar
- Overall, expect 2-3x longer than previous 2B experiments

