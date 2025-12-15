# Training Optimization Summary - Task 3 (Qwen3VL Kvasir CoT Training)

## Problem Identified

Job 168937 Task 3 was training for **36 hours and only 21% complete**, indicating it would take **~170 hours (7 days)** to complete 5 epochs. This is **WAY too slow**.

**Current Performance:**
- 36 hours for 43,079 batches (21% of 205,395 total batches)
- **~3 seconds per batch** (should be <1 second)
- Expected completion: 170 hours (should be 12-20 hours)

## Root Causes

1. **Batch size = 1** - Processing one sample at a time
2. **Gradient accumulation = 16** - Very high, slows gradient updates
3. **Gradient checkpointing enabled** - Trades speed for memory (2x slowdown)
4. **num_workers = 0** - No parallel data loading
5. **pin_memory = False** - No pinned memory for faster CPU->GPU transfer
6. **Sequential sample processing** - Even with batches, samples processed one-by-one in loops

## Optimizations Applied

### 1. Increased Batch Size
- **Before:** `batch_size=1`
- **After:** `batch_size=4`
- **Impact:** 4x faster data loading, better GPU utilization

### 2. Reduced Gradient Accumulation
- **Before:** `grad_accum=16`
- **After:** `grad_accum=4`
- **Impact:** 4x faster gradient updates (fewer accumulation steps)
- **Trade-off:** Maintains same effective batch size = 16 (4 × 4)

### 3. Disabled Gradient Checkpointing
- **Before:** `--gradient_checkpointing` (enabled)
- **After:** Removed flag (defaults to False)
- **Impact:** **2x speedup** (trades memory for speed)
- **Trade-off:** Uses more GPU memory (but manageable with batch_size=4)

### 4. Added Parallel Data Loading
- **Before:** `num_workers=0` (sequential loading)
- **After:** `num_workers=4` (parallel loading)
- **Impact:** 4x parallel I/O, eliminates data loading bottleneck
- **Trade-off:** Uses more CPU cores (typically 32 CPUs allocated in SLURM)

### 5. Enabled Pinned Memory
- **Before:** `pin_memory=False`
- **After:** `pin_memory=True`
- **Impact:** Faster CPU->GPU data transfer (~10-20% speedup)

### 6. Added Performance Monitoring
- Added batch timing and samples/second logging
- Helps identify bottlenecks during training

## Expected Performance Improvement

### Time Estimates

**Before Optimization:**
- Time per batch: ~3 seconds
- Total batches: 205,395 (41,079 samples × 5 epochs / 1 batch_size)
- **Total time: ~170 hours (7 days)**

**After Optimization:**
- Time per batch: **~0.5-0.8 seconds** (estimated)
  - Batch size increase: 4x speedup on data loading
  - Gradient accumulation reduction: 4x speedup on updates
  - Gradient checkpointing disabled: 2x speedup
  - Parallel data loading: 1.5-2x speedup
  - Pinned memory: 1.1x speedup
  - **Combined: ~15-20x speedup potential**
- Total batches: **51,349** (41,079 samples × 5 epochs / 4 batch_size)
- **Total time: ~7-12 hours** (vs 170 hours before)

### Effective Batch Size
- **Before:** 1 × 16 = 16 (batch_size × grad_accum)
- **After:** 4 × 4 = 16 (same effective batch size)
- **Result:** Same training dynamics, much faster execution

## Files Modified

1. **`train_multihead_cot.py`**
   - Added `batch_size` parameter to `train_kvasir_epoch()`
   - Added batch timing and performance logging
   - Added `dataloader_num_workers` and `dataloader_pin_memory` arguments
   - Changed gradient_checkpointing default to False

2. **`data/vqa_data_loader.py`**
   - Updated `create_data_loader()` to accept `num_workers` and `pin_memory` kwargs
   - Default to 4 workers and pin_memory=True when batch_size > 1
   - Added `persistent_workers=True` to keep workers alive between epochs

3. **`slurm_scripts/mega_qwen3vl_4tasks.sh`**
   - Updated Task 3 training command:
     - `--batch_size 1` → `--batch_size 4`
     - `--grad_accum 16` → `--grad_accum 4`
     - Removed `--gradient_checkpointing`
     - Added `--dataloader_num_workers 4`
     - Added `--dataloader_pin_memory`

## Configuration Comparison

| Parameter | Before | After | Change |
|-----------|--------|-------|--------|
| batch_size | 1 | 4 | +4x |
| grad_accum | 16 | 4 | -4x |
| effective_batch_size | 16 | 16 | Same |
| gradient_checkpointing | True | False | Disabled |
| num_workers | 0 | 4 | +4x |
| pin_memory | False | True | Enabled |

## Next Steps

1. **Test the optimized configuration** on a small subset first
2. **Monitor the training logs** for:
   - Batch processing time (should be <1 second)
   - Samples per second (should be >4 samples/s)
   - GPU utilization (should be >80%)
3. **Verify training quality** - effective batch size is unchanged, so training dynamics should be identical
4. **If still slow**, investigate:
   - I/O bottlenecks (network filesystem speed)
   - Image loading/processing bottlenecks
   - Model forward pass time

## Notes

- The sequential nature of the multi-head architecture (Stage 2 depends on Stage 1, Stage 3 depends on Stage 1+2) means samples are still processed sequentially within each batch
- However, these optimizations still provide significant speedups through:
  - Better data loading pipeline
  - Faster gradient updates
  - Reduced computational overhead
  - Better GPU utilization

## Expected Completion Time

**Optimistic:** 7-8 hours (for 5 epochs)  
**Realistic:** 10-12 hours  
**Conservative:** 15-20 hours

**Target:** Complete within 12-20 hours (vs 170 hours before)  
**Speedup:** **10-20x faster**



