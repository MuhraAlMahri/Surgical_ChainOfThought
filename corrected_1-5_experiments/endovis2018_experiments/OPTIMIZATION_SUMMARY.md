# Training Optimization Summary

## Goal
Train and evaluate all 5 experiments within 6 hours.

## Optimizations Applied

### 1. Batch Size Increase
- **Before**: `train_bs: 1`
- **After**: `train_bs: 4`
- **Impact**: 4x faster data loading, better GPU utilization
- **Trade-off**: Maintains effective batch size = 16 (4 × 4 grad_accum)

### 2. Gradient Accumulation Reduction
- **Before**: `grad_accum: 16`
- **After**: `grad_accum: 4`
- **Impact**: Faster gradient updates (4x fewer accumulation steps)
- **Trade-off**: None - effective batch size remains 16

### 3. Gradient Checkpointing Disabled
- **Before**: `gradient_checkpointing: true`
- **After**: `gradient_checkpointing: false`
- **Impact**: 2x speedup (trades memory for speed)
- **Trade-off**: Uses more GPU memory (but with batch_size=4, still manageable)

### 4. DataLoader Workers Increase
- **Before**: `dataloader_num_workers: 4` (hardcoded)
- **After**: `dataloader_num_workers: 16` (configurable)
- **Impact**: 4x parallel I/O, eliminates data loading bottleneck
- **Trade-off**: Uses more CPU cores (32 CPUs allocated)

### 5. Checkpoint Frequency
- **Before**: `save_steps: 500` (or stage-specific)
- **After**: `save_steps: ~125` (or stage-specific, ~1 epoch)
- **Impact**: Faster checkpoint visibility, better progress monitoring
- **Trade-off**: More checkpoints (but `save_total_limit=3` prevents disk bloat)

## Expected Performance

### Before Optimization
- Training time: ~20+ hours per experiment
- Total time (5 experiments): ~100+ hours
- Bottlenecks: Network I/O, small batch size, gradient checkpointing

### After Optimization
- Training time: ~1-2 hours per experiment (estimated)
- Total time (5 experiments in parallel): ~2-4 hours training + ~1-2 hours evaluation = **~4-6 hours total**
- Speedup: **4-8x faster**

## GPU Distribution

The optimized job runs all experiments in parallel:
- **GPU 0**: Exp1 → Exp5 (sequential on same GPU)
- **GPU 1**: Exp2
- **GPU 2**: Exp3 (Stage 1 → Stage 2, sequential)
- **GPU 3**: Exp4 (Stage 1 → Stage 2, sequential, curriculum)

## Configuration Files Updated

All experiment configs have been optimized:
- `exp1_random.yaml`
- `exp2_qwen_reordered.yaml`
- `exp3_stage1.yaml`
- `exp3_stage2.yaml`
- `exp3_stage3.yaml`
- `exp4_stage1.yaml`
- `exp4_stage2.yaml`
- `exp5_sequential_cot.yaml`

## Training Script Updated

- `train_qlora_qwen3vl.py`: Now uses `dataloader_num_workers` from config (default: 16)

## SLURM Job

- **Job file**: `slurm/mega_job_all_experiments_optimized.slurm`
- **Time limit**: 6 hours
- **GPUs**: 4
- **CPUs**: 32
- **Memory**: 240GB

## Notes

1. **Effective batch size remains 16**: All optimizations maintain the same effective batch size for fair comparison
2. **Memory usage**: With batch_size=4 and no gradient checkpointing, GPU memory usage increases but should still fit in 40GB A100s
3. **Network I/O**: Still a bottleneck, but 16 workers help parallelize it
4. **Evaluation**: Will run after training completes (separate job or same job)

## Monitoring

Check job status:
```bash
squeue -u muhra.almahri
```

Check logs:
```bash
tail -f slurm/logs/mega_job_all_optimized_<JOB_ID>.out
```

Monitor GPU usage:
```bash
ssh <node> nvidia-smi
```

