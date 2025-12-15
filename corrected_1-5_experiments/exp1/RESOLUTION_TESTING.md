# Resolution Testing Guide

This guide helps you find the optimal image resolution for training - balancing speed and accuracy.

## Quick Start

### 1. Submit the Test Job

```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments
sbatch exp1/slurm/test_resolutions.slurm
```

**Expected runtime:** 3-5 hours  
**Tests:** 4 different resolutions with 50 training steps each

### 2. Monitor Progress

```bash
# Check job status
squeue -u muhra.almahri

# Watch the output log (replace JOBID with actual job ID)
tail -f exp1/slurm/logs/resolution_test_JOBID.out
```

### 3. Analyze Results

After the job completes:

```bash
python3 exp1/analyze_resolution_results.py
```

This will show:
- Speed comparison (time per step, estimated full training time)
- Quality comparison (evaluation loss)
- Recommendations based on speed/quality tradeoff

## What Gets Tested

| Resolution | Size | Expected Image Tokens | Expected Training Time | Use Case |
|------------|------|----------------------|------------------------|----------|
| **Low (448×448)** | 448×448 | ~400 | ~13 hours | Fast iteration, like previous experiments |
| **Medium (768×768)** | 768×768 | ~1,500 | ~25 hours | Balanced speed/quality |
| **High (1024×1024)** | 1024×1024 | ~2,000 | ~35 hours | Better quality |
| **Full (~1036×1288)** | Adaptive | ~2,622 | ~50 hours | Maximum quality (current) |

## Understanding Results

### Key Metrics

1. **Time/Step**: How many seconds per training step
   - Lower = faster training
   - Target: <10s for fast, <20s acceptable

2. **Estimated Full Training**: Total hours for 3 epochs on full dataset
   - Your full training = 7,704 steps
   - Based on actual measured speed

3. **Eval Loss**: Model performance after 50 steps
   - Lower = better quality
   - Small differences (<0.05) may not be significant

### Example Output

```
Resolution Test Name      Time/Step    Full Train    Eval Loss    Speedup
low_res_448               6.2s         13.0h         0.8234       3.8x
medium_res_768            12.5s        26.0h         0.7891       1.9x
high_res_1024             18.3s        38.0h         0.7756       1.3x
full_res_current          23.5s        49.0h         0.7702       1.0x
```

### Decision Guide

**Choose Low (448×448) if:**
- ✅ Loss difference < 5% vs full resolution
- ✅ You want fast iteration
- ✅ You need results quickly (within 1 day)

**Choose Medium (768×768) if:**
- ✅ Loss improvement > 2% vs low
- ✅ Can afford 24-30 hours training
- ✅ Want balanced approach

**Choose High/Full if:**
- ✅ Medical image details are critical
- ✅ Can afford 40-50 hours
- ✅ Need maximum accuracy

## Applying Your Choice

After deciding on a resolution, update your config:

### Option 1: Modify dataset.py (Recommended)

Edit `exp1/dataset.py` around line 14:

```python
self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Add these lines to set resolution:
if hasattr(self.processor, 'image_processor'):
    self.processor.image_processor.min_pixels = 200704   # For 448x448
    self.processor.image_processor.max_pixels = 200704   # For 448x448
```

**Resolution settings:**
- 448×448: `min_pixels = max_pixels = 200704`
- 768×768: `min_pixels = max_pixels = 589824`
- 1024×1024: `min_pixels = max_pixels = 1048576`
- Full adaptive: `min_pixels = 3136, max_pixels = 12845056`

### Option 2: Update config file

Edit `exp1/config_exp1_category_based.yaml`:

```yaml
train:
  max_seq_len: 800   # For 448x448 (was 2900)
  train_bs: 2        # Can increase with shorter sequences
  grad_accum: 8      # Keep effective batch = 16
```

**Recommended max_seq_len by resolution:**
- 448×448: 800
- 768×768: 1800  
- 1024×1024: 2500
- Full: 2900

## Advanced: Custom Resolution

To test your own resolution:

```python
# Edit test_resolutions.py, add to resolutions list:
("custom_600", 360000, 360000, "600x600 custom test"),
```

Then resubmit the test job.

## Troubleshooting

### Job fails with OOM (Out of Memory)

- Full resolution might be too large for your GPU
- Try medium (768×768) or low (448×448) instead

### Results show NaN loss

- Training might need more steps
- Increase `--max_steps` in the test script (e.g., to 100)

### No speedup observed

- GPU might be bottlenecked by I/O or preprocessing
- Check that gradient checkpointing is enabled in config

## Files Created

After running tests:

```
exp1/resolution_tests/
├── comparison_summary.json       # All results in one file
├── low_res_448_results.json     # Individual test results
├── medium_res_768_results.json
├── high_res_1024_results.json
└── full_res_current_results.json
```

## Next Steps

1. Run resolution test (3-5 hours)
2. Analyze results with `analyze_resolution_results.py`
3. Choose optimal resolution based on speed/quality tradeoff
4. Update config and dataset.py with chosen resolution
5. Restart full training with optimized settings

**Goal:** Find the sweet spot where you get good quality without excessive training time!







