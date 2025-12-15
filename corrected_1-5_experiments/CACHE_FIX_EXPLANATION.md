# LLaVA-Med Disk Quota Fix - Explanation

## Problem Summary

Your training jobs were failing with **"Disk quota exceeded"** errors because:
- Each job created its own cache directory (`hf_cache_${JOB_ID}`) ~15GB each
- Cache grew to ~600GB with many old job-specific directories
- CLIP processor (`openai/clip-vit-large-patch14`) was being re-downloaded for each job
- No reuse of cached models between jobs

## Solution Overview

**Shared Cache Approach**: All jobs now use a single shared cache location (`/l/users/muhra.almahri/.cache/hf_shared`) so models are downloaded once and reused.

## Files Created/Modified

### 1. `cleanup_cache.sh` - Cache Cleanup Script

**What it does:**
- Shows current disk usage
- Lists all old job-specific cache directories (`hf_cache_*`)
- **Optionally deletes** old cache directories (frees ~400GB)
- Creates shared cache directory
- **Pre-downloads** CLIP processor and LLaVA-Med components to shared cache
- This ensures the models are available before jobs start

**Why it's needed:**
- Frees up disk space by removing duplicate caches
- Pre-downloads models so jobs don't hit quota during download
- Sets up the shared cache structure

**Usage:**
```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments
bash cleanup_cache.sh
# Answer 'y' when prompted to delete old caches
```

### 2. `qlora_experiments/train_llava_manual.py` - Updated Training Script

**What changed:**
- Added shared cache configuration at the **very top** (before any imports)
- Sets `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_HUB_CACHE` to shared location
- This ensures all HuggingFace operations use the shared cache

**Key difference from old version:**
- **Old**: Each job used `hf_cache_${JOB_ID}` (per-job cache)
- **New**: All jobs use `/l/users/muhra.almahri/.cache/hf_shared` (shared cache)

**No other changes needed** - all your existing fixes (image token ID, dataset fields, etc.) are preserved.

### 3. SLURM Scripts Updated

**Files modified:**
- `exp1/slurm/train_exp1_llava_v15_manual.slurm` (Kvasir-VQA)
- `endovis2018_experiments/slurm/train_exp1_llava_v15_manual.slurm` (EndoVis2018)

**What changed:**
- Removed per-job cache creation (`hf_cache_${SLURM_JOB_ID}`)
- Set all cache variables to shared cache location
- Removed cache copying logic (no longer needed)
- Added cache size display for monitoring

## Step-by-Step Usage

### Step 1: Run Cleanup Script (One Time)

```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments
bash cleanup_cache.sh
```

**What happens:**
1. Shows current cache usage (~600GB)
2. Lists old cache directories
3. Asks if you want to delete them → **Answer 'y'**
4. Deletes old caches (frees ~400GB)
5. Creates shared cache directory
6. Pre-downloads CLIP processor (prevents quota errors)
7. Pre-downloads LLaVA-Med config/tokenizer

**Expected output:**
```
✓ Old cache directories deleted
✓ Created shared cache: /l/users/muhra.almahri/.cache/hf_shared
✓ CLIP processor downloaded successfully
✓ Config downloaded
✓ Tokenizer downloaded
```

### Step 2: Submit Jobs (As Normal)

Your existing SLURM scripts are already updated! Just submit:

```bash
# Kvasir-VQA
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1/slurm
sbatch train_exp1_llava_v15_manual.slurm

# EndoVis2018
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/endovis2018_experiments/slurm
sbatch train_exp1_llava_v15_manual.slurm
```

**What happens now:**
- Jobs use shared cache (no per-job cache creation)
- CLIP processor is already cached (no download needed)
- LLaVA-Med model components are cached (faster loading)
- No disk quota errors!

## Customization Needed

### ✅ Already Configured:
- Username: `muhra.almahri`
- Shared cache path: `/l/users/muhra.almahri/.cache/hf_shared`
- Partition: `cscc-gpu-p`
- QOS: `cscc-gpu-qos`
- Conda environment: `base` (from your SLURM scripts)

### ⚠️ Check These:
1. **Conda environment name**: Your SLURM scripts use `source ~/miniconda3/bin/activate base`
   - If you use a different environment, update the SLURM scripts

2. **Dataset paths**: Already correct in your config files
   - Kvasir: `/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_ULTRA_CONDENSED/`
   - EndoVis: `/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/endovis18_surgery_r1_split/`

## Expected Results After Fix

With disk quota fixed, your training should proceed smoothly:

- **Kvasir-VQA**: ~92-93% accuracy (matching Qwen3-VL results)
- **EndoVis2018**: ~95-99% accuracy (matching MedGemma results)

## Monitoring

After submitting jobs, check:
```bash
# Job status
squeue -u muhra.almahri

# Cache usage (should stay stable, not grow per job)
du -sh /l/users/muhra.almahri/.cache/hf_shared

# Training progress
tail -f exp1/slurm/slurm/logs/train_exp1_llava_v15_manual_*.err
```

## Troubleshooting

**If you still get disk quota errors:**
1. Check if cleanup script ran successfully
2. Verify shared cache exists: `ls -la /l/users/muhra.almahri/.cache/hf_shared`
3. Check disk quota: `quota -s`
4. Clean up more old caches manually if needed

**If jobs fail with "CLIP processor not found":**
1. Re-run cleanup script to pre-download it
2. Check network connectivity (download might have failed)

## Summary

✅ **Cleanup script**: Removes old caches, sets up shared cache, pre-downloads models  
✅ **Training script**: Uses shared cache (no code logic changes)  
✅ **SLURM scripts**: Updated to use shared cache instead of per-job cache  

**Result**: No more disk quota errors, faster job startup (models already cached), and ~400GB disk space freed!





