# Fixed SLURM Submission Guide

## Issues Fixed

### Issue 1: "idle/monitoring commands" rejection
**Problem:** SLURM rejected scripts with `tail` commands  
**Fix:** Removed `tail` commands and replaced with `awk` alternatives

### Issue 2: Dependency job ID placeholder
**Problem:** Using `NEW_JOB_ID` placeholder doesn't work  
**Fix:** Created helper scripts to capture and use actual job IDs

## ✅ Fixed Scripts

- `slurm/01_categorize_questions_v2.slurm` - Fixed (no more `tail` or `tee`)
- `slurm/submit_categorization.sh` - New helper script
- `slurm/submit_with_dependency.sh` - New helper for dependencies

## 🚀 How to Submit Jobs

### Option 1: Use Helper Scripts (Recommended)

**Step 1: Submit categorization**
```bash
cd /l/users/muhra.almahri/Surgical_COT
./slurm/submit_categorization.sh kvasir
```

This will:
- Submit the job
- Show you the job ID
- Give you the exact command to submit training

**Step 2: Submit training with dependency**
```bash
# Use the job ID from Step 1
./slurm/submit_with_dependency.sh CATEG_JOB_ID kvasir Qwen/Qwen3-VL-8B-Instruct
```

### Option 2: Manual Submission

**Step 1: Submit categorization**
```bash
JOB_ID=$(sbatch --parsable slurm/01_categorize_questions_v2.slurm kvasir)
echo "Categorization job ID: $JOB_ID"
```

**Step 2: Submit training with dependency**
```bash
# Replace CATEG_JOB_ID with the actual job ID from Step 1
sbatch --dependency=afterok:CATEG_JOB_ID slurm/03_train_unified.slurm kvasir
```

### Option 3: Complete Pipeline

```bash
# Submit all jobs with dependencies automatically
sbatch slurm/submit_all.sh kvasir Qwen/Qwen3-VL-8B-Instruct
```

## 📋 Quick Reference

```bash
# Check job status
squeue -u $USER

# View logs (after job starts)
tail -f slurm/logs/categorize_questions_v2_JOBID.out

# Cancel job
scancel JOB_ID

# Check if categorization completed
ls -lh data/categorized/train_categorized.json
```

## ⚠️ Important Notes

1. **Job IDs are required** - Don't use placeholders like `NEW_JOB_ID`
2. **Wait for categorization** - Training needs categorized data to exist
3. **Check logs** - If a job fails, check the `.err` file
4. **Disk quota** - The v2 script uses `/tmp` to avoid quota issues

## 🔧 Troubleshooting

### Job rejected for monitoring commands
- ✅ Fixed in v2 script
- If you see this error, make sure you're using `01_categorize_questions_v2.slurm`

### Dependency error
- Make sure the job ID exists: `squeue -j JOB_ID`
- Or check history: `sacct -j JOB_ID`
- Use the helper script: `./slurm/submit_with_dependency.sh JOB_ID`

### File not found errors
- Check that input files exist before submitting
- Verify paths in the script match your dataset location














