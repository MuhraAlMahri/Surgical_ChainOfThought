# Cleanup Proposal - Files Safe to Delete

## Summary of Findings

I've identified several categories of files that appear to be safe for cleanup. **Please review and approve before deletion.**

## 1. Temporary/Progress Files (SAFE TO DELETE)

### Location: `corrected experiments/datasets/`
These are temporary files from dataset processing that are no longer needed:

- `kvasir_raw_qwen_reordered.json.progress.log` (2.6 KB)
- `kvasir_raw_qwen_reordered.json.tmp.ndjson` (12 MB) ⚠️ **Largest file**
- `kvasir_qwen_stage_ordered_image_level_70_15_15/train.json.progress.log`
- `kvasir_qwen_stage_ordered_image_level_70_15_15/train.json.tmp.ndjson`
- `kvasir_qwen_stage_ordered_image_level_70_15_15/val.json.progress.log`
- `kvasir_qwen_stage_ordered_image_level_70_15_15/val.json.tmp.ndjson`
- `kvasir_qwen_stage_ordered_image_level_70_15_15/test.json.progress.log`
- `kvasir_qwen_stage_ordered_image_level_70_15_15/test.json.tmp.ndjson`

**Status:** These are checkpoint/temp files from dataset processing scripts. The final JSON files already exist, so these are safe to delete.

**Estimated space saved:** ~15-20 MB

## 2. Old Log Files (POTENTIALLY SAFE)

### Location: `corrected experiments/logs/`
- Total size: 32 MB
- Total files: 100 .out + 100 .err = 200 log files

**These are from OLD experiments (not current "corrected 1-5 experiments")**

**Recommendation:** Keep logs from:
- Currently running jobs (if any)
- Recent successful/failed experiments for debugging
- Delete logs older than 30 days or from clearly failed/obsolete runs

**Would you like me to:**
- A) List all log files with dates so you can choose which to keep?
- B) Delete logs older than a specific date?
- C) Keep only logs from specific job IDs you specify?

## 3. Old Experiment Results (NEED CONFIRMATION)

### Location: `corrected experiments/results/`
Need to check if these are still needed for comparison or can be archived/deleted.

## 4. Other Temporary Files

- `.DS_Store` files (Mac system files) - Safe to delete
- `.swp` files (vim swap files) - Safe to delete if no unsaved edits
- `*~` backup files - Safe to delete

## What I Will NOT Delete (Without Explicit Permission)

✅ Model checkpoints and trained models
✅ Dataset JSON files (train.json, val.json, test.json)
✅ Training scripts (.slurm, .py files)
✅ Documentation (.md files)
✅ Active job logs (jobs currently running)
✅ Configuration files

## Proposed Cleanup Actions

**Please confirm which you'd like me to proceed with:**

1. **Delete all `.tmp.ndjson` and `.progress.log` files** (saves ~15-20 MB)
2. **Clean old log files** (options above - A, B, or C)
3. **Delete system files** (.DS_Store, .swp, *~)
4. **Check for other temporary files** in `corrected experiments/` folder

