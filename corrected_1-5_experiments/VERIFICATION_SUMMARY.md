# Training Scripts Verification Summary

## ✅ 1. Time Limits
**Status: VERIFIED ✓**

All scripts have `--time=72:00:00` (72 hours), which is the maximum allowed for cscc-gpu-qos partition.

**Scripts checked:**
- exp1_random/train_random_baseline.slurm: 72:00:00 ✓
- exp2_qwen_reordered/train_qwen_reordered.slurm: 72:00:00 ✓
- exp3_cxrtrek_sequential/train_stage{1,2,3}.slurm: 72:00:00 ✓
- exp4_curriculum_learning/train_stage{1,2,3}.slurm: 72:00:00 ✓

## ✅ 2. Progress Tracking
**Status: VERIFIED ✓**

Training script (`train_qwen_lora.py`) includes:
- `logging_steps=10` - Logs every 10 steps
- `save_steps=500` - Saves checkpoint every 500 steps
- `eval_steps=500` - Evaluates every 500 steps
- `save_total_limit=3` - Keeps last 3 checkpoints

**Progress monitoring:**
```bash
# View live logs
tail -f "corrected 1-5 experiments/logs/exp1_random_<JOB_ID>.out"

# Check checkpoint progress
ls -lh "corrected 1-5 experiments/models/exp1_random_baseline/checkpoint-*"

# Monitor job status
squeue -u muhra.almahri
```

**Estimated checkpoint frequency:**
- Train samples: 41,079
- Batch size: 1, Gradient accumulation: 16
- Steps per epoch: ~2,567
- Total steps (3 epochs): ~7,702
- Checkpoints saved: ~every 500 steps = ~15 checkpoints total

## ✅ 3. Model Configuration
**Status: VERIFIED ✓**

All scripts use: `--model_name "Qwen/Qwen2-VL-7B-Instruct"`

**Verified in:**
- exp1_random/train_random_baseline.slurm ✓
- exp2_qwen_reordered/train_qwen_reordered.slurm ✓
- exp3_cxrtrek_sequential/train_stage{1,2,3}.slurm ✓
- exp4_curriculum_learning/train_stage{1,2,3}.slurm ✓

## ✅ 4. Dataset Paths (Qwen-Reordered, Image-Level Split)
**Status: VERIFIED ✓**

### Experiment 1 & 2 (Full Dataset):
```
DATASET_DIR: corrected 1-5 experiments/datasets/kvasir_raw_6500_image_level_70_15_15/
TRAIN_FILE: train.json (41,079 QA pairs from 4,550 images)
VAL_FILE: val.json (8,786 QA pairs from 975 images)
```

**Dataset verification:**
- ✓ Image-level split (no image overlap between train/val/test)
- ✓ Qwen-reordered (has 'stage' field: 1, 2, or 3)
- ✓ Split ratio: 70/15/15
- ✓ Total: 6,500 images, 58,849 QA pairs

**Sample entry structure:**
```json
{
  "image_id": "...",
  "image_filename": "...",
  "question": "...",
  "answer": "...",
  "stage": 1  // or 2, 3
}
```

### Experiment 3 & 4 (Stage-Specific Splits):
```
DATASET_DIR: corrected 1-5 experiments/datasets/
Stage 1: kvasir_stage_splits_stage1/{train,val,test}.json
Stage 2: kvasir_stage_splits_stage2/{train,val,test}.json
Stage 3: kvasir_stage_splits_stage3/{train,val,test}.json
```

**Stage split counts:**
- Stage 1: 14,679 train, 3,086 val, 3,275 test
- Stage 2: 26,357 train, 5,689 val, 5,703 test
- Stage 3: 43 train, 11 val, 6 test

**Verification:**
- ✓ Created from image-level split (preserves image-level separation)
- ✓ Filtered by 'stage' field from Qwen-reordered dataset
- ✓ No data leakage (same images never in train/test)

## 📋 Additional Configuration

**LoRA Settings:**
- LoRA r: 32
- LoRA alpha: 64
- LoRA dropout: 0.05

**Training Settings:**
- Epochs: 3
- Batch size: 1
- Gradient accumulation: 16
- Learning rate: 5e-6
- Max length: 128
- Memory: 128GB

**Cluster Settings:**
- Partition: cscc-gpu-p
- QoS: cscc-gpu-qos
- GPUs: 1 per job
- CPUs: 8 per job

## ✅ Final Status: READY TO SUBMIT

All requirements verified:
1. ✓ Time limits sufficient (72 hours max)
2. ✓ Progress trackable (logs, checkpoints every 500 steps)
3. ✓ Model is 7B (Qwen/Qwen2-VL-7B-Instruct)
4. ✓ Correct dataset (Qwen-reordered, image-level split, 70/15/15)

