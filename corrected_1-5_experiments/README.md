# Corrected Experiments 1-5

This directory contains the rerun of experiments 1-4 using the corrected dataset split.

## Dataset Information

**Location:** `datasets/kvasir_raw_6500_image_level_70_15_15/`

**Split Details:**
- **Split Method:** Image-level (zero overlap between splits)
- **Split Ratio:** 70/15/15 (train/val/test)
- **Total Images:** 6,500
- **Total QA Pairs:** 58,849
- **Training:** 41,079 QA pairs from 4,550 images
- **Validation:** 8,786 QA pairs from 975 images
- **Test:** 8,984 QA pairs from 975 images

**Source File:** `kvasir_raw_qwen_reordered.json` - All QA pairs reordered with Qwen, with stage field (1, 2, or 3)

**Stage-Specific Splits:** Created in `datasets/kvasir_stage_splits_stage{1,2,3}/` for experiments 3 and 4.

## Experiments

### Experiment 1: Random Baseline
**Directory:** `experiments/exp1_random/`  
**Script:** `train_random_baseline.slurm`  
**Description:** Standard training with random question ordering (no stage organization).  
**Dataset:** Uses main train/val/test splits  
**Output:** `models/exp1_random_baseline/`

**Submit:**
```bash
cd "corrected 1-5 experiments"
sbatch experiments/exp1_random/train_random_baseline.slurm
```

---

### Experiment 2: Qwen Ordering
**Directory:** `experiments/exp2_qwen_reordered/`  
**Script:** `train_qwen_reordered.slurm`  
**Description:** Training with Qwen-reordered questions organized by clinical flow stages, but trained all at once (not sequentially).  
**Dataset:** Uses main train/val/test splits (already reordered)  
**Output:** `models/exp2_qwen_reordered/`

**Submit:**
```bash
sbatch experiments/exp2_qwen_reordered/train_qwen_reordered.slurm
```

---

### Experiment 3: CXRTrek Sequential
**Directory:** `experiments/exp3_cxrtrek_sequential/`  
**Scripts:** `train_stage1.slurm`, `train_stage2.slurm`, `train_stage3.slurm`  
**Description:** Training of THREE separate specialized models - one for each clinical stage. All models trained independently from the base model (no checkpoint loading).  
**Approach:** Parallel training (can submit all 3 stages simultaneously)  
**Dataset:** Uses stage-specific splits:
- Stage 1: `datasets/kvasir_stage_splits_stage1/`
- Stage 2: `datasets/kvasir_stage_splits_stage2/`
- Stage 3: `datasets/kvasir_stage_splits_stage3/`

**Outputs:**
- `models/exp3_cxrtrek_seq/stage1/`
- `models/exp3_cxrtrek_seq/stage2/`
- `models/exp3_cxrtrek_seq/stage3/`

**Submit (parallel):**
```bash
sbatch experiments/exp3_cxrtrek_sequential/train_stage1.slurm
sbatch experiments/exp3_cxrtrek_sequential/train_stage2.slurm
sbatch experiments/exp3_cxrtrek_sequential/train_stage3.slurm
```

---

### Experiment 4: Curriculum Learning
**Directory:** `experiments/exp4_curriculum_learning/`  
**Scripts:** `train_stage1.slurm`, `train_stage2.slurm`, `train_stage3.slurm`  
**Description:** Progressive training where each stage continues from the previous checkpoint.  
**Approach:** Sequential training (Stage 1 → Stage 2 → Stage 3 with dependencies)  
**Dataset:** Uses stage-specific splits (same as Experiment 3)

**Outputs:**
- `models/exp4_curriculum/stage1/`
- `models/exp4_curriculum/stage2/` (continues from stage1)
- `models/exp4_curriculum/stage3/` (continues from stage2)

**Submit (sequential with dependencies):**
```bash
JOB1=$(sbatch experiments/exp4_curriculum_learning/train_stage1.slurm | grep -o '[0-9]*')
JOB2=$(sbatch --dependency=afterok:$JOB1 experiments/exp4_curriculum_learning/train_stage2.slurm | grep -o '[0-9]*')
JOB3=$(sbatch --dependency=afterok:$JOB2 experiments/exp4_curriculum_learning/train_stage3.slurm | grep -o '[0-9]*')
```

---

## Quick Start

### Submit All Experiments at Once:
```bash
cd "corrected 1-5 experiments"
./submit_all_experiments.sh
```

### Monitor Jobs:
```bash
squeue -u muhra.almahri
```

### View Logs:
```bash
# Experiment logs are in: logs/
tail -f logs/exp1_random_<JOB_ID>.out
tail -f logs/exp2_qwen_<JOB_ID>.out
# etc.
```

## Model Configuration

- **Base Model:** `Qwen/Qwen2-VL-7B-Instruct`
- **LoRA:** r=32, alpha=64, dropout=0.05
- **Training:** 3 epochs
- **Batch Size:** 1 (with gradient accumulation 16)
- **Learning Rate:** 5e-6
- **Max Length:** 128
- **Memory:** 128GB
- **Time Limit:** 72 hours per job
- **GPU:** 1 GPU per job

## Cluster Requirements

All scripts follow CIAI cluster guidelines:
- Partition: `cscc-gpu-p`
- QoS: `cscc-gpu-qos`
- Max 4 GPUs per job (we use 1)
- Max 4 jobs per user (consider this when submitting)

## Directory Structure

```
corrected 1-5 experiments/
├── datasets/
│   ├── kvasir_raw_6500_image_level_70_15_15/
│   │   ├── train.json
│   │   ├── val.json
│   │   ├── test.json
│   │   ├── split_metadata.json
│   │   └── kvasir_raw_qwen_reordered.json
│   ├── kvasir_stage_splits_stage1/
│   ├── kvasir_stage_splits_stage2/
│   └── kvasir_stage_splits_stage3/
├── experiments/
│   ├── exp1_random/
│   ├── exp2_qwen_reordered/
│   ├── exp3_cxrtrek_sequential/
│   └── exp4_curriculum_learning/
├── models/
│   ├── exp1_random_baseline/
│   ├── exp2_qwen_reordered/
│   ├── exp3_cxrtrek_seq/
│   └── exp4_curriculum/
├── logs/
├── scripts/
│   └── dataset_splits/
│       └── create_stage_splits.py
├── training/
└── submit_all_experiments.sh
```

## Notes

- All experiments use the **same corrected dataset split** (image-level, 70/15/15)
- Stage-specific splits were created automatically using the `stage` field in the dataset
- Experiment 3 stages can run in parallel (independent models)
- Experiment 4 stages must run sequentially (curriculum learning with checkpoint loading)

