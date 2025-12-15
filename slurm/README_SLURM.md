# SLURM Job Scripts for Multi-Head Temporal CoT

This directory contains SLURM job scripts for running the multi-head temporal CoT system on MBZUAI HPC cluster.

## ⚠️ Important: HPC Policy

**NEVER run training or heavy computation on login nodes!** All jobs must be submitted through SLURM.

## Job Scripts

### 1. Question Categorization (`01_categorize_questions.slurm`)

Categorizes questions into clinical stages using LLM.

**Usage:**
```bash
sbatch slurm/01_categorize_questions.slurm [dataset] [input_file]
```

**Examples:**
```bash
# Kvasir-VQA
sbatch slurm/01_categorize_questions.slurm kvasir datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json

# EndoVis 2018
sbatch slurm/01_categorize_questions.slurm endovis datasets/EndoVis2018/train.json
```

**Resources:**
- GPU: 1 (for LLM inference)
- Memory: 64GB
- Time: 4 hours
- CPUs: 16

---

### 2. Temporal Structure Creation (`02_create_temporal_structure.slurm`)

Creates temporal structure for video sequences (EndoVis only).

**Usage:**
```bash
sbatch slurm/02_create_temporal_structure.slurm [sequence_dir] [qa_file] [sequence_id]
```

**Example:**
```bash
sbatch slurm/02_create_temporal_structure.slurm \
    datasets/EndoVis2018/data/images \
    datasets/EndoVis2018/train.json \
    seq_1
```

**Resources:**
- GPU: None (CPU only)
- Memory: 32GB
- Time: 2 hours
- CPUs: 8

---

### 3. Unified Training (`03_train_unified.slurm`)

Trains the multi-head model with unified training strategy.

**Usage:**
```bash
sbatch slurm/03_train_unified.slurm [dataset] [base_model] [num_epochs] [batch_size] [learning_rate] [grad_accum]
```

**Examples:**
```bash
# Kvasir-VQA with default settings
sbatch slurm/03_train_unified.slurm kvasir

# EndoVis with custom settings
sbatch slurm/03_train_unified.slurm endovis Qwen/Qwen2-VL-2B-Instruct 10 4 2e-5 4
```

**Resources:**
- GPU: 1
- Memory: 128GB
- Time: 24 hours
- CPUs: 16

**Parameters:**
- `dataset`: kvasir or endovis
- `base_model`: Model name (default: Qwen/Qwen2-VL-2B-Instruct)
- `num_epochs`: Training epochs (default: 10)
- `batch_size`: Batch size (default: 4)
- `learning_rate`: Learning rate (default: 2e-5)
- `grad_accum`: Gradient accumulation steps (default: 4)

---

### 4. Sequential Training (`04_train_sequential.slurm`)

Trains the multi-head model with sequential curriculum learning.

**Usage:**
```bash
sbatch slurm/04_train_sequential.slurm [dataset] [base_model] [epochs_per_stage] [batch_size] [learning_rate] [grad_accum]
```

**Example:**
```bash
sbatch slurm/04_train_sequential.slurm kvasir Qwen/Qwen2-VL-2B-Instruct 5 4 2e-5 4
```

**Resources:**
- GPU: 1
- Memory: 128GB
- Time: 36 hours (longer due to sequential stages)
- CPUs: 16

**Parameters:**
- `dataset`: kvasir or endovis
- `base_model`: Model name (default: Qwen/Qwen2-VL-2B-Instruct)
- `epochs_per_stage`: Epochs per stage (default: 5, total: 15 epochs)
- `batch_size`: Batch size (default: 4)
- `learning_rate`: Learning rate (default: 2e-5)
- `grad_accum`: Gradient accumulation steps (default: 4)

---

### 5. Evaluation (`05_evaluate.slurm`)

Evaluates trained model on test set.

**Usage:**
```bash
sbatch slurm/05_evaluate.slurm [model_path] [test_data] [baseline_results]
```

**Examples:**
```bash
# Basic evaluation
sbatch slurm/05_evaluate.slurm \
    checkpoints/kvasir_unified/best_model.pt \
    data/categorized/test_categorized.json

# With baseline comparison
sbatch slurm/05_evaluate.slurm \
    checkpoints/kvasir_unified/best_model.pt \
    data/categorized/test_categorized.json \
    results/baseline_results.json
```

**Resources:**
- GPU: 1
- Memory: 64GB
- Time: 4 hours
- CPUs: 8

---

### 6. Complete Pipeline (`submit_all.sh`)

Submits all jobs in sequence with proper dependencies.

**Usage:**
```bash
sbatch slurm/submit_all.sh [dataset] [base_model]
```

**Example:**
```bash
sbatch slurm/submit_all.sh kvasir Qwen/Qwen2-VL-2B-Instruct
```

This script:
1. Submits question categorization
2. Submits temporal structure creation (EndoVis only)
3. Submits unified training (depends on categorization)
4. Submits sequential training (depends on categorization)
5. Submits evaluation for unified model (depends on unified training)
6. Submits evaluation for sequential model (depends on sequential training)

## Quick Start

### For Kvasir-VQA:

```bash
cd /l/users/muhra.almahri/Surgical_COT

# Option 1: Run complete pipeline
sbatch slurm/submit_all.sh kvasir Qwen/Qwen2-VL-2B-Instruct

# Option 2: Run steps individually
# Step 1: Categorize questions
sbatch slurm/01_categorize_questions.slurm kvasir

# Step 2: Train unified model (after categorization completes)
sbatch --dependency=afterok:JOB_ID slurm/03_train_unified.slurm kvasir

# Step 3: Train sequential model
sbatch --dependency=afterok:JOB_ID slurm/04_train_sequential.slurm kvasir

# Step 4: Evaluate
sbatch --dependency=afterok:JOB_ID slurm/05_evaluate.slurm
```

### For EndoVis 2018:

```bash
cd /l/users/muhra.almahri/Surgical_COT

# Option 1: Run complete pipeline
sbatch slurm/submit_all.sh endovis Qwen/Qwen2-VL-2B-Instruct

# Option 2: Run steps individually
# Step 1: Categorize questions
JOB1=$(sbatch --parsable slurm/01_categorize_questions.slurm endovis)

# Step 2: Create temporal structure
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/02_create_temporal_structure.slurm)

# Step 3: Train unified model
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/03_train_unified.slurm endovis)

# Step 4: Train sequential model
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/04_train_sequential.slurm endovis)

# Step 5: Evaluate
sbatch --dependency=afterok:$JOB3 slurm/05_evaluate.slurm
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job output
tail -f slurm/logs/train_unified_JOBID.out

# View job errors
tail -f slurm/logs/train_unified_JOBID.err

# Cancel a job
scancel JOB_ID
```

## Resource Requirements

| Job | GPU | Memory | CPUs | Time |
|-----|-----|--------|------|------|
| Categorization | 1 | 64GB | 16 | 4h |
| Temporal Structure | 0 | 32GB | 8 | 2h |
| Unified Training | 1 | 128GB | 16 | 24h |
| Sequential Training | 1 | 128GB | 16 | 36h |
| Evaluation | 1 | 64GB | 8 | 4h |

## Troubleshooting

### Job fails immediately
- Check if input files exist
- Verify paths are correct
- Check SLURM logs: `slurm/logs/*.err`

### Out of memory
- Reduce batch size
- Increase gradient accumulation steps
- Use smaller model (2B instead of 7B)

### Job times out
- Increase time limit in SLURM script
- Reduce number of epochs
- Use smaller dataset subset for testing

### Module not found
- Check conda environment: `source ~/miniconda3/bin/activate base`
- Install missing packages: `pip install -r requirements.txt`

## Notes

- All scripts use `cscc-gpu-p` partition and `cscc-gpu-qos` QOS
- Logs are saved to `slurm/logs/`
- Checkpoints are saved to `checkpoints/`
- Results are saved to `results/`
- Make sure to run categorization before training
- For EndoVis, create temporal structure before training














