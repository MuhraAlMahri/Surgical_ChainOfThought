# Multi-Head CoT SLURM Jobs Guide

This guide explains how to submit the new multi-head Chain-of-Thought (CoT) training and evaluation jobs through SLURM on MBZUAI HPC.

## ⚠️ Important: HPC Policy

**NEVER run jobs directly on login nodes!** All jobs must be submitted through SLURM.

## Available SLURM Scripts

### 1. `06_categorize_questions_new.slurm`
**Purpose**: Categorize questions into 3 clinical stages using the new `categorize_questions.py` script.

**Usage**:
```bash
sbatch slurm/06_categorize_questions_new.slurm \
    <kvasir_path> \
    <endovis_path> \
    <output_file> \
    [model_name]
```

**Example**:
```bash
sbatch slurm/06_categorize_questions_new.slurm \
    "datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json" \
    "datasets/EndoVis2018/raw/metadata/vqa_pairs.json" \
    "results/multihead_cot/question_categories.json" \
    "Qwen/Qwen2.5-7B-Instruct"
```

**Resources**:
- 1 GPU
- 64GB RAM
- 4 hours time limit
- 16 CPUs

### 2. `07_train_multihead_cot.slurm`
**Purpose**: Train multi-head CoT model using `train_multihead_cot.py`.

**Usage**:
```bash
sbatch slurm/07_train_multihead_cot.slurm \
    <model_type> \
    <dataset> \
    <base_checkpoint> \
    [learning_rate] \
    [epochs] \
    [batch_size]
```

**Arguments**:
- `model_type`: `qwen3vl`, `medgemma`, or `llava_med`
- `dataset`: `kvasir` or `endovis`
- `base_checkpoint`: Path to fine-tuned checkpoint (required)
- `learning_rate`: Default 2e-5 (3e-5 for MedGemma)
- `epochs`: Default 3 (5 for MedGemma)
- `batch_size`: Default 1 (2 for MedGemma)

**Examples**:

**Qwen3-VL on Kvasir**:
```bash
sbatch slurm/07_train_multihead_cot.slurm \
    qwen3vl \
    kvasir \
    checkpoints/qwen3vl_kvasir_finetuned \
    2e-5 \
    3 \
    1
```

**MedGemma on EndoVis**:
```bash
sbatch slurm/07_train_multihead_cot.slurm \
    medgemma \
    endovis \
    checkpoints/medgemma_endovis_finetuned \
    3e-5 \
    5 \
    2
```

**Resources**:
- 1 GPU
- 128GB RAM
- 24 hours time limit
- 16 CPUs

### 3. `08_evaluate_multihead_cot.slurm`
**Purpose**: Evaluate trained multi-head CoT model using `evaluate_multihead.py`.

**Usage**:
```bash
sbatch slurm/08_evaluate_multihead_cot.slurm \
    <checkpoint> \
    <model_type> \
    <dataset> \
    [test_data] \
    [baseline_results]
```

**Arguments**:
- `checkpoint`: Path to trained model checkpoint (required)
- `model_type`: `qwen3vl`, `medgemma`, or `llava_med`
- `dataset`: `kvasir` or `endovis`
- `test_data`: Path to test data JSON (optional, uses defaults)
- `baseline_results`: Path to baseline results JSON (optional)

**Example**:
```bash
sbatch slurm/08_evaluate_multihead_cot.slurm \
    results/multihead_cot/qwen3vl_kvasir_cot_20250101_120000/checkpoint_epoch_3.pt \
    qwen3vl \
    kvasir \
    datasets/Kvasir-VQA/raw/metadata/test_metadata.json \
    baseline_results.json
```

**Resources**:
- 1 GPU
- 128GB RAM
- 4 hours time limit
- 16 CPUs

## Complete Pipeline Submission

### Option 1: Automated Pipeline Script
Use the provided script to submit all jobs with proper dependencies:

```bash
bash slurm/submit_multihead_cot_pipeline.sh
```

**Note**: You need to update the checkpoint paths in `submit_multihead_cot_pipeline.sh` to match your actual checkpoint locations.

### Option 2: Manual Step-by-Step

**Step 1: Categorize Questions**
```bash
CATEG_JOB=$(sbatch --parsable slurm/06_categorize_questions_new.slurm \
    "datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json" \
    "datasets/EndoVis2018/raw/metadata/vqa_pairs.json" \
    "results/multihead_cot/question_categories.json")
echo "Categorization job: $CATEG_JOB"
```

**Step 2: Train Models (after categorization completes)**
```bash
# Qwen3-VL on Kvasir
TRAIN_JOB_1=$(sbatch --parsable --dependency=afterok:$CATEG_JOB \
    slurm/07_train_multihead_cot.slurm \
    qwen3vl kvasir checkpoints/qwen3vl_kvasir_finetuned)

# Qwen3-VL on EndoVis
TRAIN_JOB_2=$(sbatch --parsable --dependency=afterok:$CATEG_JOB \
    slurm/07_train_multihead_cot.slurm \
    qwen3vl endovis checkpoints/qwen3vl_endovis_finetuned)

# ... repeat for other models
```

**Step 3: Evaluate Models (after training completes)**
```bash
# Wait for all training jobs
TRAIN_DEPENDENCY="afterok:$TRAIN_JOB_1:$TRAIN_JOB_2"

# Evaluate Qwen3-VL on Kvasir
sbatch --dependency=$TRAIN_DEPENDENCY \
    slurm/08_evaluate_multihead_cot.slurm \
    results/multihead_cot/qwen3vl_kvasir_cot_*/checkpoint_epoch_3.pt \
    qwen3vl kvasir

# ... repeat for other models
```

## Monitoring Jobs

**Check job status**:
```bash
squeue -u $USER
```

**View specific job details**:
```bash
scontrol show job <job_id>
```

**View job output**:
```bash
tail -f slurm/logs/<job_name>_<job_id>.out
tail -f slurm/logs/<job_name>_<job_id>.err
```

**Cancel a job**:
```bash
scancel <job_id>
```

## Expected Outputs

### After Categorization
- `results/multihead_cot/question_categories.json` - Question category mappings

### After Training
- `results/multihead_cot/<model_type>_<dataset>_cot_<timestamp>/` - Training checkpoints
  - `checkpoint_epoch_1.pt`
  - `checkpoint_epoch_2.pt`
  - `checkpoint_epoch_3.pt`
  - ...

### After Evaluation
- `results/multihead_cot/evaluation/evaluation_results_<model_type>_<dataset>.json` - Evaluation results

## Troubleshooting

### Job Fails Immediately
- Check that input files exist
- Verify checkpoint paths are correct
- Check SLURM logs: `slurm/logs/<job_name>_<job_id>.err`

### Out of Memory Errors
- Reduce batch size
- Increase gradient accumulation steps
- Request more memory: `--mem=256G`

### Disk Quota Exceeded
- Scripts use `/tmp` for HuggingFace cache automatically
- Check available space: `df -h`
- Clean up old checkpoints if needed

### Job Stuck in Queue
- Check QOS limits: `sacctmgr show qos`
- Check partition availability: `sinfo`
- Consider reducing resource requests

## Configuration Notes

### Model-Specific Defaults

**Qwen3-VL**:
- Learning rate: 2e-5
- Batch size: 1
- Epochs: 3
- LoRA r: 8, alpha: 16

**MedGemma**:
- Learning rate: 3e-5
- Batch size: 2
- Epochs: 5
- LoRA r: 4, alpha: 16

**LLaVA-Med**:
- Learning rate: 2e-5
- Batch size: 1
- Epochs: 3
- LoRA r: 8, alpha: 16
- Freezes vision tower

## Quick Reference

| Script | Purpose | Time Limit | GPU | RAM |
|--------|---------|------------|-----|-----|
| `06_categorize_questions_new.slurm` | Question categorization | 4h | 1 | 64GB |
| `07_train_multihead_cot.slurm` | Training | 24h | 1 | 128GB |
| `08_evaluate_multihead_cot.slurm` | Evaluation | 4h | 1 | 128GB |

## Support

For issues or questions:
1. Check SLURM logs first
2. Verify file paths and permissions
3. Check HPC documentation: https://hpc.mbzuai.ac.ae/













