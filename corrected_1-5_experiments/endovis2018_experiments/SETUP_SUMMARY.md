# EndoVis2018 Experiments - Setup Summary

## ✅ Completed Setup

### 1. Data Preparation Script
- **File**: `scripts/prepare_endovis2018_for_vqa.py`
- **Purpose**: Converts EndoVis2018 segmentation data to VQA format (JSONL)
- **Features**:
  - Uses proper sequence-based split (no overlap)
  - Generates VQA questions from segmentation masks
  - Creates train/val/test JSONL files

### 2. Zero-Shot Evaluation
- **Script**: `slurm/zeroshot_endovis2018.slurm`
- **Purpose**: Evaluate base model without fine-tuning
- **Output**: `results/endovis2018_zeroshot.json`

### 3. Training Configuration
- **Config**: `configs/exp1_random.yaml`
- **Model**: Qwen/Qwen3-VL-8B-Instruct
- **Method**: QLoRA (r=4, alpha=8)
- **Training**: 5 epochs, batch size 1, grad_accum 16

### 4. Training Script
- **Script**: `slurm/train_exp1.slurm`
- **Purpose**: Train Exp1 (Random Baseline)
- **Output**: `models/exp1_random/`

### 5. Instruction Templates
- **File**: `../datasets/endovis2018_vqa/INSTRUCTIONS_PER_CATEGORY.txt`
- **Categories**: 4 categories
  1. INSTRUMENT_DETECTION (multi_label)
  2. ANATOMY_DETECTION (multi_label)
  3. INSTRUMENT_COUNT (single_choice)
  4. PROCEDURE_TYPE (single_choice)

### 6. Submission Script
- **File**: `SUBMIT_JOBS.sh`
- **Purpose**: Easy job submission for all experiments

## 📋 Next Steps

### Step 1: Prepare Data (Run on compute node via SLURM)
```bash
# Create a SLURM job for data preparation
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=prep_endovis_data
#SBATCH --time=2:00:00
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /l/users/muhra.almahri/Surgical_COT

# Organize images (if needed)
python scripts/organize_endovis2018.py

# Prepare VQA format
python scripts/prepare_endovis2018_for_vqa.py \
    --output_dir corrected_1-5_experiments/datasets/endovis2018_vqa \
    --image_dir datasets/EndoVis2018/raw/images \
    --use_proper_split
EOF
```

### Step 2: Review Instruction Templates
- **File**: `datasets/endovis2018_vqa/INSTRUCTIONS_PER_CATEGORY.txt`
- **Action**: Review and approve the templates
- **Note**: Templates follow the same ultra-condensed format as Kvasir-VQA

### Step 3: Run Zero-Shot Evaluation
```bash
cd corrected_1-5_experiments/endovis2018_experiments
sbatch slurm/zeroshot_endovis2018.slurm
```

### Step 4: Run Training (After zero-shot)
```bash
sbatch slurm/train_exp1.slurm
```

### Step 5: Instruction Fine-Tuning (After template approval)
- Will create instruction-based training configs
- Similar to Kvasir-VQA instruction fine-tuning

## 📊 Expected Dataset Structure

After data preparation:
```
datasets/endovis2018_vqa/
├── train.jsonl          # Training samples
├── validation.jsonl     # Validation samples
├── test.jsonl           # Test samples
└── INSTRUCTIONS_PER_CATEGORY.txt  # Instruction templates
```

## 🔧 Configuration Details

### Model
- **Base**: Qwen/Qwen3-VL-8B-Instruct
- **LoRA**: r=4, alpha=8
- **Target Modules**: q_proj, k_proj, v_proj, o_proj

### Training
- **Epochs**: 5
- **Batch Size**: 1 (per device)
- **Gradient Accumulation**: 16 (effective batch = 16)
- **Learning Rate**: 5.0e-5
- **Max Sequence Length**: 3072

### Data Split
- **Method**: Sequence-based (no overlap)
- **Train**: 9 sequences
- **Validation**: 2 sequences
- **Test**: 4 sequences

## ⚠️ Important Reminders

1. **MBZUAI HPC Policy**: All jobs MUST run via SLURM (compute nodes)
2. **Data Preparation**: Must be done before training/evaluation
3. **Instruction Templates**: Review and approve before instruction fine-tuning
4. **Sequence Split**: Uses proper sequence-based split (no overlap)

## 📝 Notes

- EndoVis2018 is a segmentation dataset, so VQA questions are generated from masks
- Question generation is basic - can be enhanced later
- Instruction templates follow Kvasir-VQA format for consistency
- All experiments follow the same methodology as Kvasir-VQA for fair comparison




















