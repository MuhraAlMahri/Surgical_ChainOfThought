# Zero-Shot and Instruction Fine-Tuning Locations for KAVISR and Endovis with Qwen3

This document provides the exact locations of zero-shot evaluation and instruction fine-tuning configurations for **KAVISR (Kvasir)** and **Endovis** datasets using **Qwen3-VL-8B-Instruct**.

---

## 📍 Quick Reference

### **KAVISR (Kvasir-VQA) - Qwen3**

#### Zero-Shot Evaluation
- **Script**: `corrected_1-5_experiments/qlora_experiments/slurm/zeroshot_batch1.slurm`
- **Evaluation Script**: `corrected_1-5_experiments/scripts/evaluation/evaluate_zeroshot.py`
- **Results**: `corrected_1-5_experiments/qlora_experiments/results/exp1_zeroshot.json` (and exp2, exp3, exp4, exp5)

#### Instruction Fine-Tuning
- **Training Script**: `corrected_1-5_experiments/qlora_experiments/train_instruction_finetuning.py`
- **Config Files**: `corrected_1-5_experiments/qlora_experiments/configs/exp1_random.yaml` (and exp2, exp3, exp4, exp5)
- **SLURM Scripts**: `corrected_1-5_experiments/qlora_experiments/slurm/train_*.slurm`

### **Endovis (EndoVis2018) - Qwen3**

#### Zero-Shot Evaluation
- **Script**: `corrected_1-5_experiments/endovis2018_experiments/slurm/zeroshot_endovis2018.slurm`
- **Evaluation Script**: `corrected_1-5_experiments/scripts/evaluation/evaluate_zeroshot.py`
- **Results**: `corrected_1-5_experiments/endovis2018_experiments/results/endovis2018_zeroshot.json`

#### Instruction Fine-Tuning
- **Training Script**: `corrected_1-5_experiments/qlora_experiments/train_instruction_finetuning.py`
- **Config Files**: `corrected_1-5_experiments/endovis2018_experiments/configs/exp1_instruction_finetuning.yaml` (and exp2, exp3, exp4, exp5)
- **SLURM Scripts**: `corrected_1-5_experiments/endovis2018_experiments/slurm/train_exp1_instruction_finetuning.slurm`

---

## 🔍 Detailed Locations

### 1. Zero-Shot Evaluation

#### **KAVISR (Kvasir-VQA)**

**Main Zero-Shot Scripts:**
```
corrected_1-5_experiments/qlora_experiments/slurm/
├── zeroshot_all_experiments.slurm    # All experiments (Exp1-5)
├── zeroshot_batch1.slurm              # Exp1-4 batch
└── zeroshot_batch2.slurm              # Exp5 batch
```

**Evaluation Script:**
```
corrected_1-5_experiments/scripts/evaluation/evaluate_zeroshot.py
```

**Key Configuration:**
- Base Model: `Qwen/Qwen3-VL-8B-Instruct`
- Test Data: `corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl`
- Image Directory: `datasets/Kvasir-VQA/raw/images`
- Results Directory: `corrected_1-5_experiments/qlora_experiments/results/`

**Usage Example:**
```bash
sbatch corrected_1-5_experiments/qlora_experiments/slurm/zeroshot_batch1.slurm
```

#### **Endovis (EndoVis2018)**

**Zero-Shot Script:**
```
corrected_1-5_experiments/endovis2018_experiments/slurm/zeroshot_endovis2018.slurm
```

**Key Configuration:**
- Base Model: `Qwen/Qwen3-VL-8B-Instruct`
- Test Data: `corrected_1-5_experiments/datasets/endovis2018_vqa/test.jsonl`
- Image Directory: `datasets/EndoVis2018/raw/images`
- Results: `corrected_1-5_experiments/endovis2018_experiments/results/endovis2018_zeroshot.json`

**Usage Example:**
```bash
sbatch corrected_1-5_experiments/endovis2018_experiments/slurm/zeroshot_endovis2018.slurm
```

---

### 2. Instruction Fine-Tuning

#### **KAVISR (Kvasir-VQA)**

**Training Script:**
```
corrected_1-5_experiments/qlora_experiments/train_instruction_finetuning.py
```

**Configuration Files:**
```
corrected_1-5_experiments/qlora_experiments/configs/
├── exp1_random.yaml              # Exp1: Random Baseline
├── exp2_qwen_reordered.yaml       # Exp2: Qwen Reordered
├── exp3_stage1.yaml               # Exp3: Sequential Stage 1
├── exp3_stage2.yaml               # Exp3: Sequential Stage 2
├── exp3_stage3.yaml               # Exp3: Sequential Stage 3
├── exp4_stage1.yaml               # Exp4: Curriculum Stage 1
├── exp4_stage2.yaml               # Exp4: Curriculum Stage 2
├── exp4_stage3.yaml               # Exp4: Curriculum Stage 3
└── exp5_sequential_cot.yaml       # Exp5: Sequential CoT
```

**Key Configuration (Exp1 Example):**
```yaml
model_name: Qwen/Qwen3-VL-8B-Instruct
output_dir: /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/models/exp1_random
lora:
  r: 4
  alpha: 8
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
train:
  train_bs: 1
  grad_accum: 16
  lr: 5.0e-5
  epochs: 5
  max_seq_len: 3072
data:
  train_jsonl: /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/train.jsonl
  val_jsonl: /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/val.jsonl
  image_root: /l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images
```

**SLURM Training Scripts:**
```
corrected_1-5_experiments/qlora_experiments/slurm/
├── train_exp4_all_stages.slurm
├── train_exp5.slurm
└── train_parallel_experiments.slurm
```

**Usage Example:**
```bash
# Train Exp1
python3 corrected_1-5_experiments/qlora_experiments/train_instruction_finetuning.py \
    corrected_1-5_experiments/qlora_experiments/configs/exp1_random.yaml
```

#### **Endovis (EndoVis2018)**

**Training Script:**
```
corrected_1-5_experiments/qlora_experiments/train_instruction_finetuning.py
```

**Configuration Files:**
```
corrected_1-5_experiments/endovis2018_experiments/configs/
├── exp1_instruction_finetuning.yaml      # Exp1: Random Baseline (R1 split)
├── exp2_instruction_finetuning.yaml      # Exp2: Qwen Reordered
├── exp3_instruction_finetuning.yaml      # Exp3: Sequential
├── exp4_instruction_finetuning.yaml      # Exp4: Curriculum
├── exp5_instruction_finetuning.yaml      # Exp5: Sequential CoT
└── instruction_finetuning.yaml           # General instruction fine-tuning
```

**Key Configuration (Exp1 Example):**
```yaml
model_name: Qwen/Qwen3-VL-8B-Instruct
output_dir: /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/endovis2018_experiments/models/exp1_random_instruction_r1
lora:
  r: 4
  alpha: 8
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
train:
  train_bs: 1
  grad_accum: 16
  lr: 5.0e-5
  epochs: 5
  max_seq_len: 3072
data:
  train_jsonl: /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/endovis18_surgery_r1_split/train.jsonl
  val_jsonl: /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/endovis18_surgery_r1_split/val.jsonl
  image_root: /l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images
  instruction_template: /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/endovis2018_vqa/INSTRUCTIONS_PER_CATEGORY.txt
```

**SLURM Training Scripts:**
```
corrected_1-5_experiments/endovis2018_experiments/slurm/
├── train_exp1_instruction_finetuning.slurm
├── train_all_instruction_finetuning.slurm
└── train_instruction_finetuning.slurm
```

**Usage Example:**
```bash
# Train Exp1 via SLURM
sbatch corrected_1-5_experiments/endovis2018_experiments/slurm/train_exp1_instruction_finetuning.slurm

# Or directly with Python
python3 corrected_1-5_experiments/qlora_experiments/train_instruction_finetuning.py \
    corrected_1-5_experiments/endovis2018_experiments/configs/exp1_instruction_finetuning.yaml
```

---

## 📊 Model Outputs

### **KAVISR Models:**
```
corrected_1-5_experiments/qlora_experiments/models/
├── exp1_random/
├── exp2_qwen_reordered/
├── exp3_cxrtrek/
│   ├── stage1/
│   ├── stage2/
│   └── stage3/
├── exp4_curriculum/
│   ├── stage1/
│   ├── stage2/
│   └── stage3/
└── exp5_sequential_cot/
```

### **Endovis Models:**
```
corrected_1-5_experiments/endovis2018_experiments/models/
├── exp1_random_instruction_r1/
├── exp2_instruction/
├── exp3_sequential_instruction/
├── exp4_curriculum_instruction/
└── exp5_sequential_cot_instruction/
```

---

## 🔧 Common Parameters

### **Zero-Shot Evaluation:**
- Base Model: `Qwen/Qwen3-VL-8B-Instruct`
- Flag: `--use_instruction` (uses instruction templates)
- Output Format: JSON with accuracy metrics

### **Instruction Fine-Tuning:**
- Base Model: `Qwen/Qwen3-VL-8B-Instruct`
- Method: QLoRA (4-bit quantization)
- LoRA Config: r=4, alpha=8, dropout=0.05
- Target Modules: `[q_proj, k_proj, v_proj, o_proj]` (attention only)
- Training: 5 epochs, batch size 1, gradient accumulation 16
- Learning Rate: 5.0e-5
- Max Sequence Length: 3072

---

## 📝 Instruction Templates

### **KAVISR:**
- Location: Dataset-specific instruction files in each experiment's data directory
- Format: Category-based instruction templates

### **Endovis:**
- Location: `corrected_1-5_experiments/datasets/endovis2018_vqa/INSTRUCTIONS_PER_CATEGORY.txt`
- Categories:
  1. INSTRUMENT_DETECTION (multi_label)
  2. ANATOMY_DETECTION (multi_label)
  3. INSTRUMENT_COUNT (single_choice)
  4. PROCEDURE_TYPE (single_choice)

---

## 🚀 Quick Start Commands

### **Run Zero-Shot for KAVISR:**
```bash
cd /l/users/muhra.almahri/Surgical_COT
sbatch corrected_1-5_experiments/qlora_experiments/slurm/zeroshot_batch1.slurm
```

### **Run Zero-Shot for Endovis:**
```bash
cd /l/users/muhra.almahri/Surgical_COT
sbatch corrected_1-5_experiments/endovis2018_experiments/slurm/zeroshot_endovis2018.slurm
```

### **Train Instruction Fine-Tuning for KAVISR Exp1:**
```bash
cd /l/users/muhra.almahri/Surgical_COT
python3 corrected_1-5_experiments/qlora_experiments/train_instruction_finetuning.py \
    corrected_1-5_experiments/qlora_experiments/configs/exp1_random.yaml
```

### **Train Instruction Fine-Tuning for Endovis Exp1:**
```bash
cd /l/users/muhra.almahri/Surgical_COT
sbatch corrected_1-5_experiments/endovis2018_experiments/slurm/train_exp1_instruction_finetuning.slurm
```

---

## 📚 Additional Resources

- **Architecture Documentation**: `ARCHITECTURE_COMPONENTS.md`
- **KAVISR Experiments Table**: `corrected_1-5_experiments/KVAISR_VQA_EXPERIMENTS_TABLE.md`
- **Endovis Setup Summary**: `corrected_1-5_experiments/endovis2018_experiments/SETUP_SUMMARY.md`
- **Evaluation Scripts**: `corrected_1-5_experiments/scripts/evaluation/`

---

*Last Updated: Based on current codebase structure*







