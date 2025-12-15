# EndoVis2018 Experiments

This directory contains experiments for EndoVis2018 dataset, following the same methodology as Kvasir-VQA experiments.

## 📋 Overview

- **Dataset**: EndoVis2018 (surgical scene segmentation)
- **Model**: Qwen/Qwen3-VL-8B-Instruct
- **Method**: QLoRA fine-tuning
- **Experiments**: Same as Kvasir-VQA (Exp1: Random Baseline, etc.)

## 🚀 Quick Start

### Step 1: Prepare Data

First, ensure images are uploaded and organized:
```bash
cd /l/users/muhra.almahri/Surgical_COT

# Organize images (if not done)
python scripts/organize_endovis2018.py

# Prepare VQA format
python scripts/prepare_endovis2018_for_vqa.py \
    --output_dir corrected_1-5_experiments/datasets/endovis2018_vqa \
    --image_dir datasets/EndoVis2018/raw/images \
    --use_proper_split
```

This creates:
- `train.jsonl` - Training samples
- `validation.jsonl` - Validation samples
- `test.jsonl` - Test samples

### Step 2: Run Zero-Shot + Training (Mega Job)

Run both zero-shot evaluation and training in parallel:
```bash
cd corrected_1-5_experiments/endovis2018_experiments

# Submit mega job (runs both in parallel on 3 GPUs)
./SUBMIT_MEGA_JOB.sh

# Or submit directly:
sbatch slurm/mega_job_zeroshot_and_training.slurm

# Monitor
squeue -u $USER
tail -f slurm/logs/mega_job_*.out
```

**Mega Job Details:**
- **GPU 0**: Zero-shot evaluation (baseline)
- **GPU 1**: Training Exp1 (Random Baseline)
- **GPU 2**: Reserved for future use

Results:
- Zero-shot: `results/endovis2018_zeroshot.json`
- Training: `models/exp1_random/`

### Step 4: Evaluation

After training, evaluate the fine-tuned model:
```bash
# (Evaluation scripts will be added after training)
```

## 📁 Directory Structure

```
endovis2018_experiments/
├── README.md                    # This file
├── configs/                     # Training configurations
│   └── exp1_random.yaml        # Exp1 config
├── slurm/                       # SLURM job scripts
│   ├── zeroshot_endovis2018.slurm
│   ├── train_exp1.slurm
│   └── logs/                    # Job logs
├── models/                      # Trained models
│   └── exp1_random/            # Exp1 model
└── results/                     # Evaluation results
    └── endovis2018_zeroshot.json
```

## ⚠️ Important: MBZUAI HPC Policy

**ALL jobs MUST run on compute nodes via SLURM.**
- ❌ **DO NOT** run training/evaluation on login nodes
- ✅ **DO** submit all jobs using `sbatch`
- ✅ **DO** use the provided SLURM scripts

## 📊 Experiments

### Experiment 1: Random Baseline
- **Config**: `configs/exp1_random.yaml`
- **Script**: `slurm/train_exp1.slurm`
- **Description**: Random question ordering baseline

### Future Experiments
- Exp2: Qwen-reordered (similar to Kvasir-VQA)
- Exp3: Sequential CoT
- Exp4: Curriculum Learning
- Exp5: Sequential CoT with instructions

## 🔧 Configuration

All configs follow the same structure as Kvasir-VQA:
- **Model**: Qwen/Qwen3-VL-8B-Instruct
- **LoRA**: r=4, alpha=8, attention modules only
- **Training**: 5 epochs, batch size 1, grad_accum 16
- **Learning Rate**: 5.0e-5

## 📝 Data Format

Each JSONL line contains:
```json
{
  "image_id": "endovis_seq1_frame000",
  "image_filename": "endovis_seq1_frame000.png",
  "question": "What surgical instruments are visible?",
  "answer": "instrument-shaft; instrument-clasper",
  "question_type": "instrument_detection",
  "category": "INSTRUMENT_DETECTION",
  "sequence": "1",
  "frame": "000",
  "dataset": "EndoVis2018"
}
```

## 📈 Expected Results

- **Zero-shot**: Baseline performance (to be measured)
- **Exp1**: Improved performance after fine-tuning
- **Comparison**: Will compare with Kvasir-VQA results

## 🔗 Related Files

- Data preparation: `scripts/prepare_endovis2018_for_vqa.py`
- Training script: `qlora_experiments/train_qlora_qwen3vl.py`
- Evaluation script: `scripts/evaluation/evaluate_zeroshot.py`

## 📞 Support

For issues or questions, check:
- Kvasir-VQA experiments: `corrected_1-5_experiments/qlora_experiments/`
- Dataset info: `datasets/EndoVis2018/README.md`


