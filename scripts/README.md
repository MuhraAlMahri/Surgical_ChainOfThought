# Scripts Directory

This directory contains all reusable scripts for training, evaluation, and data preparation.

## Structure

```
scripts/
├── training/           # Training scripts
│   ├── train_instruction_finetuning.py  # Main training script for instruction fine-tuning
│   └── train_qlora_qwen3vl.py          # QLoRA training for Qwen3-VL
│
├── evaluation/        # Evaluation scripts
│   ├── evaluate_exp1.py                 # Evaluate Experiment 1
│   ├── evaluate_exp2.py                 # Evaluate Experiment 2
│   ├── evaluate_exp3.py                 # Evaluate Experiment 3
│   ├── evaluate_exp4.py                 # Evaluate Experiment 4
│   ├── evaluate_exp5.py                 # Evaluate Experiment 5
│   ├── evaluate_zeroshot.py             # Zero-shot evaluation
│   ├── evaluate_finetuned_llava.py     # LLaVA-Med evaluation
│   └── metrics_utils.py                # Metrics calculation utilities
│
└── data_preparation/  # Data preparation scripts
    ├── prepare_all_datasets_qlora.py   # Prepare datasets for QLoRA training
    └── create_stage_splits.py           # Create stage-based data splits
```

## Usage

### Training
```bash
python scripts/training/train_instruction_finetuning.py --config <config_file>
```

### Evaluation
```bash
python scripts/evaluation/evaluate_exp<1-5>.py --config <config_file> --checkpoint <checkpoint_path>
```

### Data Preparation
```bash
python scripts/data_preparation/prepare_all_datasets_qlora.py
```

