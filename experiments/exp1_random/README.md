# Experiment 1: Random Baseline

## Overview
Standard training with random question ordering using Qwen3-VL-8B-Instruct.

## Results
- **Zero-Shot Accuracy**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned Accuracy**: 92.79% (8,336/8,984)

## Files
- **Config**: `configs/exp1_random.yaml`
- **Results**: 
  - `results/exp1_zeroshot.json` - Zero-shot evaluation results
  - `results/instruction_finetuned.json` - Instruction fine-tuned evaluation results

## How to Reproduce

### 1. Zero-Shot Evaluation
```bash
python scripts/evaluation/evaluate_zeroshot.py \
    --config experiments/exp1_random/configs/exp1_random.yaml \
    --output experiments/exp1_random/results/exp1_zeroshot.json
```

### 2. Training
```bash
python scripts/training/train_instruction_finetuning.py \
    --config experiments/exp1_random/configs/exp1_random.yaml
```

### 3. Evaluation After Training
```bash
python scripts/evaluation/evaluate_exp1.py \
    --config experiments/exp1_random/configs/exp1_random.yaml \
    --checkpoint <path_to_checkpoint> \
    --output experiments/exp1_random/results/instruction_finetuned.json
```

