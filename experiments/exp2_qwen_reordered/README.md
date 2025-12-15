# Experiment 2: Qwen Clinical Reordering

## Overview
Questions ordered by Qwen clinical stages (Stage1→Stage2→Stage3) using Qwen3-VL-8B-Instruct.

## Results
- **Zero-Shot Accuracy**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned Accuracy**: 92.76% (8,334/8,984)

## Files
- **Config**: `configs/exp2_qwen_reordered.yaml`
- **Results**: 
  - `results/zeroshot.json` - Zero-shot evaluation results
  - `results/instruction_finetuned.json` - Instruction fine-tuned evaluation results

## How to Reproduce

### 1. Zero-Shot Evaluation
```bash
python scripts/evaluation/evaluate_zeroshot.py \
    --config experiments/exp2_qwen_reordered/configs/exp2_qwen_reordered.yaml \
    --output experiments/exp2_qwen_reordered/results/zeroshot.json
```

### 2. Training
```bash
python scripts/training/train_instruction_finetuning.py \
    --config experiments/exp2_qwen_reordered/configs/exp2_qwen_reordered.yaml
```

### 3. Evaluation After Training
```bash
python scripts/evaluation/evaluate_exp2.py \
    --config experiments/exp2_qwen_reordered/configs/exp2_qwen_reordered.yaml \
    --checkpoint <path_to_checkpoint> \
    --output experiments/exp2_qwen_reordered/results/instruction_finetuned.json
```

