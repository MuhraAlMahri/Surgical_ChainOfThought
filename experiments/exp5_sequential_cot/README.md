# Experiment 5: Sequential Chain-of-Thought

## Overview
Sequential CoT training strategy using Qwen3-VL-8B-Instruct.

## Results
- **Zero-Shot Accuracy**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned Accuracy**: 92.62% (8,321/8,984)

## Files
- **Config**: `configs/exp5_sequential_cot.yaml`
- **Results**: 
  - `results/zeroshot.json` - Zero-shot evaluation results
  - `results/instruction_finetuned.json` - Instruction fine-tuned evaluation results

## How to Reproduce

### 1. Zero-Shot Evaluation
```bash
python scripts/evaluation/evaluate_zeroshot.py \
    --config experiments/exp5_sequential_cot/configs/exp5_sequential_cot.yaml \
    --output experiments/exp5_sequential_cot/results/zeroshot.json
```

### 2. Training
```bash
python scripts/training/train_instruction_finetuning.py \
    --config experiments/exp5_sequential_cot/configs/exp5_sequential_cot.yaml
```

### 3. Evaluation After Training
```bash
python scripts/evaluation/evaluate_exp5.py \
    --config experiments/exp5_sequential_cot/configs/exp5_sequential_cot.yaml \
    --checkpoint <path_to_checkpoint> \
    --output experiments/exp5_sequential_cot/results/instruction_finetuned.json
```

