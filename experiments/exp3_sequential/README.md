# Experiment 3: CXRTrek Sequential

## Overview
Three specialized models, one per clinical stage, using Qwen3-VL-8B-Instruct.

## Results
- **Zero-Shot Accuracy**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned Accuracy**: 92.23% (8,290/8,984)

## Files
- **Configs**: 
  - `configs/exp3_stage1.yaml` - Stage 1 model config
  - `configs/exp3_stage2.yaml` - Stage 2 model config
  - `configs/exp3_stage3.yaml` - Stage 3 model config
- **Results**: 
  - `results/zeroshot.json` - Zero-shot evaluation results
  - `results/instruction_finetuned.json` - Instruction fine-tuned evaluation results

## How to Reproduce

### 1. Zero-Shot Evaluation
```bash
python scripts/evaluation/evaluate_zeroshot.py \
    --config experiments/exp3_sequential/configs/exp3_stage1.yaml \
    --output experiments/exp3_sequential/results/zeroshot.json
```

### 2. Training (Train each stage separately)
```bash
# Stage 1
python scripts/training/train_instruction_finetuning.py \
    --config experiments/exp3_sequential/configs/exp3_stage1.yaml

# Stage 2
python scripts/training/train_instruction_finetuning.py \
    --config experiments/exp3_sequential/configs/exp3_stage2.yaml

# Stage 3
python scripts/training/train_instruction_finetuning.py \
    --config experiments/exp3_sequential/configs/exp3_stage3.yaml
```

### 3. Evaluation After Training
```bash
python scripts/evaluation/evaluate_exp3.py \
    --stage1_config experiments/exp3_sequential/configs/exp3_stage1.yaml \
    --stage2_config experiments/exp3_sequential/configs/exp3_stage2.yaml \
    --stage3_config experiments/exp3_sequential/configs/exp3_stage3.yaml \
    --checkpoint_dir <path_to_checkpoints> \
    --output experiments/exp3_sequential/results/instruction_finetuned.json
```

