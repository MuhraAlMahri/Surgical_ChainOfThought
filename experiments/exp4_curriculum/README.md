# Experiment 4: Curriculum Learning

## Overview
Progressive training through stages using Qwen3-VL-8B-Instruct.

## Results
- **Zero-Shot Accuracy**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned Accuracy**: 92.44% (8,305/8,984)

## Files
- **Configs**: 
  - `configs/exp4_stage1.yaml` - Stage 1 training config
  - `configs/exp4_stage2.yaml` - Stage 2 training config (continues from stage1)
  - `configs/exp4_stage3.yaml` - Stage 3 training config (continues from stage2)
- **Results**: 
  - `results/zeroshot.json` - Zero-shot evaluation results
  - `results/instruction_finetuned.json` - Instruction fine-tuned evaluation results

## How to Reproduce

### 1. Zero-Shot Evaluation
```bash
python scripts/evaluation/evaluate_zeroshot.py \
    --config experiments/exp4_curriculum/configs/exp4_stage1.yaml \
    --output experiments/exp4_curriculum/results/zeroshot.json
```

### 2. Training (Progressive - must train sequentially)
```bash
# Stage 1
python scripts/training/train_instruction_finetuning.py \
    --config experiments/exp4_curriculum/configs/exp4_stage1.yaml

# Stage 2 (loads checkpoint from stage1)
python scripts/training/train_instruction_finetuning.py \
    --config experiments/exp4_curriculum/configs/exp4_stage2.yaml \
    --resume_from <stage1_checkpoint>

# Stage 3 (loads checkpoint from stage2)
python scripts/training/train_instruction_finetuning.py \
    --config experiments/exp4_curriculum/configs/exp4_stage3.yaml \
    --resume_from <stage2_checkpoint>
```

### 3. Evaluation After Training
```bash
python scripts/evaluation/evaluate_exp4.py \
    --config experiments/exp4_curriculum/configs/exp4_stage3.yaml \
    --checkpoint <final_checkpoint> \
    --output experiments/exp4_curriculum/results/instruction_finetuned.json
```

