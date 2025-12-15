# Kvasir-VQA Experiments 1-5: Results Comparison Table

## Overview

This table compares the performance of experiments 1-5 across three evaluation approaches:
- **Zero-Shot**: Base model without any fine-tuning
- **Fine-Tuning**: Standard fine-tuning (not performed in these experiments - was replaced by instruction fine-tuning)
- **Instruction Fine-Tuning**: Fine-tuning with instruction templates (all qlora experiments)

---

## Results Table

| Experiment | Description | Zero-Shot Accuracy | Fine-Tuning Accuracy | Instruction Fine-Tuning Accuracy |
|------------|-------------|-------------------|----------------------|--------------------------------|
| **Exp1** | Random Baseline | 53.48% | **35.5%** | **92.79%** |
| **Exp2** | Qwen Clinical Reordering | 53.48% | **44.52%** | **92.76%** |
| **Exp3** | CXRTrek Sequential (3 specialized models) | 53.48% | **42.09%** | **92.23%** |
| **Exp4** | Curriculum Learning (progressive training) | 53.48% | **21.0%** | **92.44%** |
| **Exp5** | Sequential Chain-of-Thought | 53.48% | **36.5%** | **92.62%** |

*Note: Fine-tuning results (without instruction templates) show the performance when models were trained but evaluated without using instruction templates at inference time. These results demonstrate the importance of instruction fine-tuning, which improved accuracy by ~50-70 percentage points.

---

## Detailed Breakdown

### Zero-Shot Results (Baseline)
- **All Experiments**: 53.48% (4,805/8,984 correct)
- **Model**: Qwen3-VL-8B-Instruct (base model, no training)
- **Purpose**: Baseline to measure improvement from fine-tuning

### Instruction Fine-Tuning Results

#### Experiment 1: Random Baseline
- **Accuracy**: 92.79% (8,336/8,984 correct)
- **Approach**: Standard training with random question ordering
- **Training**: QLoRA (r=4, alpha=8), 5 epochs
- **Dataset**: Random order, all stages mixed

#### Experiment 2: Qwen Clinical Reordering
- **Accuracy**: 92.76% (8,334/8,984 correct)
- **Approach**: Training with Qwen-reordered questions (clinical flow: Stage1→Stage2→Stage3)
- **Training**: QLoRA (r=4, alpha=8), 5 epochs
- **Dataset**: Clinically ordered (Stage1: 35%, Stage2: 64%, Stage3: 0.1%)

#### Experiment 3: CXRTrek Sequential
- **Overall Accuracy**: 92.23% (8,286/8,984 correct)
- **Approach**: Three specialized models (one per clinical stage)
- **Training**: QLoRA (r=4, alpha=8), 5 epochs per stage
- **Per-Stage Performance**:
  - Stage 1: 91.18% (2,986/3,275)
  - Stage 2: 92.88% (5,297/5,703)
  - Stage 3: 50.00% (3/6) - Very few samples

#### Experiment 4: Curriculum Learning
- **Accuracy**: 92.44% (8,305/8,984 correct)
- **Approach**: Progressive training (Stage1 → Stage2 → Stage3 with checkpoint loading)
- **Training**: QLoRA (r=4, alpha=8), 5 epochs per stage
- **Note**: Each stage continues from previous checkpoint

#### Experiment 5: Sequential Chain-of-Thought
- **Accuracy**: 92.62% (8,321/8,984 correct)
- **Approach**: Sequential CoT inference (cascading predictions across stages)
- **Training**: QLoRA (r=4, alpha=8), 5 epochs
- **Dataset**: Uses Qwen-reordered dataset (same as Exp2)

---

## Key Observations

1. **Zero-Shot Baseline**: All experiments share the same baseline (53.48%), confirming consistent evaluation setup.

2. **Instruction Fine-Tuning Impact**: All instruction fine-tuned models achieve ~92% accuracy, showing a **+39 percentage point improvement** over zero-shot.

3. **Experiment Comparison**: 
   - **Best**: Exp1 (Random Baseline) - 92.79%
   - **Smallest gap**: Only 0.56% difference between best and worst
   - All experiments perform similarly, suggesting instruction fine-tuning is the dominant factor

4. **Standard Fine-Tuning**: Results when models were trained with instruction fine-tuning but evaluated without using instruction templates at inference:
   - Exp1: 35.5% (vs 92.79% with instructions) - **+57.29 pts improvement**
   - Exp2: 44.52% (vs 92.76% with instructions) - **+48.24 pts improvement**
   - Exp3: 42.09% (vs 92.23% with instructions) - **+50.14 pts improvement**
   - Exp4: 21.0% (vs 92.44% with instructions) - **+71.44 pts improvement**
   - Exp5: 36.5% (vs 92.62% with instructions) - **+56.12 pts improvement**
   - **Key Finding**: Using instruction templates at inference is critical, even when models were trained with instructions

---

## Technical Details

### Model Configuration (All Instruction Fine-Tuning)
- **Base Model**: Qwen/Qwen3-VL-8B-Instruct
- **Method**: QLoRA (4-bit quantization + LoRA)
- **LoRA Config**: r=4, alpha=8, dropout=0.05
- **Target Modules**: q_proj, k_proj, v_proj, o_proj (attention only)
- **Trainable Parameters**: ~2.5M (0.03% of total)
- **Training**: 5 epochs, batch size 1, gradient accumulation 16
- **Learning Rate**: 5.0e-5
- **Max Sequence Length**: 3072

### Instruction Templates
All experiments use instruction templates that:
- Specify answer format (single word, yes/no, multi-label, etc.)
- Provide candidate lists for constrained questions
- Explicitly instruct model to avoid verbose explanations
- Include question-type specific guidance

---

## Summary

| Metric | Value |
|--------|-------|
| **Zero-Shot Baseline** | 53.48% |
| **Instruction Fine-Tuning Range** | 92.23% - 92.79% |
| **Average Improvement** | +39.3 percentage points |
| **Best Experiment** | Exp1 (Random Baseline) - 92.79% |
| **Standard Fine-Tuning** | Not performed (replaced by instruction fine-tuning) |

---

*Last Updated: Based on results from qlora_experiments/results/*

