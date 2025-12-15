# Experiment 5: Sequential Chain-of-Thought (CoT)

## Overview

Experiment 5 implements **cascading inference** where each stage builds on previous predictions, creating a sequential chain of reasoning.

**Key Concept:** Unlike training-based approaches (Exp 3 & 4), this uses **inference-time cascading** with a single trained model.

## Sequential Flow

```
Stage 1: Input → Model → Pre 1 (prediction)
Stage 2: Input + Pre 1 (from Stage 1) → Model → Pre 2 (prediction)
Stage 3: Input + Pre 1 (from Stage 1) + Pre 2 (from Stage 2) → Model → Pre 3 (final prediction)
```

## Hypothesis

Sequential reasoning with previous context improves final prediction accuracy by:
- Providing initial assessment context (Stage 1) for findings identification (Stage 2)
- Using both initial assessment and findings context (Stages 1+2) for clinical reasoning (Stage 3)

## Implementation Details

### Training
**No separate training needed!** This experiment uses an already-trained model from:
- **Experiment 1** (Random Baseline) or
- **Experiment 2** (Qwen Reordering)

### Evaluation
The evaluation script implements cascading inference:

1. Loads trained model checkpoint (from Exp1 or Exp2)
2. Groups test questions by image (requires 3 questions per image with stages 1, 2, 3)
3. For each image:
   - **Stage 1**: Generates prediction with just image + question
   - **Stage 2**: Generates prediction with image + question + Stage 1 prediction
   - **Stage 3**: Generates prediction with image + question + Stage 1 + Stage 2 predictions
4. Evaluates accuracy at each stage

## Data Requirements

The test dataset must have:
- Entries grouped by `image_id`
- Each image should have exactly 3 questions with `stage` field: 1, 2, and 3
- Questions should be from the Qwen-reordered dataset

Example structure:
```json
[
  {"image_id": "img_001", "image_filename": "img_001.jpg", "question": "...", "answer": "...", "stage": 1},
  {"image_id": "img_001", "image_filename": "img_001.jpg", "question": "...", "answer": "...", "stage": 2},
  {"image_id": "img_001", "image_filename": "img_001.jpg", "question": "...", "answer": "...", "stage": 3}
]
```

## Usage

### After Exp1 or Exp2 completes:

```bash
cd "/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments"

# Edit evaluate_sequential_cot.slurm to set MODEL_PATH:
#   MODEL_PATH="${BASE_DIR}/corrected 1-5 experiments/models/exp1_random_baseline"
#   OR
#   MODEL_PATH="${BASE_DIR}/corrected 1-5 experiments/models/exp2_qwen_reordered"

# Then submit:
sbatch experiments/exp5_sequential_cot/evaluate_sequential_cot.slurm
```

### Manual evaluation:

```bash
python3 experiments/exp5_sequential_cot/evaluate_sequential_cot.py \
    --model_path "corrected 1-5 experiments/models/exp1_random_baseline" \
    --test_data "corrected 1-5 experiments/datasets/kvasir_raw_6500_image_level_70_15_15/test.json" \
    --image_dir "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images" \
    --output "corrected 1-5 experiments/results/exp5_sequential_cot_results.json" \
    --base_model "Qwen/Qwen2-VL-7B-Instruct"
```

## Output

The evaluation produces:
- Accuracy for each stage (Stage 1, Stage 2, Stage 3)
- Overall accuracy (average across all stages)
- Detailed predictions with context used at each stage
- Error log for debugging

Results are saved as JSON with the following structure:
```json
{
  "total_images": 975,
  "stage1": {"total": 975, "correct": 800, "accuracy": 82.05},
  "stage2": {"total": 975, "correct": 700, "accuracy": 71.79},
  "stage3": {"total": 975, "correct": 920, "accuracy": 94.36},
  "overall": {"total": 2925, "correct": 2420, "accuracy": 82.74},
  "predictions": [...]
}
```

## Differences from Other Experiments

| Experiment | Approach | Training | Inference |
|------------|----------|----------|-----------|
| Exp 1-2 | Single model, all stages | Train once | Single pass |
| Exp 3 | 3 separate models | Train 3 models independently | Use stage-specific model |
| Exp 4 | Progressive curriculum | Train sequentially (checkpoint loading) | Use final checkpoint |
| **Exp 5** | **Cascading CoT** | **Use trained model from Exp1/2** | **Chain predictions** |

## Notes

- This experiment tests whether **inference-time context** (previous predictions) improves later stage accuracy
- The model itself is not retrained - it uses a model trained on all stages simultaneously
- The cascade effect shows whether the model can leverage its own predictions as context
- This is useful for understanding if the model benefits from explicit reasoning chains

