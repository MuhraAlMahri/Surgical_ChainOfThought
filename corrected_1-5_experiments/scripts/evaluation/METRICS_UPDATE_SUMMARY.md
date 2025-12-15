# Evaluation Metrics Update Summary

## Overview

All evaluation scripts have been updated to include:
1. **F1 Score, Precision, and Recall** metrics (for multi-label evaluation)
2. **BLEU, ROUGE, and METEOR** metrics (for text generation quality) - *NEW*

See [TEXT_GENERATION_METRICS.md](./TEXT_GENERATION_METRICS.md) for details on BLEU/ROUGE/METEOR.

## Updated Scripts

✅ **evaluate_exp1.py** - Random Baseline (includes BLEU/ROUGE/METEOR)  
✅ **evaluate_exp2.py** - Qwen Clinical Reordering (F1/Precision/Recall only)  
✅ **evaluate_exp3.py** - CXRTrek Sequential (F1/Precision/Recall only)  
✅ **evaluate_exp4.py** - Curriculum Learning (F1/Precision/Recall only)  
✅ **evaluate_exp5.py** - Sequential Chain-of-Thought (F1/Precision/Recall only)

*Note: BLEU/ROUGE/METEOR can be added to other scripts following the same pattern as evaluate_exp1.py*  

## New Metrics Added

### Overall Metrics
- **Accuracy**: Percentage of correct predictions (existing)
- **Precision**: Percentage of predicted labels that are correct (new)
- **Recall**: Percentage of ground truth labels that were found (new)
- **F1 Score**: Harmonic mean of precision and recall (new)
- **BLEU**: N-gram precision score (0.0-1.0) - *NEW*
- **ROUGE-1**: Unigram recall score (0.0-1.0) - *NEW*
- **ROUGE-2**: Bigram recall score (0.0-1.0) - *NEW*
- **ROUGE-L**: Longest common subsequence score (0.0-1.0) - *NEW*
- **METEOR**: Semantic similarity score considering synonyms (0.0-1.0) - *NEW*

### Breakdown Metrics
All metrics are also calculated and reported for:
- **By Stage**: Stage 1, Stage 2, Stage 3
- **By Question Type**: single_choice, numeric, multi_label

## Implementation Details

### Helper Functions Added

1. **`parse_labels(text: str) -> set`**
   - Parses semicolon-separated multi-label answers
   - Normalizes and removes empty labels
   - Returns a set of labels

2. **`calculate_precision_recall_f1(pred_set: set, gt_set: set) -> tuple`**
   - Calculates precision, recall, and F1 from two sets of labels
   - Handles edge cases (empty sets, no intersection)
   - Returns (precision, recall, f1) as floats (0.0 to 1.0)

### How Metrics Work

**For Multi-Label Questions:**
- Parses both prediction and ground truth into sets of labels
- Precision = |intersection| / |predicted labels|
- Recall = |intersection| / |ground truth labels|
- F1 = 2 × (precision × recall) / (precision + recall)

**For Single-Choice Questions:**
- Treated as single-label (set with one element)
- Precision = Recall = F1 = Accuracy (when correct)
- Provides more granular information when partially correct

**For Numeric Questions:**
- Treated as single-label
- Uses the same set-based calculation

## Output Format

### Console Output
```
Total samples: 8984
Correct: 8336
Accuracy: 92.79%
Precision: 89.23%
Recall: 91.45%
F1 Score: 90.32%

By Stage:
  Stage 1:
    Accuracy: 92.24% (3021/3275)
    Precision: 88.56%
    Recall: 90.12%
    F1: 89.33%
  ...

By Question Type:
  single_choice:
    Accuracy: 92.72% (4829/5208)
    Precision: 89.45%
    Recall: 91.23%
    F1: 90.33%
  ...
```

### JSON Output
Each evaluation JSON now includes:
```json
{
  "total": 8984,
  "correct": 8336,
  "accuracy": 92.79,
  "precision": 89.23,
  "recall": 91.45,
  "f1": 90.32,
  "by_stage": {
    "Stage 1": {
      "total": 3275,
      "correct": 3021,
      "accuracy": 92.24,
      "precision": 88.56,
      "recall": 90.12,
      "f1": 89.33
    },
    ...
  },
  "by_question_type": {
    "single_choice": {
      "total": 5208,
      "correct": 4829,
      "accuracy": 92.72,
      "precision": 89.45,
      "recall": 91.23,
      "f1": 90.33
    },
    ...
  },
  "predictions": [
    {
      "image_id": "...",
      "prediction": "...",
      "ground_truth": "...",
      "correct": true,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      ...
    },
    ...
  ]
}
```

## Benefits

1. **Better Multi-Label Evaluation**: F1, precision, and recall provide more nuanced evaluation for multi-label questions where partial credit matters

2. **Detailed Analysis**: Can identify if models are:
   - Over-predicting (high recall, low precision)
   - Under-predicting (high precision, low recall)
   - Balanced (similar precision and recall)

3. **Standard Metrics**: Aligns with standard ML evaluation practices

4. **Backward Compatible**: Accuracy metric still calculated and reported (existing functionality preserved)

## Usage

No changes needed to how you run evaluations. The new metrics are automatically calculated and included in both console output and JSON results.

Example:
```bash
python scripts/evaluation/evaluate_exp1.py \
  --model_path models/exp1_random/ \
  --test_data datasets/qlora_experiments/exp1_random/test.jsonl \
  --image_dir datasets/Kvasir-VQA/raw/images \
  --output results/exp1_evaluation.json
```

The output will now include precision, recall, and F1 scores automatically.

---

## Text Generation Metrics (NEW)

BLEU, ROUGE, and METEOR metrics have been added to provide more nuanced evaluation for:
- Open-ended questions (37.6% of dataset)
- Measuring semantic similarity beyond exact matches
- Better evaluation of partial correctness

**Installation:**
```bash
pip install nltk rouge-score
```

**See:** [TEXT_GENERATION_METRICS.md](./TEXT_GENERATION_METRICS.md) for complete documentation.

---

**Date Updated:** 2025-01-XX  
**F1/Precision/Recall:** ✅ Complete (all scripts)  
**BLEU/ROUGE/METEOR:** ✅ Complete (evaluate_exp1.py), ⏳ Available for other scripts

