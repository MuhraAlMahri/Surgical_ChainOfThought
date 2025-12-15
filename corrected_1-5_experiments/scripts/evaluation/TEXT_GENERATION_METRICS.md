# Text Generation Metrics (BLEU, ROUGE, METEOR) - Implementation Guide

## Overview

BLEU, ROUGE, and METEOR metrics have been added to complement the existing accuracy-based evaluation. These metrics are particularly useful for:

- **Open-ended questions** (37.6% of your dataset)
- **Multi-label questions** where partial correctness matters
- **Measuring semantic similarity** beyond exact matches
- **Comparing model performance** on text generation quality

---

## Metrics Explained

### BLEU (Bilingual Evaluation Understudy)
- **What it measures**: N-gram precision between prediction and reference
- **Range**: 0.0 to 1.0 (higher is better)
- **Best for**: Measuring word-level overlap
- **Limitation**: Doesn't consider synonyms or word order well

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Unigram (word) overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest Common Subsequence (considers word order)
- **Range**: 0.0 to 1.0 (higher is better)
- **Best for**: Measuring recall of important information

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)
- **What it measures**: Semantic similarity considering synonyms and word order
- **Range**: 0.0 to 1.0 (higher is better)
- **Best for**: Most semantic evaluation, handles synonyms well
- **Advantage**: More correlated with human judgment than BLEU

---

## Installation

Install required dependencies:

```bash
pip install nltk rouge-score
```

The code will automatically download required NLTK data (punkt, wordnet) on first run.

---

## Usage

### In Evaluation Scripts

The metrics are automatically calculated in all evaluation scripts. Example from `evaluate_exp1.py`:

```python
from metrics_utils import calculate_text_generation_metrics

# Calculate all text generation metrics at once
text_metrics = calculate_text_generation_metrics(prediction, ground_truth)
bleu = text_metrics['bleu']
rouge1 = text_metrics['rouge1']
rouge2 = text_metrics['rouge2']
rougeL = text_metrics['rougeL']
meteor = text_metrics['meteor']
```

### Output Format

Results are included in the JSON output:

```json
{
  "total": 8984,
  "correct": 8336,
  "accuracy": 92.79,
  "bleu": 45.23,
  "rouge1": 67.89,
  "rouge2": 52.34,
  "rougeL": 64.56,
  "meteor": 58.12,
  "by_stage": {
    "Stage 1": {
      "bleu": 46.12,
      "rouge1": 68.45,
      ...
    }
  },
  "by_question_type": {
    "open_ended": {
      "bleu": 38.23,
      "rouge1": 61.34,
      ...
    }
  }
}
```

### Console Output

```
--- Text Generation Metrics ---
BLEU: 45.23%
ROUGE-1: 67.89%
ROUGE-2: 52.34%
ROUGE-L: 64.56%
METEOR: 58.12%
```

---

## When to Use Each Metric

### Use Accuracy When:
- ✅ Binary correctness is most important
- ✅ Single-choice or yes/no questions
- ✅ Exact match is required

### Use BLEU When:
- ✅ Measuring word-level precision
- ✅ Comparing n-gram overlap
- ✅ Standard NLP benchmark comparison

### Use ROUGE When:
- ✅ Measuring recall of important information
- ✅ Evaluating summarization quality
- ✅ Need to know if key terms are mentioned

### Use METEOR When:
- ✅ Need semantic understanding
- ✅ Synonyms should be considered correct
- ✅ Want metric closest to human judgment

---

## Example Scenarios

### Scenario 1: Open-Ended Question
**Question**: "What findings are present in this image?"

**Ground Truth**: "polyp; bleeding; ulcer"

**Prediction 1**: "polyp; bleeding"  
- Accuracy: ❌ 0% (not exact match)
- BLEU: 66.7% (2/3 words match)
- ROUGE-1: 66.7% (2/3 unigrams)
- METEOR: 70.2% (considers partial match)

**Prediction 2**: "polyp; bleeding; ulcer"  
- Accuracy: ✅ 100% (exact match)
- BLEU: 100%
- ROUGE-1: 100%
- METEOR: 100%

### Scenario 2: Synonym Handling
**Ground Truth**: "no findings"

**Prediction**: "no abnormalities detected"  
- Accuracy: ❌ 0% (different words)
- BLEU: 0% (no word overlap)
- ROUGE-1: 0% (no unigram match)
- METEOR: 45.2% (recognizes semantic similarity)

---

## Integration with Existing Metrics

The new metrics work alongside existing metrics:

- **Accuracy**: Binary correctness (existing)
- **Precision/Recall/F1**: Set-based matching for multi-label (existing)
- **BLEU/ROUGE/METEOR**: Text generation quality (new)

All metrics are calculated for:
- Overall performance
- By stage (Stage 1, 2, 3)
- By question type (single_choice, numeric, multi_label, open_ended)

---

## Performance Considerations

- **BLEU/ROUGE/METEOR calculation** adds ~0.01-0.02 seconds per sample
- For 8,984 samples: ~2-3 minutes additional evaluation time
- Metrics gracefully degrade if dependencies are missing (return 0.0)

---

## Troubleshooting

### Import Errors
If you see `ImportError` for `nltk` or `rouge_score`:
```bash
pip install nltk rouge-score
```

### NLTK Data Missing
The code automatically downloads required NLTK data, but if you see errors:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### Zero Scores
If all metrics are 0.0:
- Check that dependencies are installed
- Verify predictions and ground truth are non-empty strings
- Check console for error messages

---

## Updated Scripts

✅ **evaluate_exp1.py** - Updated with BLEU/ROUGE/METEOR  
⏳ **evaluate_exp2.py** - Can be updated similarly  
⏳ **evaluate_exp3.py** - Can be updated similarly  
⏳ **evaluate_exp4.py** - Can be updated similarly  
⏳ **evaluate_exp5.py** - Can be updated similarly  

To update other scripts, follow the same pattern as `evaluate_exp1.py`.

---

## References

- **BLEU**: Papineni et al., "BLEU: a method for automatic evaluation of machine translation" (2002)
- **ROUGE**: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries" (2004)
- **METEOR**: Banerjee & Lavie, "METEOR: An Automatic Metric for MT Evaluation" (2005)

---

**Date Added**: 2025-01-XX  
**Status**: ✅ Implemented in evaluate_exp1.py, ready for other scripts









