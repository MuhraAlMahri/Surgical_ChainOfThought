"""
Shared utility functions for evaluation metrics (Accuracy, Precision, Recall, F1, BLEU, ROUGE, METEOR)
Used across all experiment evaluation scripts.
"""

from difflib import SequenceMatcher
import re

# Try to import optional dependencies for text generation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().replace(".", "").replace(",", "").replace(";", "")


def parse_labels(text: str) -> set:
    """Parse labels from text (handles semicolon-separated multi-label format)."""
    if not text:
        return set()
    # Split by semicolon and normalize
    labels = [normalize_text(label) for label in text.split(';')]
    # Remove empty labels
    return set(label for label in labels if label)


def calculate_precision_recall_f1(pred_set: set, gt_set: set) -> tuple:
    """Calculate precision, recall, and F1 score for two sets of labels.
    
    Args:
        pred_set: Set of predicted labels
        gt_set: Set of ground truth labels
    
    Returns:
        Tuple of (precision, recall, f1) as floats (0.0 to 1.0)
    """
    if not gt_set:
        # If no ground truth, precision/recall are undefined
        return (0.0, 0.0, 0.0)
    
    if not pred_set:
        # If no prediction but there is ground truth
        return (0.0, 0.0, 0.0)
    
    # Calculate intersection
    intersection = pred_set & gt_set
    
    # Precision: how many predicted labels are correct
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    
    # Recall: how many ground truth labels were found
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    
    # F1: harmonic mean of precision and recall
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return (precision, recall, f1)


def smart_match(prediction: str, ground_truth: str, threshold: float = 0.7) -> bool:
    """Smart matching with multiple strategies for binary correctness.
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
        threshold: Similarity threshold for fuzzy matching (default: 0.7)
    
    Returns:
        True if prediction matches ground truth, False otherwise
    """
    pred_n = normalize_text(prediction)
    gt_n = normalize_text(ground_truth)
    
    # CRITICAL FIX: Empty predictions only correct if ground truth is also empty
    if not pred_n:
        return not gt_n  # Both empty = match, otherwise False
    
    if not gt_n:
        return False  # Empty ground truth, non-empty prediction = False
    
    # Exact match
    if pred_n == gt_n:
        return True
    
    # Substring match (only if both are non-empty)
    if gt_n in pred_n or pred_n in gt_n:
        return True
    
    # Fuzzy similarity
    similarity = SequenceMatcher(None, pred_n, gt_n).ratio()
    return similarity >= threshold


def tokenize_for_metrics(text: str) -> list:
    """Tokenize text for BLEU/ROUGE/METEOR metrics.
    
    Args:
        text: Input text string
    
    Returns:
        List of tokens (words)
    """
    if not text:
        return []
    
    # Simple tokenization: split on whitespace and punctuation
    # Normalize to lowercase
    text = text.lower().strip()
    # Split on whitespace and common punctuation
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def calculate_bleu(prediction: str, ground_truth: str) -> float:
    """Calculate BLEU score between prediction and ground truth.
    
    BLEU measures n-gram overlap between prediction and reference.
    Range: 0.0 to 1.0 (higher is better)
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
    
    Returns:
        BLEU score (0.0 to 1.0), or 0.0 if NLTK not available
    """
    if not NLTK_AVAILABLE:
        return 0.0
    
    if not ground_truth:
        return 0.0
    
    try:
        # Tokenize
        pred_tokens = tokenize_for_metrics(prediction)
        gt_tokens = tokenize_for_metrics(ground_truth)
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        # Use smoothing to handle cases where no n-grams match
        smoothing = SmoothingFunction().method1
        # Calculate BLEU-4 (4-gram precision)
        score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothing)
        return score
    except Exception:
        return 0.0


def calculate_rouge(prediction: str, ground_truth: str) -> dict:
    """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between prediction and ground truth.
    
    ROUGE measures recall of n-grams and longest common subsequence.
    Range: 0.0 to 1.0 (higher is better)
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
    
    Returns:
        Dictionary with 'rouge1', 'rouge2', 'rougeL' scores (0.0 to 1.0),
        or all zeros if rouge-score not available
    """
    if not ROUGE_AVAILABLE:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    if not ground_truth:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except Exception:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def calculate_meteor(prediction: str, ground_truth: str) -> float:
    """Calculate METEOR score between prediction and ground truth.
    
    METEOR considers synonymy and word order, making it more semantic than BLEU.
    Range: 0.0 to 1.0 (higher is better)
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
    
    Returns:
        METEOR score (0.0 to 1.0), or 0.0 if NLTK not available
    """
    if not NLTK_AVAILABLE:
        return 0.0
    
    if not ground_truth:
        return 0.0
    
    try:
        # Tokenize
        pred_tokens = tokenize_for_metrics(prediction)
        gt_tokens = tokenize_for_metrics(ground_truth)
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        # METEOR requires word_tokenize from NLTK
        pred_tokens_nltk = word_tokenize(prediction.lower())
        gt_tokens_nltk = word_tokenize(ground_truth.lower())
        
        score = meteor_score([gt_tokens_nltk], pred_tokens_nltk)
        return score
    except Exception:
        return 0.0


def calculate_text_generation_metrics(prediction: str, ground_truth: str) -> dict:
    """Calculate all text generation metrics (BLEU, ROUGE, METEOR) at once.
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
    
    Returns:
        Dictionary with 'bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor' scores
    """
    bleu = calculate_bleu(prediction, ground_truth)
    rouge = calculate_rouge(prediction, ground_truth)
    meteor = calculate_meteor(prediction, ground_truth)
    
    return {
        'bleu': bleu,
        'rouge1': rouge['rouge1'],
        'rouge2': rouge['rouge2'],
        'rougeL': rouge['rougeL'],
        'meteor': meteor
    }

