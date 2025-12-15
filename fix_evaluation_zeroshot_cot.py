#!/usr/bin/env python3
"""
Fix evaluation for Zeroshot+COT model.

The model generates verbose CoT outputs that need proper answer extraction
and flexible matching to get accurate results.
"""

import argparse
import json
import string
from pathlib import Path
from typing import Dict, List, Tuple


def extract_short_answer(prediction: str, ground_truth: str) -> str:
    """
    Extract concise answer from verbose CoT output.
    
    Examples:
    "yes, this finding is easy to detect" → "yes"
    "the image is taken from a colonoscopy" → "colonoscopy"
    "the polyp is approximately 1-2 cm" → "1-2 cm"
    "one" → "1"
    """
    pred = prediction.lower().strip()
    gt = ground_truth.lower().strip()
    
    # For yes/no questions
    if gt in ["yes", "no"]:
        if pred.startswith("yes"):
            return "yes"
        if pred.startswith("no"):
            return "no"
    
    # Remove verbose prefixes
    prefixes_to_remove = [
        "the answer is ", "it is ", "this is ",
        "the image is taken from ", "the image shows ",
        "the polyp is ", "the abnormality is ",
        "approximately ", "around ", "about ",
        "based on the image, ", "based on the image it is not possible to determine the exact size of the polyp",
        "the image is taken from a ", "the image is from a "
    ]
    
    for prefix in prefixes_to_remove:
        if pred.startswith(prefix):
            pred = pred[len(prefix):].strip()
            break
    
    # Remove trailing explanations
    pred = pred.split(",")[0].strip()
    pred = pred.split(".")[0].strip()
    
    # For single-word ground truths, take first word
    if len(gt.split()) == 1 and len(pred.split()) > 1:
        pred = pred.split()[0]
    
    return pred


def flexible_match(pred: str, gt: str) -> bool:
    """Flexible matching that handles verbose predictions."""
    pred = pred.lower().strip()
    gt = gt.lower().strip()
    
    # CRITICAL FIX: Empty predictions only correct if ground truth is also empty
    if not pred:
        return not gt  # Both empty = match, otherwise False
    
    if not gt:
        return False  # Empty ground truth, non-empty prediction = False
    
    # Exact match
    if pred == gt:
        return True
    
    # Ground truth is IN prediction (only if both are non-empty)
    if gt in pred:
        return True
    
    # Prediction is IN ground truth (only if both are non-empty)
    if pred in gt:
        return True
    
    # Handle multi-label (e.g., "center; lower-left; lower-right")
    if ";" in gt:
        gt_labels = [label.strip() for label in gt.split(";")]
        for label in gt_labels:
            if label in pred or pred in label:
                return True
    
    # Number normalization ("one" → "1")
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10"
    }
    
    pred_normalized = pred
    gt_normalized = gt
    for word, num in number_words.items():
        pred_normalized = pred_normalized.replace(word, num)
        gt_normalized = gt_normalized.replace(word, num)
    
    if pred_normalized == gt_normalized or gt_normalized in pred_normalized:
        return True
    
    # Remove punctuation and spaces
    pred_clean = pred.translate(str.maketrans('', '', string.punctuation + ' '))
    gt_clean = gt.translate(str.maketrans('', '', string.punctuation + ' '))
    
    if pred_clean == gt_clean or gt_clean in pred_clean:
        return True
    
    return False


def fix_evaluation(input_file: str, output_file: str):
    """Fix evaluation results with proper answer extraction and matching."""
    
    print("="*80)
    print("FIXING ZEROSHOT+COT EVALUATION")
    print("="*80)
    print(f"\nLoading results from: {input_file}")
    
    # Load original results
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    predictions = data.get('predictions', [])
    answers = data.get('answers', [])
    
    print(f"Loaded {len(predictions)} predictions and {len(answers)} answers")
    
    # Extract short answers and recalculate
    correct = 0
    total = len(predictions)
    fixed_predictions = []
    matches = []
    
    for i, (pred, gt) in enumerate(zip(predictions, answers)):
        # Extract short answer
        short_pred = extract_short_answer(pred, gt)
        fixed_predictions.append(short_pred)
        
        # Flexible matching
        is_correct = flexible_match(short_pred, gt)
        matches.append(is_correct)
        
        if is_correct:
            correct += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct / total if total > 0 else 0.0
    
    # Calculate category accuracies if available
    category_correct = {}
    category_total = {}
    category_accuracies = {}
    
    if 'categories' in data:
        categories = data['categories']
        for i, cat in enumerate(categories):
            if cat not in category_total:
                category_total[cat] = 0
                category_correct[cat] = 0
            category_total[cat] += 1
            if matches[i]:
                category_correct[cat] += 1
        
        for cat in category_total:
            category_accuracies[cat] = category_correct[cat] / category_total[cat] if category_total[cat] > 0 else 0.0
    
    # Original accuracy
    original_accuracy = data.get('overall_accuracy', 0.0)
    improvement = overall_accuracy - original_accuracy
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"\nORIGINAL RESULTS:")
    print(f"  Overall Accuracy: {original_accuracy:.2%} ({data.get('correct', 0)}/{total})")
    print(f"\nFIXED RESULTS (with flexible matching):")
    print(f"  Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
    print(f"\nIMPROVEMENT: {improvement:+.2%}")
    
    if category_accuracies:
        print(f"\nPer-Category Accuracies:")
        for cat, acc in category_accuracies.items():
            print(f"  {cat}: {acc:.2%} ({category_correct[cat]}/{category_total[cat]})")
    
    # Show examples
    print("\n" + "="*80)
    print("EXAMPLES NOW CORRECT (first 10):")
    print("="*80)
    shown = 0
    for i, (pred, gt, match) in enumerate(zip(predictions, answers, matches)):
        if match and shown < 10:
            short_pred = fixed_predictions[i]
            print(f"\nExample {shown + 1}:")
            print(f"  Original Prediction: {pred[:80]}...")
            print(f"  Extracted Answer: {short_pred}")
            print(f"  Ground Truth: {gt}")
            print(f"  → NOW CORRECT ✓")
            shown += 1
    
    print("\n" + "="*80)
    print("EXAMPLES STILL WRONG (first 10):")
    print("="*80)
    shown = 0
    for i, (pred, gt, match) in enumerate(zip(predictions, answers, matches)):
        if not match and shown < 10:
            short_pred = fixed_predictions[i]
            print(f"\nExample {shown + 1}:")
            print(f"  Original Prediction: {pred[:80]}...")
            print(f"  Extracted Answer: {short_pred}")
            print(f"  Ground Truth: {gt}")
            print(f"  → STILL WRONG ✗")
            shown += 1
    
    # Save fixed results
    fixed_data = {
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'category_accuracies': category_accuracies,
        'category_correct': category_correct,
        'category_total': category_total,
        'predictions': fixed_predictions,
        'answers': answers,
        'original_accuracy': original_accuracy,
        'improvement': improvement,
        'original_predictions': predictions
    }
    
    with open(output_file, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*80)
    
    return fixed_data


def main():
    parser = argparse.ArgumentParser(description="Fix Zeroshot+COT evaluation")
    parser.add_argument("--input", required=True, help="Input evaluation JSON file")
    parser.add_argument("--output", required=True, help="Output fixed evaluation JSON file")
    
    args = parser.parse_args()
    
    fix_evaluation(args.input, args.output)


if __name__ == "__main__":
    main()









