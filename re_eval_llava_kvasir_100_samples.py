#!/usr/bin/env python3
"""
Quick re-evaluation script for LLaVA-Med Kvasir on 100 samples.
Uses the FIXED evaluation logic to verify the bug fix.

This will:
1. Load the existing result file
2. Re-evaluate first 100 samples with FIXED matching logic
3. Compare old vs new accuracy
4. Show sample predictions to verify they're not all empty
"""

import json
import sys
from pathlib import Path
from difflib import SequenceMatcher

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "corrected_1-5_experiments" / "scripts" / "evaluation"))

# Import normalize_text and smart_match directly to avoid import issues
def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().replace(".", "").replace(",", "").replace(";", "")


def smart_match(prediction: str, ground_truth: str, threshold: float = 0.7) -> bool:
    """Smart matching with FIXED logic - empty predictions only match empty GT."""
    pred_n = normalize_text(prediction)
    gt_n = normalize_text(ground_truth)
    
    # CRITICAL FIX: Empty predictions only correct if ground truth is also empty
    if not pred_n:
        return not gt_n  # Both empty = match, otherwise False
    
    if not gt_n:
        return False  # Empty ground truth, non-empty prediction = False
    
    if pred_n == gt_n:
        return True
    
    # Substring match (only if both are non-empty)
    if gt_n in pred_n:
        return True
    
    similarity = SequenceMatcher(None, pred_n, gt_n).ratio()
    return similarity >= threshold

def count_empty_predictions(results):
    """Count how many predictions are empty."""
    empty_count = 0
    total = 0
    
    for item in results:
        if 'prediction' in item:
            pred = item.get('prediction', '').strip()
            if not pred:
                empty_count += 1
            total += 1
    
    return empty_count, total


def re_evaluate_with_fixed_logic(result_file: str, max_samples: int = 100):
    """Re-evaluate results with fixed matching logic."""
    print("=" * 80)
    print("RE-EVALUATING LLaVA-Med Kvasir with FIXED Logic")
    print("=" * 80)
    print()
    
    # Load existing results
    print(f"Loading results from: {result_file}")
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        # Check for common keys
        if 'results' in data:
            results = data['results']
        elif 'predictions' in data:
            results = data['predictions']
        else:
            # Try to find a list value
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    results = value
                    break
            else:
                raise ValueError(f"Could not find results list in JSON file. Keys: {list(data.keys())}")
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")
    
    print(f"✓ Loaded {len(results)} samples")
    
    # Limit to first 100 samples (only if results is a list)
    if isinstance(results, list):
        if max_samples and len(results) > max_samples:
            results = results[:max_samples]
            print(f"✓ Limited to first {len(results)} samples for quick test")
    else:
        raise ValueError(f"Results is not a list: {type(results)}")
    
    print()
    
    # Count empty predictions
    empty_count, total = count_empty_predictions(results)
    print(f"Empty predictions: {empty_count}/{total} ({100*empty_count/total:.1f}%)")
    print()
    
    # Re-evaluate with fixed logic
    print("Re-evaluating with FIXED matching logic...")
    print("-" * 80)
    
    correct_old = 0
    correct_new = 0
    total_samples = 0
    
    # Sample predictions to show
    sample_predictions = []
    
    for i, item in enumerate(results):
        if 'prediction' not in item or 'ground_truth' not in item:
            continue
        
        pred = item.get('prediction', '').strip()
        gt = item.get('ground_truth', '').strip()
        
        # Old logic (broken): "" in "yes" returns True
        pred_n = normalize_text(pred)
        gt_n = normalize_text(gt)
        old_match = False
        if pred_n == gt_n:
            old_match = True
        elif gt_n in pred_n or pred_n in gt_n:  # BUG: "" in "yes" = True!
            old_match = True
        
        # New logic (fixed): Empty predictions only match empty GT
        new_match = smart_match(pred, gt)
        
        if old_match:
            correct_old += 1
        if new_match:
            correct_new += 1
        
        total_samples += 1
        
        # Collect sample predictions
        if len(sample_predictions) < 10:
            sample_predictions.append({
                'index': i,
                'prediction': pred[:50] if pred else "(empty)",
                'ground_truth': gt[:50],
                'old_match': old_match,
                'new_match': new_match
            })
    
    # Calculate accuracies
    accuracy_old = (correct_old / total_samples * 100) if total_samples > 0 else 0
    accuracy_new = (correct_new / total_samples * 100) if total_samples > 0 else 0
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Total samples evaluated: {total_samples}")
    print()
    print(f"OLD (BROKEN) Logic:")
    print(f"  Correct: {correct_old}/{total_samples}")
    print(f"  Accuracy: {accuracy_old:.2f}%")
    print()
    print(f"NEW (FIXED) Logic:")
    print(f"  Correct: {correct_new}/{total_samples}")
    print(f"  Accuracy: {accuracy_new:.2f}%")
    print()
    print(f"Difference: {accuracy_old - accuracy_new:.2f}% (old was inflated by bug)")
    print()
    
    print("=" * 80)
    print("SAMPLE PREDICTIONS (first 10)")
    print("=" * 80)
    print()
    for sample in sample_predictions:
        print(f"Sample {sample['index']}:")
        print(f"  Prediction: '{sample['prediction']}'")
        print(f"  Ground Truth: '{sample['ground_truth']}'")
        print(f"  Old Match: {sample['old_match']} | New Match: {sample['new_match']}")
        if sample['prediction'] == "(empty)":
            print(f"  ⚠️  EMPTY PREDICTION")
        print()
    
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()
    
    if accuracy_new < accuracy_old:
        print("✅ FIX VERIFIED: New accuracy is lower (as expected)")
        print(f"   Old accuracy was inflated by {accuracy_old - accuracy_new:.2f}% due to bug")
    else:
        print("⚠️  Unexpected: New accuracy is not lower")
    
    if empty_count > 0:
        print(f"⚠️  WARNING: {empty_count} empty predictions found ({100*empty_count/total:.1f}%)")
        print("   This explains the inflated accuracy with old logic")
    else:
        print("✅ No empty predictions found")
    
    if accuracy_new < 30:
        print(f"✅ New accuracy ({accuracy_new:.2f}%) is reasonable (not inflated)")
    else:
        print(f"⚠️  New accuracy ({accuracy_new:.2f}%) is still high - may need further investigation")
    
    print()
    print("=" * 80)
    
    return {
        'total_samples': total_samples,
        'empty_predictions': empty_count,
        'accuracy_old': accuracy_old,
        'accuracy_new': accuracy_new,
        'difference': accuracy_old - accuracy_new
    }


if __name__ == "__main__":
    # Default result file
    result_file = "corrected_1-5_experiments/qlora_experiments/results/kvasir_finetuned_llava_med_v15.json"
    
    if len(sys.argv) > 1:
        result_file = sys.argv[1]
    
    max_samples = 100
    if len(sys.argv) > 2:
        max_samples = int(sys.argv[2])
    
    try:
        results = re_evaluate_with_fixed_logic(result_file, max_samples)
        
        # Save summary
        summary_file = "re_eval_llava_kvasir_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        
    except FileNotFoundError:
        print(f"❌ ERROR: Result file not found: {result_file}")
        print("\nAvailable result files:")
        import glob
        for f in glob.glob("corrected_1-5_experiments/qlora_experiments/results/kvasir*.json"):
            print(f"  - {f}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

