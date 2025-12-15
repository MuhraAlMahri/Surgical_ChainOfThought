"""
Evaluate Exp2 final checkpoint predictions (2.5 epochs, full resolution)
"""
import json
import re
from pathlib import Path
from collections import defaultdict


def normalize(text):
    """Normalize text for comparison"""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text


def maybe_float(s):
    """Try to extract float from string"""
    try:
        return float(s)
    except:
        match = re.findall(r'[-+]?\d*\.?\d+', str(s))
        if match:
            try:
                return float(match[0])
            except:
                return None
        return None


def evaluate(pred_path):
    """Evaluate predictions"""
    print("="*80)
    print("EXP2 EVALUATION - Checkpoint-6000 (2.34 Epochs, Full Resolution)")
    print("="*80)
    
    counts = defaultdict(int)
    correct = defaultdict(int)
    
    with open(pred_path) as f:
        for line in f:
            ex = json.loads(line)
            category = ex.get("category", "unknown")
            gt = normalize(ex["gt"])
            pred = normalize(ex["pred"])
            
            counts[category] += 1
            
            # Simple exact match (can be improved)
            if gt == pred:
                correct[category] += 1
    
    # Calculate per-category accuracy
    results = {}
    for cat in sorted(counts.keys()):
        acc = (correct[cat] / counts[cat]) * 100 if counts[cat] > 0 else 0
        results[cat] = {
            "count": counts[cat],
            "correct": correct[cat],
            "accuracy": acc
        }
    
    # Overall accuracy
    total_count = sum(counts.values())
    total_correct = sum(correct.values())
    overall_acc = (total_correct / total_count) * 100 if total_count > 0 else 0
    
    # Print results
    print(f"\n{'Category':<30} {'Count':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-"*80)
    
    for cat in sorted(results.keys()):
        r = results[cat]
        print(f"{cat:<30} {r['count']:<10} {r['correct']:<10} {r['accuracy']:<10.2f}%")
    
    print("-"*80)
    print(f"{'OVERALL':<30} {total_count:<10} {total_correct:<10} {overall_acc:<10.2f}%")
    print("="*80)
    
    return results, overall_acc


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    pred_file = script_dir / "outputs/checkpoint-6000/predictions_final.jsonl"
    
    if not pred_file.exists():
        print(f"❌ Predictions not found: {pred_file}")
        print("Run prediction first!")
    else:
        results, overall = evaluate(pred_file)
        
        # Save results
        output = script_dir / "outputs/checkpoint-6000/eval_final.json"
        with open(output, "w") as f:
            json.dump({
                "overall_accuracy": overall,
                "per_category": results
            }, f, indent=2)
        print(f"\n✅ Results saved to: {output}")


