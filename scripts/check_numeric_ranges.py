import json

with open('corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED/train_CATEGORY_BASED.json') as f:
    data = json.load(f)

print("="*80)
print("CHECKING NUMERIC CATEGORY RANGES")
print("="*80)

from collections import defaultdict
by_category = defaultdict(list)
for item in data:
    cat = item.get('category')
    qtype = item.get('question_type')
    if qtype == 'numeric':
        by_category[cat].append(item)

for category, items in sorted(by_category.items()):
    candidates = items[0].get('candidates', [])
    sample_q = items[0].get('question', '')
    
    # Convert to integers for analysis
    try:
        numeric_candidates = sorted([int(c) for c in candidates if c.isdigit()])
        min_val = min(numeric_candidates)
        max_val = max(numeric_candidates)
        num_values = len(numeric_candidates)
        
        print(f"\n{category.upper()}")
        print(f"  Question: {sample_q}")
        print(f"  Current candidates: {candidates}")
        print(f"  Range: {min_val} to {max_val}")
        print(f"  Total values: {num_values}")
        
        # Check for gaps
        full_range = set(range(min_val, max_val + 1))
        available = set(numeric_candidates)
        gaps = full_range - available
        if gaps:
            print(f"  ⚠️  GAPS in range: {sorted(gaps)}")
        
        # Suggest expanded range
        if max_val < 20:
            print(f"  💡 SUGGESTION: Expand to 0-20 for robustness")
        else:
            print(f"  💡 SUGGESTION: Keep broader range")
            
    except:
        print(f"\n{category.upper()}")
        print(f"  Candidates: {candidates}")
        print(f"  ⚠️  Cannot parse as numeric")

print("\n" + "="*80)
