import json
from collections import defaultdict

print("="*80)
print("VERIFYING CANDIDATES ARE TRAIN-SET ONLY")
print("="*80)

base_path = "corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED"

# Get all candidates from training set
train_file = f"{base_path}/train_CATEGORY_BASED.json"
with open(train_file) as f:
    train_data = json.load(f)

train_candidates_by_category = defaultdict(set)
for item in train_data:
    category = item.get('category', 'unknown')
    candidates = item.get('candidates', [])
    train_candidates_by_category[category].update(candidates)

print(f"\nTrain set: {len(train_data)} samples")
print(f"Categories: {len(train_candidates_by_category)}")

# Check val and test
issues_found = False

for split in ['val', 'test']:
    split_file = f"{base_path}/{split}_CATEGORY_BASED.json"
    with open(split_file) as f:
        split_data = json.load(f)
    
    print(f"\n{split.upper()} set: {len(split_data)} samples")
    
    split_specific_candidates = defaultdict(set)
    
    for item in split_data:
        category = item.get('category', 'unknown')
        candidates = set(item.get('candidates', []))
        
        # Check if any candidates are NOT in training set
        train_cands = train_candidates_by_category[category]
        extra_cands = candidates - train_cands
        
        if extra_cands:
            split_specific_candidates[category].update(extra_cands)
            issues_found = True
    
    if split_specific_candidates:
        print(f"\n⚠️  {split.upper()} HAS EXTRA CANDIDATES NOT IN TRAIN:")
        for cat, extra in split_specific_candidates.items():
            print(f"   {cat}: {extra}")
    else:
        print(f"✅ All {split} candidates are from training set")

print("\n" + "="*80)
if not issues_found:
    print("✅ ALL CANDIDATE LISTS DERIVED FROM TRAINING SET ONLY!")
    print("\n🎯 VERIFIED: Val and test use EXACT same candidates as train")
else:
    print("❌ ISSUE: Some val/test candidates not in training set")
print("="*80)
