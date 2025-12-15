import json

print("="*80)
print("VERIFYING CONSISTENCY ACROSS SPLITS")
print("="*80)

splits = ['train', 'val', 'test']
categories_per_split = {}
instruction_samples = {}

for split in splits:
    file = f'corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED/{split}_CATEGORY_BASED.json'
    with open(file) as f:
        data = json.load(f)
    
    # Get categories
    categories = set(item.get('category') for item in data if item.get('category'))
    categories_per_split[split] = categories
    
    # Get sample instruction per category
    if split == 'train':
        for item in data:
            cat = item.get('category')
            if cat and cat not in instruction_samples:
                instruction_samples[cat] = item.get('instruction', '')[:200]  # First 200 chars
    
    print(f"\n{split.upper()}:")
    print(f"  Total samples: {len(data)}")
    print(f"  Categories: {len(categories)}")
    print(f"  Categories: {sorted(categories)}")

# Check consistency
print("\n" + "="*80)
print("CONSISTENCY CHECK")
print("="*80)

train_cats = categories_per_split['train']
val_cats = categories_per_split['val']
test_cats = categories_per_split['test']

if train_cats == val_cats == test_cats:
    print("\n✅ SUCCESS: All splits use exactly the same categories!")
    print(f"\nTotal categories: {len(train_cats)}")
    print(f"Categories: {sorted(train_cats)}")
else:
    print("\n❌ ERROR: Inconsistent categories across splits!")
    print(f"\nTrain only: {train_cats - val_cats - test_cats}")
    print(f"Val only: {val_cats - train_cats - test_cats}")
    print(f"Test only: {test_cats - train_cats - val_cats}")

print("\n" + "="*80)
