import json

with open('corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED/train_CATEGORY_BASED.json') as f:
    data = json.load(f)

print("="*80)
print("CHECKING 'NONE' and 'NOT RELEVANT' INCONSISTENCIES")
print("="*80)

categories_with_issues = []

# Group by category
from collections import defaultdict
by_category = defaultdict(list)
for item in data:
    cat = item.get('category')
    by_category[cat].append(item)

for category, items in sorted(by_category.items()):
    if not items:
        continue
    
    sample_item = items[0]
    instruction = sample_item.get('instruction', '')
    candidates = sample_item.get('candidates', [])
    
    has_none = 'none' in candidates
    has_normal = 'normal' in candidates
    has_not_relevant = 'not relevant' in candidates
    
    # Check if instruction mentions 'none'/'normal' that aren't in candidates
    mentions_none_or_normal = "'none' or 'normal'" in instruction
    mentions_not_relevant = "'not relevant'" in instruction
    
    issue = False
    issues_list = []
    
    if mentions_none_or_normal and not has_none and not has_normal:
        issue = True
        issues_list.append("Instruction says 'none' or 'normal' but neither in candidates")
    
    if mentions_not_relevant and not has_not_relevant:
        issue = True
        issues_list.append("Instruction says 'not relevant' but not in candidates")
    
    if issue:
        print(f"\n❌ {category.upper()}")
        print(f"   Candidates: {candidates}")
        for iss in issues_list:
            print(f"   - {iss}")
        categories_with_issues.append(category)

print("\n" + "="*80)
print(f"TOTAL CATEGORIES WITH ISSUES: {len(categories_with_issues)}")
if categories_with_issues:
    print("\nCategories that need fixing:")
    for cat in categories_with_issues:
        print(f"  - {cat}")
print("="*80)
