import json

with open('corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED/train_CATEGORY_BASED.json') as f:
    data = json.load(f)

color_categories = ['abnormality_color', 'landmark_color']

for category in color_categories:
    answers = [item.get('answer', '') for item in data if item.get('category') == category]
    multi = [a for a in answers if ';' in a]
    single = [a for a in answers if ';' not in a]
    
    print(f"\n{'='*80}")
    print(f"{category.upper()}")
    print(f"{'='*80}")
    print(f"Total: {len(answers)}")
    print(f"Single: {len(single)} ({len(single)/len(answers)*100:.1f}%)")
    print(f"Multi: {len(multi)} ({len(multi)/len(answers)*100:.1f}%)")
    
    if len(multi) > 0:
        print(f"\n❌ MISMATCH: {len(multi)} have multiple answers!")
        print("Examples:")
        for ans in multi[:3]:
            print(f"   - '{ans}'")

print("\n" + "="*80)
print("BOTH color categories need to be MULTI_LABEL!")
print("="*80)
