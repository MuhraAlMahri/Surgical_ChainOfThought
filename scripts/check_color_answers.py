import json

with open('corrected_1-5_experiments/datasets/kvasir_CATEGORY_BASED/train_CATEGORY_BASED.json') as f:
    data = json.load(f)

# Get all color-related answers
color_answers = []
for item in data:
    if item.get('category') == 'abnormality_color':
        answer = item.get('answer', '')
        color_answers.append(answer)

# Check how many have semicolons (multiple colors)
multi_color = [a for a in color_answers if ';' in a]
single_color = [a for a in color_answers if ';' not in a]

print("="*80)
print("ABNORMALITY_COLOR GROUND TRUTH ANALYSIS")
print("="*80)
print(f"\nTotal color questions in training: {len(color_answers)}")
print(f"Single color answers: {len(single_color)} ({len(single_color)/len(color_answers)*100:.1f}%)")
print(f"Multiple color answers: {len(multi_color)} ({len(multi_color)/len(color_answers)*100:.1f}%)")

print(f"\n❌ CRITICAL ISSUE:")
if len(multi_color) > 0:
    print(f"   {len(multi_color)} samples have MULTIPLE colors in ground truth")
    print(f"   But instruction says 'Select ONE option'")
    print(f"   This is INSTRUCTION-LABEL MISMATCH!")
    print(f"\nExamples of multi-color ground truth:")
    for i, answer in enumerate(multi_color[:5], 1):
        print(f"   {i}. '{answer}'")

print("\n" + "="*80)
