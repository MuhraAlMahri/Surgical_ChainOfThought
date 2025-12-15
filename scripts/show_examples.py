import json

data_file = 'corrected_1-5_experiments/datasets/kvasir_REVISED_test/train_REVISED.json'

with open(data_file) as f:
    data = json.load(f)

print("="*80)
print("EXAMPLE REVISED INSTRUCTIONS")
print("="*80)

# Show one example of each type
question_types = ['multi_label', 'single_choice', 'binary', 'numeric', 'open_constrained']
shown_types = set()

for item in data:
    qtype = item.get('question_type')
    if qtype in question_types and qtype not in shown_types:
        shown_types.add(qtype)
        
        print(f"\n{'='*80}")
        print(f"EXAMPLE {len(shown_types)}: {qtype.upper()}")
        print(f"{'='*80}")
        print(f"\nOriginal Question: {item['question']}")
        print(f"Ground Truth Answer: {item['answer']}")
        print(f"\nRevised Instruction:")
        print("-"*80)
        print(item['instruction'])
        print("-"*80)
        
        if item.get('candidates'):
            print(f"\nCandidates Provided: {item['candidates']}")
        
        print(f"\nOutput Format: {item.get('output_format')}")
        
        if len(shown_types) >= 5:
            break

print("\n" + "="*80)
print("END OF EXAMPLES")
print("="*80)
print(f"\nShowed {len(shown_types)} different question types")
print("\nThese examples show:")
print("  1. Original question")
print("  2. Ground truth answer")
print("  3. Revised instruction template")
print("  4. Candidate lists (for close-ended)")
print("  5. Expected output format")
print("\nReview these examples to ensure templates are correct")
print("before sending to advisor!")
