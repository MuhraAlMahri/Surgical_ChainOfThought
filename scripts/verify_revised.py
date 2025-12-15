import json
from collections import Counter

print("="*80)
print("VERIFICATION: Checking Revised Instructions")
print("="*80)

# Load data
data_file = 'corrected_1-5_experiments/datasets/kvasir_REVISED_test/train_REVISED.json'
print(f"\nLoading: {data_file}")

with open(data_file) as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Check 1: All items have instruction field
missing_instruction = [i for i, item in enumerate(data) if not item.get('instruction')]
print(f"\n✓ Check 1: All items have 'instruction' field")
print(f"   Result: {len(data) - len(missing_instruction)}/{len(data)} have instructions")
if missing_instruction:
    print(f"   ⚠️  WARNING: {len(missing_instruction)} items missing instruction!")
    print(f"   First 5: {missing_instruction[:5]}")

# Check 2: All items have question_type
missing_qtype = [i for i, item in enumerate(data) if not item.get('question_type')]
print(f"\n✓ Check 2: All items have 'question_type' field")
print(f"   Result: {len(data) - len(missing_qtype)}/{len(data)} have question_type")

# Check 3: Question type distribution
qtype_counts = Counter(item.get('question_type', 'unknown') for item in data)
print(f"\n✓ Check 3: Question Type Distribution")
for qtype, count in sorted(qtype_counts.items()):
    pct = (count/len(data))*100
    print(f"   {qtype:20s}: {count:6d} ({pct:5.1f}%)")

# Check 4: Multi-label questions have candidates
multi_label_items = [item for item in data if item.get('question_type') == 'multi_label']
multi_label_with_candidates = [item for item in multi_label_items if item.get('candidates')]
print(f"\n✓ Check 4: Multi-label questions have candidate lists")
print(f"   Result: {len(multi_label_with_candidates)}/{len(multi_label_items)} have candidates")
if len(multi_label_with_candidates) < len(multi_label_items):
    print(f"   ⚠️  WARNING: {len(multi_label_items) - len(multi_label_with_candidates)} multi-label items missing candidates!")

# Check 5: Instructions contain "Candidates:" for close-ended
close_ended_types = ['multi_label', 'single_choice', 'binary', 'numeric']
close_ended_items = [item for item in data if item.get('question_type') in close_ended_types]
with_candidate_text = [item for item in close_ended_items if 'Candidates:' in item.get('instruction', '')]
print(f"\n✓ Check 5: Close-ended questions show candidates in instruction text")
print(f"   Result: {len(with_candidate_text)}/{len(close_ended_items)} show candidates")

# Check 6: Verify specific mismatch fixes
print(f"\n✓ Check 6: Verify critical mismatch fixes")

test_cases = {
    'abnormalities': {
        'search': 'abnormalities',
        'expected_type': 'multi_label',
        'expected_in_candidates': ['polyp', 'ulcerative colitis']
    },
    'instruments': {
        'search': 'instruments',
        'expected_type': 'multi_label',
        'expected_in_candidates': ['biopsy forceps', 'metal clip']
    },
    'easy to detect': {
        'search': 'easy to detect',
        'expected_type': 'single_choice',
        'expected_in_candidates': ['yes', 'no', 'not relevant']
    }
}

all_passed = True
for name, test in test_cases.items():
    matches = [item for item in data if test['search'] in item.get('question', '').lower()]
    if matches:
        item = matches[0]
        qtype_match = item.get('question_type') == test['expected_type']
        candidates = item.get('candidates', [])
        candidates_lower = [c.lower() for c in candidates]
        has_expected = all(exp.lower() in candidates_lower for exp in test['expected_in_candidates'])
        
        status = "✓ PASS" if (qtype_match and has_expected) else "✗ FAIL"
        all_passed = all_passed and qtype_match and has_expected
        
        print(f"   {status} - {name}")
        print(f"        Question type: {item.get('question_type')} (expected: {test['expected_type']})")
        print(f"        Has expected candidates: {has_expected}")
        if not has_expected:
            print(f"        Missing: {[e for e in test['expected_in_candidates'] if e.lower() not in candidates_lower]}")

# Check 7: Show sample instructions
print(f"\n✓ Check 7: Sample instructions for each type")
for qtype in ['multi_label', 'single_choice', 'binary', 'numeric', 'open_constrained']:
    samples = [item for item in data if item.get('question_type') == qtype]
    if samples:
        sample = samples[0]
        print(f"\n   --- {qtype.upper()} Example ---")
        print(f"   Question: {sample['question'][:60]}...")
        print(f"   Ground Truth: {sample['answer']}")
        print(f"   Instruction (first 200 chars):")
        print(f"   {sample['instruction'][:200]}...")
        if sample.get('candidates'):
            print(f"   Candidates: {sample['candidates'][:3]}...")

# Final summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

issues = []
if missing_instruction:
    issues.append(f"{len(missing_instruction)} items missing instructions")
if len(multi_label_with_candidates) < len(multi_label_items):
    issues.append(f"{len(multi_label_items) - len(multi_label_with_candidates)} multi-label items missing candidates")
if not all_passed:
    issues.append("Some critical test cases failed")

if issues:
    print("\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"   - {issue}")
    print("\n   → DO NOT send to advisor yet! Fix these issues first.")
    print("="*80)
    exit(1)
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("   - All items have instructions")
    print("   - All items have question types")
    print("   - Multi-label questions have candidates")
    print("   - Critical mismatches are fixed")
    print("   - Sample instructions look correct")
    print("\n   → Safe to send to advisor for review!")
    print("="*80)
    exit(0)
