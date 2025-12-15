#!/usr/bin/env python3
"""
Create a template JSON file with your existing baseline results.
Fill in the numbers you already have, then use this file with evaluate_cot_only.py
"""

import json
from pathlib import Path

# Template with your existing numbers
# Fill in the missing values (LLaVA-Med) and update paths if needed
baseline_results = {
    "qwen3vl_kvasir": {
        "zeroshot": 53.48,
        "zeroshot_correct": 4805,
        "zeroshot_total": 8984,
        "finetuned": 92.79,
        "finetuned_correct": 8336,
        "finetuned_total": 8984
    },
    "qwen3vl_endovis": {
        "zeroshot": 31.12,
        "zeroshot_correct": 742,
        "zeroshot_total": 2384,
        "finetuned": 95.18,
        "finetuned_correct": 2269,
        "finetuned_total": 2384
    },
    "medgemma_kvasir": {
        "zeroshot": 32.05,
        "zeroshot_correct": 2879,
        "zeroshot_total": 8984,
        "finetuned": 91.90,
        "finetuned_correct": 8256,
        "finetuned_total": 8984
    },
    "medgemma_endovis": {
        "zeroshot": 25.08,
        "zeroshot_correct": 598,
        "zeroshot_total": 2384,
        "finetuned": 99.83,
        "finetuned_correct": 2380,
        "finetuned_total": 2384
    },
    "llava_med_kvasir": {
        "zeroshot": None,  # Fill in if you have this
        "zeroshot_correct": None,
        "zeroshot_total": None,
        "finetuned": None,  # Fill in if you have this
        "finetuned_correct": None,
        "finetuned_total": None
    },
    "llava_med_endovis": {
        "zeroshot": None,  # Fill in if you have this
        "zeroshot_correct": None,
        "zeroshot_total": None,
        "finetuned": None,  # Fill in if you have this
        "finetuned_correct": None,
        "finetuned_total": None
    }
}

output_file = Path("results/baseline_results.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(baseline_results, f, indent=2)

print(f"✅ Created baseline results template at: {output_file}")
print("\n📝 Please review and update the file with your actual numbers.")
print("   Then use it with: --baseline-results results/baseline_results.json")







