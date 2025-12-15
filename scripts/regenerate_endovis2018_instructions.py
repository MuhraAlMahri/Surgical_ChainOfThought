#!/usr/bin/env python3
"""
Regenerate EndoVis2018 instructions from corrected training data.

This script:
1. Converts JSONL files to JSON format (temporary)
2. Runs create_category_based_instructions.py
3. Cleans up temporary files
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

def jsonl_to_json(jsonl_file: Path, json_file: Path):
    """Convert JSONL file to JSON array."""
    print(f"Converting {jsonl_file.name} to JSON...")
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✓ Converted {len(data)} samples")
    return json_file

def main():
    dataset_dir = BASE_DIR / "corrected_1-5_experiments/datasets/endovis2018_vqa"
    temp_dir = dataset_dir / "temp_json"
    temp_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("REGENERATING ENDOVIS2018 INSTRUCTIONS FROM CORRECTED DATA")
    print("="*80)
    print(f"Dataset directory: {dataset_dir}")
    print()
    
    # Convert JSONL to JSON
    train_jsonl = dataset_dir / "train.jsonl"
    val_jsonl = dataset_dir / "validation.jsonl"
    test_jsonl = dataset_dir / "test.jsonl"
    
    train_json = temp_dir / "train.json"
    val_json = temp_dir / "validation.json"
    test_json = temp_dir / "test.json"
    
    print("Step 1: Converting JSONL to JSON...")
    jsonl_to_json(train_jsonl, train_json)
    jsonl_to_json(val_jsonl, val_json)
    jsonl_to_json(test_jsonl, test_json)
    print()
    
    # Run instruction generation
    print("Step 2: Generating category-based instructions...")
    script_path = BASE_DIR / "scripts/create_category_based_instructions.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--train_file", str(train_json),
        "--val_file", str(val_json),
        "--test_file", str(test_json),
        "--output_dir", str(dataset_dir)
    ]
    
    result = subprocess.run(cmd, cwd=BASE_DIR)
    
    if result.returncode != 0:
        print(f"❌ Error running instruction generation script")
        sys.exit(1)
    
    print()
    print("Step 3: Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir)
    print(f"  ✓ Removed {temp_dir}")
    
    print()
    print("="*80)
    print("✓ INSTRUCTION REGENERATION COMPLETE!")
    print("="*80)
    print(f"\nUpdated files:")
    print(f"  - {dataset_dir}/INSTRUCTIONS_PER_CATEGORY.txt")
    print(f"  - {dataset_dir}/train_CATEGORY_BASED.json")
    print(f"  - {dataset_dir}/val_CATEGORY_BASED.json")
    print(f"  - {dataset_dir}/test_CATEGORY_BASED.json")
    print()

if __name__ == "__main__":
    main()
















