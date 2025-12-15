#!/usr/bin/env python3
"""
Quick script to evaluate Exp1 without instructions
Converts JSONL to JSON and runs evaluation with --use_question_only flag
"""

import json
import subprocess
import sys
import os
import tempfile

# Paths
BASE_DIR = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments"
SCRIPTS_DIR = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/scripts/evaluation"
TEST_JSONL = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/qlora_experiments/exp1_random/test.jsonl"
MODEL_PATH = f"{BASE_DIR}/models/exp1_random"
IMAGE_ROOT = "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
OUTPUT_FILE = f"{BASE_DIR}/results/exp1_evaluation_no_instruction.json"

# Convert JSONL to JSON
print("Converting JSONL to JSON...")
test_data = []
with open(TEST_JSONL, 'r') as f:
    for line in f:
        test_data.append(json.loads(line))

# Create temporary JSON file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_data, f, indent=2)
    temp_json = f.name

print(f"Converted {len(test_data)} samples")
print(f"Temporary JSON file: {temp_json}")

# Run evaluation
eval_script = f"{SCRIPTS_DIR}/evaluate_exp1.py"
cmd = [
    "python3", eval_script,
    "--model_path", MODEL_PATH,
    "--test_data", temp_json,
    "--image_dir", IMAGE_ROOT,
    "--output", OUTPUT_FILE,
    "--base_model", "Qwen/Qwen3-VL-8B-Instruct",
    "--use_question_only"
]

print("\n" + "="*80)
print("Running evaluation WITHOUT instructions...")
print("="*80)
print(f"Command: {' '.join(cmd)}")
print()

# Run the evaluation
result = subprocess.run(cmd)

# Cleanup
os.unlink(temp_json)

if result.returncode == 0:
    print(f"\n✓ Evaluation complete! Results saved to: {OUTPUT_FILE}")
    
    # Read and display results
    with open(OUTPUT_FILE, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("RESULTS (WITHOUT INSTRUCTIONS)")
    print("="*80)
    print(f"Total samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    
    print(f"\nBy Stage:")
    for stage in sorted(results['by_stage'].keys()):
        stage_data = results['by_stage'][stage]
        print(f"  {stage}: {stage_data['correct']}/{stage_data['total']} ({stage_data['accuracy']:.2f}%)")
    
    print(f"\nBy Question Type:")
    for qtype in sorted(results['by_question_type'].keys()):
        qtype_data = results['by_question_type'][qtype]
        print(f"  {qtype:15s}: {qtype_data['correct']}/{qtype_data['total']} ({qtype_data['accuracy']:.2f}%)")
else:
    print(f"\n✗ Evaluation failed with exit code: {result.returncode}")
    sys.exit(result.returncode)




