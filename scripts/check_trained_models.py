#!/usr/bin/env python3
"""
Check which multi-head CoT models have been trained.
"""
import os
from pathlib import Path

models = ["qwen3vl", "medgemma", "llava_med"]
datasets = ["kvasir", "endovis"]

print("="*80)
print("TRAINED MULTI-HEAD COT MODELS STATUS")
print("="*80)

trained_count = 0
total_count = len(models) * len(datasets)

for model in models:
    for dataset in datasets:
        # Check for checkpoints in results/multihead_cot/
        results_dir = Path(f"results/multihead_cot")
        
        # Look for checkpoints matching pattern: {model}_{dataset}_cot_*
        pattern = f"{model}_{dataset}_cot_*"
        found_checkpoints = []
        
        if results_dir.exists():
            for checkpoint_dir in results_dir.glob(pattern):
                # Check for checkpoint files
                checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                if checkpoint_files:
                    # Get latest checkpoint
                    latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                    size_mb = latest.stat().st_size / (1024 * 1024)
                    found_checkpoints.append((checkpoint_dir, latest, size_mb))
        
        if found_checkpoints:
            # Use the most recent checkpoint directory
            checkpoint_dir, latest_file, size_mb = max(found_checkpoints, key=lambda x: x[2].stat().st_mtime)
            print(f"✓ {model:12} + {dataset:8}: TRAINED ({size_mb:.1f} MB)")
            print(f"  └─ Latest: {latest_file.name} in {checkpoint_dir.name}")
            trained_count += 1
        else:
            print(f"✗ {model:12} + {dataset:8}: NOT TRAINED")

print("="*80)
print(f"\nSummary: {trained_count}/{total_count} models trained")
print("="*80)

if trained_count < total_count:
    print("\nTo train missing models, run:")
    print("  bash scripts/run_all_cot_training.sh")
    print("="*80)
else:
    print("\n✅ All models trained! Ready for evaluation.")
    print("  Run: bash scripts/evaluate_cot_all.sh")
    print("="*80)





