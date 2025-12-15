#!/usr/bin/env python3
"""
Prepare all 5 experiment datasets from ULTRA_CONDENSED source
All data already has instructions + stage fields embedded
"""
import json
import random
from pathlib import Path
from collections import Counter

# Paths
BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "datasets/kvasir_ULTRA_CONDENSED"
OUTPUT_BASE = BASE_DIR / "datasets/qlora_experiments"
OUTPUT_BASE.mkdir(exist_ok=True)

print("="*80)
print("PREPARING ALL 5 EXPERIMENT DATASETS (QLoRA Setup)")
print("="*80)
print(f"\nSource: {SOURCE_DIR}")
print(f"Output: {OUTPUT_BASE}")

# Load source data
splits = {}
for split in ['train', 'val', 'test']:
    jsonl_file = SOURCE_DIR / f"{split}_CATEGORY_BASED.jsonl"
    print(f"\nLoading {split}...")
    with open(jsonl_file) as f:
        splits[split] = [json.loads(line) for line in f]
    print(f"  Loaded {len(splits[split]):,} samples")

# ============================================================================
# EXP1: Random Baseline (Shuffle to remove clinical ordering)
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 1: Random Baseline")
print("="*80)

exp1_dir = OUTPUT_BASE / "exp1_random"
exp1_dir.mkdir(exist_ok=True)

for split in ['train', 'val', 'test']:
    data = splits[split].copy()
    if split == 'train':
        random.seed(42)
        random.shuffle(data)
        print(f"✓ Shuffled {len(data):,} training samples (seed=42)")
    
    output_file = exp1_dir / f"{split}.jsonl"
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"✓ Saved {split}: {output_file} ({len(data):,} samples)")

# ============================================================================
# EXP2: Qwen Reordered (Keep clinical ordering as-is)
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: Qwen Clinical Reordering")
print("="*80)

exp2_dir = OUTPUT_BASE / "exp2_qwen_reordered"
exp2_dir.mkdir(exist_ok=True)

for split in ['train', 'val', 'test']:
    data = splits[split]  # Keep original order
    
    if split == 'train':
        stage_dist = Counter(d['stage'] for d in data)
        print(f"✓ Preserving clinical ordering:")
        print(f"  Stage 1 (Initial Assessment): {stage_dist[1]:,} ({stage_dist[1]/len(data)*100:.1f}%)")
        print(f"  Stage 2 (Findings): {stage_dist[2]:,} ({stage_dist[2]/len(data)*100:.1f}%)")
        print(f"  Stage 3 (Clinical Context): {stage_dist.get(3, 0):,} ({stage_dist.get(3, 0)/len(data)*100:.1f}%)")
    
    output_file = exp2_dir / f"{split}.jsonl"
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"✓ Saved {split}: {output_file} ({len(data):,} samples)")

# ============================================================================
# EXP3 & EXP4: Split by Stage (3 datasets each)
# ============================================================================
for exp_name, exp_num in [("exp3_cxrtrek_sequential", 3), ("exp4_curriculum", 4)]:
    print("\n" + "="*80)
    print(f"EXPERIMENT {exp_num}: {exp_name.upper().replace('_', ' ')}")
    print("="*80)
    
    exp_dir = OUTPUT_BASE / exp_name
    exp_dir.mkdir(exist_ok=True)
    
    for stage_num in [1, 2, 3]:
        stage_dir = exp_dir / f"stage{stage_num}"
        stage_dir.mkdir(exist_ok=True)
        
        print(f"\n--- Stage {stage_num} ---")
        
        for split in ['train', 'val', 'test']:
            # Filter by stage
            stage_data = [item for item in splits[split] if item.get('stage') == stage_num]
            
            output_file = stage_dir / f"{split}.jsonl"
            with open(output_file, 'w') as f:
                for item in stage_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"  {split}: {len(stage_data):,} samples → {output_file}")

# ============================================================================
# EXP5: No dataset needed (uses Exp1 or Exp2 model)
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT 5: Sequential CoT (Inference-Time Cascading)")
print("="*80)
print("✓ No separate dataset needed - uses Exp1 or Exp2 trained model")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("✓ ALL DATASETS PREPARED!")
print("="*80)

print(f"\nOutput structure:")
print(f"  {OUTPUT_BASE}/")
print(f"    exp1_random/")
print(f"      train.jsonl  ({len(splits['train']):,} shuffled samples)")
print(f"      val.jsonl    ({len(splits['val']):,} samples)")
print(f"      test.jsonl   ({len(splits['test']):,} samples)")
print(f"    exp2_qwen_reordered/")
print(f"      train.jsonl  ({len(splits['train']):,} clinically ordered)")
print(f"      val.jsonl    ({len(splits['val']):,} samples)")
print(f"      test.jsonl   ({len(splits['test']):,} samples)")
print(f"    exp3_cxrtrek_sequential/")
print(f"      stage1/ stage2/ stage3/  (split by stage)")
print(f"    exp4_curriculum/")
print(f"      stage1/ stage2/ stage3/  (split by stage)")
print(f"    exp5_sequential_cot/")
print(f"      (inference-only, no training data)")

print(f"\n✓ Ready for QLoRA training with Qwen3-VL-8B-Instruct!")
print("="*80)

