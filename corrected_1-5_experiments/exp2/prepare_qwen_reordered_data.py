"""
Prepare Qwen reordered data for Exp2:
1. Load existing train/val/test splits (already image-level split)
2. Apply ULTRA_CONDENSED general instruction
3. Save as exp2 dataset
"""
import json
from pathlib import Path
from collections import Counter

# Paths
base_dir = Path(__file__).parent.parent
data_dir = base_dir / "datasets/kvasir_raw_6500_image_level_70_15_15"
ultra_condensed_file = base_dir / "datasets/kvasir_ULTRA_CONDENSED/INSTRUCTIONS_PER_CATEGORY.txt"
output_dir = base_dir / "datasets/kvasir_qwen_reordered_ultra_condensed"
output_dir.mkdir(exist_ok=True)

print("="*80)
print("PREPARING EXP2: QWEN REORDERED DATA WITH ULTRA_CONDENSED INSTRUCTIONS")
print("="*80)

# Parse ULTRA_CONDENSED instructions - extract general instruction
print(f"\nParsing ULTRA_CONDENSED instructions from:")
print(f"  {ultra_condensed_file}")

instructions_text = open(ultra_condensed_file).read()
general_start = instructions_text.find("You are a surgical image analysis assistant")
general_end = instructions_text.find("Character count:", general_start)
general_block = instructions_text[general_start:general_end]

# Extract just the instruction text
lines = general_block.split('\n')
instruction_lines = []
for line in lines:
    if line.strip() and not line.startswith('-') and 'You are' in line:
        instruction_lines.append(line.strip())
        break

for line in lines[1:]:
    stripped = line.strip()
    if stripped and (stripped.startswith('-') or 'Select' in stripped or 'Output format' in stripped):
        instruction_lines.append(stripped)
    if 'Output format' in stripped:
        break

general_instruction = '\n'.join(instruction_lines)
print(f"✓ General instruction: {len(general_instruction)} chars")
print(f"\n{general_instruction}\n")

# Process each split
for split_name in ['train', 'val', 'test']:
    input_file = data_dir / f"{split_name}.json"
    output_file = output_dir / f"{split_name}.json"
    
    print(f"\nProcessing {split_name}...")
    with open(input_file) as f:
        data = json.load(f)
    
    # Apply instruction format
    enhanced_data = []
    for item in data:
        enhanced_item = {
            'image': item['image_filename'],
            'question': item['question'],
            'answer': item['answer'],
            'stage': item['stage'],
            'instruction': f"{general_instruction}\n\nQuestion: {item['question']}\nAnswer:"
        }
        enhanced_data.append(enhanced_item)
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    # Stats
    stage_dist = Counter(d['stage'] for d in enhanced_data)
    print(f"  {split_name}: {len(enhanced_data):,} QA pairs")
    print(f"    Stage 1 (Initial Assessment): {stage_dist[1]:,} ({stage_dist[1]/len(enhanced_data)*100:.1f}%)")
    print(f"    Stage 2 (Findings): {stage_dist[2]:,} ({stage_dist[2]/len(enhanced_data)*100:.1f}%)")
    print(f"    Stage 3 (Clinical Context): {stage_dist[3]:,} ({stage_dist[3]/len(enhanced_data)*100:.1f}%)")
    print(f"  ✓ Saved to: {output_file}")

print("\n" + "="*80)
print("✓ DATA PREPARATION COMPLETE!")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print(f"\nEXP2 Strategy:")
print(f"  - Train on Qwen-reordered QA pairs (stages 1, 2, 3 mixed in clinical order)")
print(f"  - Use ULTRA_CONDENSED general instruction for conciseness")
print(f"  - Model: Qwen3-VL-8B-Instruct")
print(f"  - Resolution: 768×768 letterbox")
print(f"  - Hardware: 2 GPUs")
print(f"  - Expected time: ~14 hours")
print("="*80)
