#!/usr/bin/env python3
"""
Create proper 80/20 train/test split for baseline and reordered experiments.
Uses the SAME image-based split as CXRTrek to ensure consistency across all experiments.
"""

import json
import os
import sys
from pathlib import Path

def flatten_qa_pairs(data_dict, include_stage_prefix=False):
    """
    Flatten QA pairs from the structured format.
    
    Args:
        data_dict: Dictionary with image_id as keys
        include_stage_prefix: If True, prefix questions with stage names
    
    Returns:
        List of {image_id, image_filename, question, answer} dicts
    """
    qa_pairs = []
    
    for image_id, item in data_dict.items():
        image_filename = item.get('image_filename', f"{image_id}.jpg")
        
        # Extract from clinical_flow_stages
        stages = item.get('clinical_flow_stages', {})
        
        for stage_name, stage_content in stages.items():
            # Handle list format (Stage 1)
            if isinstance(stage_content, list):
                for qa in stage_content:
                    if include_stage_prefix:
                        question = f"[{stage_name}] {qa['question']}"
                    else:
                        question = qa['question']
                    
                    qa_pairs.append({
                        'image_id': image_id,
                        'image_filename': image_filename,
                        'question': question,
                        'answer': qa['answer']
                    })
            
            # Handle dict format (Stage 2 with subcategories)
            elif isinstance(stage_content, dict):
                for category, qa_list in stage_content.items():
                    if isinstance(qa_list, list):
                        for qa in qa_list:
                            if include_stage_prefix:
                                question = f"[{stage_name}] {qa['question']}"
                            else:
                                question = qa['question']
                            
                            qa_pairs.append({
                                'image_id': image_id,
                                'image_filename': image_filename,
                                'question': question,
                                'answer': qa['answer']
                            })
    
    return qa_pairs


def create_splits(input_file, output_dir, split_metadata_file, include_stage_prefix=False):
    """
    Create train/test splits using the proper image-based split from CXRTrek.
    
    Args:
        input_file: Path to integrated surgical dataset JSON
        output_dir: Where to save the split files
        split_metadata_file: Path to split_metadata.json from CXRTrek (has train/test image lists)
        include_stage_prefix: Whether to include stage prefixes in questions
    """
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loading image ID lists from: {split_metadata_file}")
    with open(split_metadata_file, 'r') as f:
        metadata = json.load(f)
    
    train_images = set(metadata['train_images'])
    test_images = set(metadata['test_images'])
    
    print(f"\nSplit metadata:")
    print(f"  Train images: {len(train_images)}")
    print(f"  Test images: {len(test_images)}")
    
    # Flatten all QA pairs
    print(f"\nFlattening QA pairs...")
    all_qa = flatten_qa_pairs(data, include_stage_prefix=include_stage_prefix)
    print(f"  Total QA pairs: {len(all_qa)}")
    
    # Split by image ID
    train_qa = [qa for qa in all_qa if qa['image_id'] in train_images]
    test_qa = [qa for qa in all_qa if qa['image_id'] in test_images]
    
    print(f"\nSplit results:")
    print(f"  Train QA pairs: {len(train_qa)} ({100*len(train_qa)/len(all_qa):.1f}%)")
    print(f"  Test QA pairs: {len(test_qa)} ({100*len(test_qa)/len(all_qa):.1f}%)")
    
    # Verify no overlap
    train_img_check = set(qa['image_id'] for qa in train_qa)
    test_img_check = set(qa['image_id'] for qa in test_qa)
    overlap = train_img_check & test_img_check
    
    if overlap:
        print(f"\n⚠️  WARNING: Found {len(overlap)} overlapping images!")
    else:
        print(f"\n✅ VERIFIED: Zero image overlap!")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.json')
    test_file = os.path.join(output_dir, 'test.json')
    
    with open(train_file, 'w') as f:
        json.dump(train_qa, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_qa, f, indent=2)
    
    print(f"\n✅ Files created:")
    print(f"   {train_file}")
    print(f"   {test_file}")
    
    return train_qa, test_qa


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    
    # Use the SAME image split from CXRTrek (image ID lists)
    split_metadata = base_dir / "experiments/cxrtrek_curriculum_learning/data/proper_splits_80_20/image_ids.json"
    
    # Input data
    integrated_train = base_dir / "datasets/integrated_surgical_dataset/integrated_surgical_train.json"
    integrated_val = base_dir / "datasets/integrated_surgical_dataset/integrated_surgical_val.json"
    
    if not split_metadata.exists():
        print(f"❌ ERROR: Split metadata not found: {split_metadata}")
        print("Please run the CXRTrek splitting script first!")
        sys.exit(1)
    
    if not integrated_train.exists():
        print(f"❌ ERROR: Training data not found: {integrated_train}")
        sys.exit(1)
    
    # Merge train and val (we'll re-split properly by image ID)
    print("Loading and merging integrated surgical dataset...")
    with open(integrated_train, 'r') as f:
        train_data = json.load(f)
    
    if integrated_val.exists():
        with open(integrated_val, 'r') as f:
            val_data = json.load(f)
        # Merge
        all_data = {**train_data, **val_data}
        print(f"  Merged {len(train_data)} + {len(val_data)} = {len(all_data)} images")
    else:
        all_data = train_data
        print(f"  Using {len(all_data)} images from train file")
    
    # Create baseline split (no stage prefix)
    print("\n" + "="*80)
    print("CREATING BASELINE SPLIT (no stage prefix, random order)")
    print("="*80)
    baseline_dir = base_dir / "datasets/kvasir_baseline_proper_80_20"
    create_splits(
        input_file=integrated_train,  # We'll use the merged data
        output_dir=baseline_dir,
        split_metadata_file=split_metadata,
        include_stage_prefix=False
    )
    
    # For this to work with merged data, let me save merged data first
    merged_file = base_dir / "datasets/integrated_surgical_dataset/integrated_surgical_merged.json"
    with open(merged_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved merged data: {merged_file}")
    
    # Now create splits
    print("\n" + "="*80)
    print("CREATING BASELINE SPLIT (no stage prefix)")
    print("="*80)
    baseline_dir = base_dir / "datasets/kvasir_baseline_proper_80_20"
    create_splits(
        input_file=merged_file,
        output_dir=baseline_dir,
        split_metadata_file=split_metadata,
        include_stage_prefix=False
    )
    
    # Create reordered split (with stage prefix to preserve ordering info)
    print("\n" + "="*80)
    print("CREATING REORDERED SPLIT (with stage prefix)")
    print("="*80)
    reordered_dir = base_dir / "datasets/kvasir_reordered_proper_80_20"
    create_splits(
        input_file=merged_file,
        output_dir=reordered_dir,
        split_metadata_file=split_metadata,
        include_stage_prefix=True
    )
    
    print("\n" + "="*80)
    print("✅ ALL KVASIR SPLITS CREATED SUCCESSFULLY!")
    print("="*80)
    print("\nAll experiments now use the SAME 80/20 image-based split:")
    print(f"  - CXRTrek Sequential: experiments/cxrtrek_curriculum_learning/data/proper_splits_80_20/")
    print(f"  - CXRTrek Curriculum: (uses same split)")
    print(f"  - Baseline: {baseline_dir}/")
    print(f"  - Reordered: {reordered_dir}/")


if __name__ == "__main__":
    main()
