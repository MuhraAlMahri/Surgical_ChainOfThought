#!/usr/bin/env python3
"""
Create proper 80/20 train/test split for TAMSET experiments.
Splits by VIDEO ID to prevent data leakage (no video appears in both train and test).
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List

def flatten_tamset_qa(data_dict, include_stage_prefix=False):
    """
    Flatten TAMSET QA pairs from the structured format.
    
    Args:
        data_dict: Dictionary with video_id as keys
        include_stage_prefix: If True, prefix questions with stage names
    
    Returns:
        List of {video_id, timestamp, question, answer, stage} dicts
    """
    qa_pairs = []
    
    for video_id, video_data in data_dict.items():
        stages = video_data.get('clinical_flow_stages', {})
        
        for stage_name, qa_list in stages.items():
            # Ensure qa_list is a list
            if not isinstance(qa_list, list):
                print(f"Warning: Skipping non-list stage data for {video_id}, stage {stage_name}")
                continue
            
            for qa in qa_list:
                # Ensure qa is a dict
                if not isinstance(qa, dict):
                    continue
                
                if include_stage_prefix:
                    question = f"[{stage_name}] {qa.get('question', '')}"
                else:
                    question = qa.get('question', '')
                
                qa_pairs.append({
                    'video_id': video_id,
                    'timestamp': qa.get('timestamp', 0.0),
                    'question': question,
                    'answer': qa.get('answer', ''),
                    'stage': stage_name,
                    'label': qa.get('label', '')
                })
    
    return qa_pairs


def create_video_based_split(input_file, output_dir, train_ratio=0.8, seed=42, include_stage_prefix=False):
    """
    Create train/test splits by VIDEO ID (not QA pairs).
    
    Args:
        input_file: Path to TAMSET JSON file
        output_dir: Where to save the split files
        train_ratio: Ratio for train split (default 0.8 for 80/20)
        seed: Random seed for reproducibility
        include_stage_prefix: Whether to include stage prefixes in questions
    """
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get all video IDs
    video_ids = list(data.keys())
    print(f"\nTotal videos: {len(video_ids)}")
    print(f"Videos: {video_ids}")
    
    # Split videos
    random.seed(seed)
    random.shuffle(video_ids)
    
    split_idx = int(len(video_ids) * train_ratio)
    train_video_ids = video_ids[:split_idx]
    test_video_ids = video_ids[split_idx:]
    
    print(f"\nSplit by video ID (seed={seed}):")
    print(f"  Train videos: {len(train_video_ids)} ({100*len(train_video_ids)/len(video_ids):.1f}%)")
    print(f"    {train_video_ids}")
    print(f"  Test videos: {len(test_video_ids)} ({100*len(test_video_ids)/len(video_ids):.1f}%)")
    print(f"    {test_video_ids}")
    
    # Flatten QA pairs
    print(f"\nFlattening QA pairs...")
    all_qa = flatten_tamset_qa(data, include_stage_prefix=include_stage_prefix)
    print(f"  Total QA pairs: {len(all_qa)}")
    
    # Split by video ID
    train_qa = [qa for qa in all_qa if qa['video_id'] in train_video_ids]
    test_qa = [qa for qa in all_qa if qa['video_id'] in test_video_ids]
    
    print(f"\nSplit results:")
    print(f"  Train QA pairs: {len(train_qa)} ({100*len(train_qa)/len(all_qa):.1f}%)")
    print(f"  Test QA pairs: {len(test_qa)} ({100*len(test_qa)/len(all_qa):.1f}%)")
    
    # Verify no overlap
    train_vids_check = set(qa['video_id'] for qa in train_qa)
    test_vids_check = set(qa['video_id'] for qa in test_qa)
    overlap = train_vids_check & test_vids_check
    
    if overlap:
        print(f"\n⚠️  WARNING: Found {len(overlap)} overlapping videos!")
        print(f"    {overlap}")
    else:
        print(f"\n✅ VERIFIED: Zero video overlap!")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.json')
    test_file = os.path.join(output_dir, 'test.json')
    
    with open(train_file, 'w') as f:
        json.dump(train_qa, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_qa, f, indent=2)
    
    # Save metadata
    metadata = {
        'split_method': 'video_based',
        'train_test_ratio': f'{int(train_ratio*100)}/{int((1-train_ratio)*100)}',
        'random_seed': seed,
        'total_videos': len(video_ids),
        'train_videos': train_video_ids,
        'test_videos': test_video_ids,
        'video_overlap': len(overlap),
        'total_qa_pairs': len(all_qa),
        'train_qa_pairs': len(train_qa),
        'test_qa_pairs': len(test_qa)
    }
    
    metadata_file = os.path.join(output_dir, 'split_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Files created:")
    print(f"   {train_file}")
    print(f"   {test_file}")
    print(f"   {metadata_file}")
    
    return train_qa, test_qa, metadata


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    
    # Input data
    tamset_file = base_dir / "datasets/temset/tamset_qa_qwen3_5videos_COMPLETE.json"
    
    if not tamset_file.exists():
        print(f"❌ ERROR: TAMSET file not found: {tamset_file}")
        sys.exit(1)
    
    # Create baseline split (no stage prefix)
    print("\n" + "="*80)
    print("CREATING TAMSET BASELINE SPLIT (no stage prefix)")
    print("="*80)
    baseline_dir = base_dir / "datasets/tamset_baseline_proper_80_20"
    create_video_based_split(
        input_file=tamset_file,
        output_dir=baseline_dir,
        train_ratio=0.8,
        seed=42,
        include_stage_prefix=False
    )
    
    # Create reordered split (with stage prefix to preserve clinical flow ordering)
    print("\n" + "="*80)
    print("CREATING TAMSET REORDERED SPLIT (with stage prefix)")
    print("="*80)
    reordered_dir = base_dir / "datasets/tamset_reordered_proper_80_20"
    create_video_based_split(
        input_file=tamset_file,
        output_dir=reordered_dir,
        train_ratio=0.8,
        seed=42,
        include_stage_prefix=True
    )
    
    print("\n" + "="*80)
    print("✅ ALL TAMSET SPLITS CREATED SUCCESSFULLY!")
    print("="*80)
    print("\nTAMSET experiments now use proper 80/20 video-based split:")
    print(f"  - TAMSET Baseline: {baseline_dir}/")
    print(f"  - TAMSET Reordered: {reordered_dir}/")
    print("\n⚠️  Note: Only 5 videos total, so split is 4 train / 1 test videos")


if __name__ == "__main__":
    main()
