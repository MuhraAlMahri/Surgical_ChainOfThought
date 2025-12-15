#!/usr/bin/env python3
"""
Create Train/Val/Test Splits for TAMSET Datasets

Splits both reordered and non-reordered versions with the same video assignments
to ensure fair comparison.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def split_by_videos(data: Dict, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                    test_ratio: float = 0.15, seed: int = 42) -> Tuple[Dict, Dict, Dict]:
    """
    Split data by videos (not individual QA pairs) to prevent data leakage.
    
    Args:
        data: Dict with video_id -> video_data
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed
    
    Returns:
        (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.seed(seed)
    
    # Get all video IDs
    video_ids = list(data.keys())
    random.shuffle(video_ids)
    
    # Calculate split points
    n_videos = len(video_ids)
    n_train = int(n_videos * train_ratio)
    n_val = int(n_videos * val_ratio)
    
    # Split video IDs
    train_ids = video_ids[:n_train]
    val_ids = video_ids[n_train:n_train + n_val]
    test_ids = video_ids[n_train + n_val:]
    
    # Create split datasets
    train_data = {vid: data[vid] for vid in train_ids}
    val_data = {vid: data[vid] for vid in val_ids}
    test_data = {vid: data[vid] for vid in test_ids}
    
    return train_data, val_data, test_data, (train_ids, val_ids, test_ids)


def main():
    parser = argparse.ArgumentParser(description='Create train/val/test splits for TAMSET')
    parser.add_argument('--reordered', required=True, help='Reordered JSON file')
    parser.add_argument('--non_reordered', required=True, help='Non-reordered JSON file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Val ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading reordered data from: {args.reordered}")
    with open(args.reordered, 'r') as f:
        reordered_data = json.load(f)
    
    print(f"Loading non-reordered data from: {args.non_reordered}")
    with open(args.non_reordered, 'r') as f:
        non_reordered_data = json.load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split reordered data (get video IDs for consistency)
    print(f"\nSplitting datasets ({args.train_ratio}/{args.val_ratio}/{args.test_ratio})...")
    r_train, r_val, r_test, (train_ids, val_ids, test_ids) = split_by_videos(
        reordered_data, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    # Split non-reordered data using SAME video IDs
    nr_train = {vid: non_reordered_data[vid] for vid in train_ids}
    nr_val = {vid: non_reordered_data[vid] for vid in val_ids}
    nr_test = {vid: non_reordered_data[vid] for vid in test_ids}
    
    # Save reordered splits
    print("\nSaving reordered splits...")
    with open(output_dir / 'tamset_reordered_train.json', 'w') as f:
        json.dump(r_train, f, indent=2)
    with open(output_dir / 'tamset_reordered_val.json', 'w') as f:
        json.dump(r_val, f, indent=2)
    with open(output_dir / 'tamset_reordered_test.json', 'w') as f:
        json.dump(r_test, f, indent=2)
    
    # Save non-reordered splits
    print("Saving non-reordered splits...")
    with open(output_dir / 'tamset_non_reordered_train.json', 'w') as f:
        json.dump(nr_train, f, indent=2)
    with open(output_dir / 'tamset_non_reordered_val.json', 'w') as f:
        json.dump(nr_val, f, indent=2)
    with open(output_dir / 'tamset_non_reordered_test.json', 'w') as f:
        json.dump(nr_test, f, indent=2)
    
    # Save video ID splits for reference
    split_info = {
        'train_videos': train_ids,
        'val_videos': val_ids,
        'test_videos': test_ids,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'seed': args.seed
    }
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Print statistics
    def count_qa(data):
        return sum(len(v['qa_pairs']) for v in data.values())
    
    print(f"\n✅ Dataset splits created!")
    print(f"\nReordered splits:")
    print(f"  Train: {len(r_train)} videos, {count_qa(r_train)} QA pairs")
    print(f"  Val:   {len(r_val)} videos, {count_qa(r_val)} QA pairs")
    print(f"  Test:  {len(r_test)} videos, {count_qa(r_test)} QA pairs")
    
    print(f"\nNon-reordered splits:")
    print(f"  Train: {len(nr_train)} videos, {count_qa(nr_train)} QA pairs")
    print(f"  Val:   {len(nr_val)} videos, {count_qa(nr_val)} QA pairs")
    print(f"  Test:  {len(nr_test)} videos, {count_qa(nr_test)} QA pairs")
    
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()

