#!/usr/bin/env python3
"""
Create proper image-based train/test split for Kvasir-VQA data.

This script fixes the critical data leakage issue where the same images
appeared in both train and test sets. Instead of randomly splitting QA pairs,
we now split by unique image IDs, ensuring zero image overlap between sets.

Author: Fixed data split for scientific validity
Date: October 2025
"""

import json
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_data(data_path: str) -> List[Dict]:
    """Load the original data file."""
    with open(data_path, 'r') as f:
        return json.load(f)


def extract_all_qa_pairs_by_image(data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Extract all QA pairs grouped by image.
    
    Returns:
        Dict mapping image_path -> list of QA pairs with metadata
    """
    image_qa_map = defaultdict(list)
    
    for item in data:
        image_path = item.get('image_path', item.get('image', ''))
        
        # Normalize image path
        if image_path.startswith('images/'):
            image_path = image_path[7:]
        
        # Extract QA pairs from all stages
        for stage_num in [1, 2, 3]:
            stages_data = item.get('stages', item.get('clinical_flow_stages', {}))
            
            # Try different key formats
            stage_qa = None
            for key in [f'stage_{stage_num}', 
                       f'Stage-{stage_num}',
                       f'Stage {stage_num}',
                       f'Stage-{stage_num}: Initial Assessment',
                       f'Stage-{stage_num}: Findings Identification',
                       f'Stage-{stage_num}: Clinical Context']:
                if key in stages_data:
                    stage_qa = stages_data[key]
                    break
            
            if stage_qa:
                for qa in stage_qa:
                    image_qa_map[image_path].append({
                        'image_path': image_path,
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'stage': stage_num,
                        'image_id': item.get('image_id', '')
                    })
    
    return image_qa_map


def split_by_image(image_qa_map: Dict[str, List[Dict]], 
                   train_ratio: float = 0.9,
                   seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data by unique images (not QA pairs).
    
    Args:
        image_qa_map: Dict mapping image -> QA pairs
        train_ratio: Fraction of images for training
        seed: Random seed for reproducibility
        
    Returns:
        (train_samples, test_samples) - Lists of QA pairs
    """
    # Get unique images
    unique_images = list(image_qa_map.keys())
    
    # Shuffle images with fixed seed
    np.random.seed(seed)
    np.random.shuffle(unique_images)
    
    # Split images
    split_idx = int(train_ratio * len(unique_images))
    train_images = set(unique_images[:split_idx])
    test_images = set(unique_images[split_idx:])
    
    # Collect all QA pairs for train and test images
    train_samples = []
    test_samples = []
    
    for image_path, qa_pairs in image_qa_map.items():
        if image_path in train_images:
            train_samples.extend(qa_pairs)
        else:
            test_samples.extend(qa_pairs)
    
    # Verify no overlap
    train_img_set = set([s['image_path'] for s in train_samples])
    test_img_set = set([s['image_path'] for s in test_samples])
    overlap = train_img_set & test_img_set
    
    assert len(overlap) == 0, f"ERROR: Found {len(overlap)} overlapping images!"
    
    return train_samples, test_samples


def split_by_stage(samples: List[Dict]) -> Dict[int, List[Dict]]:
    """Group samples by stage number."""
    stage_samples = {1: [], 2: [], 3: []}
    
    for sample in samples:
        stage_samples[sample['stage']].append(sample)
    
    return stage_samples


def save_split_data(output_dir: str, 
                   train_samples: List[Dict], 
                   test_samples: List[Dict]):
    """Save the properly split data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Split by stage
    train_by_stage = split_by_stage(train_samples)
    test_by_stage = split_by_stage(test_samples)
    
    # Save full splits
    with open(os.path.join(output_dir, 'train_all_stages.json'), 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    with open(os.path.join(output_dir, 'test_all_stages.json'), 'w') as f:
        json.dump(test_samples, f, indent=2)
    
    # Save per-stage splits
    for stage_num in [1, 2, 3]:
        with open(os.path.join(output_dir, f'train_stage{stage_num}.json'), 'w') as f:
            json.dump(train_by_stage[stage_num], f, indent=2)
        
        with open(os.path.join(output_dir, f'test_stage{stage_num}.json'), 'w') as f:
            json.dump(test_by_stage[stage_num], f, indent=2)
    
    # Save split metadata
    train_images = set([s['image_path'] for s in train_samples])
    test_images = set([s['image_path'] for s in test_samples])
    
    metadata = {
        'split_method': 'image_based',
        'train_test_ratio': '90/10',
        'random_seed': 42,
        'total_images': len(train_images) + len(test_images),
        'train_images': len(train_images),
        'test_images': len(test_images),
        'image_overlap': 0,
        'total_qa_pairs': len(train_samples) + len(test_samples),
        'train_qa_pairs': len(train_samples),
        'test_qa_pairs': len(test_samples),
        'stage_breakdown': {
            'stage_1': {
                'train': len(train_by_stage[1]),
                'test': len(test_by_stage[1])
            },
            'stage_2': {
                'train': len(train_by_stage[2]),
                'test': len(test_by_stage[2])
            },
            'stage_3': {
                'train': len(train_by_stage[3]),
                'test': len(test_by_stage[3])
            }
        }
    }
    
    with open(os.path.join(output_dir, 'split_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def print_statistics(metadata: Dict):
    """Print split statistics."""
    print("\n" + "="*80)
    print("✅ PROPER IMAGE-BASED SPLIT CREATED")
    print("="*80)
    print(f"\n📊 Overall Statistics:")
    print(f"   Total unique images: {metadata['total_images']}")
    print(f"   Train images: {metadata['train_images']} ({metadata['train_images']/metadata['total_images']*100:.1f}%)")
    print(f"   Test images: {metadata['test_images']} ({metadata['test_images']/metadata['total_images']*100:.1f}%)")
    print(f"   ✅ Image overlap: {metadata['image_overlap']} (CORRECT!)")
    
    print(f"\n📝 QA Pair Statistics:")
    print(f"   Total QA pairs: {metadata['total_qa_pairs']}")
    print(f"   Train QA pairs: {metadata['train_qa_pairs']} ({metadata['train_qa_pairs']/metadata['total_qa_pairs']*100:.1f}%)")
    print(f"   Test QA pairs: {metadata['test_qa_pairs']} ({metadata['test_qa_pairs']/metadata['total_qa_pairs']*100:.1f}%)")
    
    print(f"\n📊 Per-Stage Breakdown:")
    for stage in [1, 2, 3]:
        stage_data = metadata['stage_breakdown'][f'stage_{stage}']
        total = stage_data['train'] + stage_data['test']
        print(f"   Stage {stage}:")
        print(f"      Train: {stage_data['train']} ({stage_data['train']/total*100:.1f}%)")
        print(f"      Test:  {stage_data['test']} ({stage_data['test']/total*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ Data split properly by IMAGE ID - Zero leakage!")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create proper image-based train/test split")
    parser.add_argument("--data", type=str, required=True, help="Path to original data JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for splits")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.data}")
    data = load_data(args.data)
    print(f"Loaded {len(data)} items")
    
    print("\nExtracting QA pairs grouped by image...")
    image_qa_map = extract_all_qa_pairs_by_image(data)
    print(f"Found {len(image_qa_map)} unique images")
    
    total_qa = sum(len(qa_list) for qa_list in image_qa_map.values())
    print(f"Total QA pairs: {total_qa}")
    
    print(f"\nSplitting by image (ratio: {args.train_ratio}/{1-args.train_ratio}, seed: {args.seed})...")
    train_samples, test_samples = split_by_image(image_qa_map, args.train_ratio, args.seed)
    
    print(f"\nSaving splits to: {args.output}")
    metadata = save_split_data(args.output, train_samples, test_samples)
    
    print_statistics(metadata)
    
    print(f"✅ Files created in {args.output}:")
    print(f"   - train_all_stages.json")
    print(f"   - test_all_stages.json")
    print(f"   - train_stage1.json, train_stage2.json, train_stage3.json")
    print(f"   - test_stage1.json, test_stage2.json, test_stage3.json")
    print(f"   - split_metadata.json")


if __name__ == "__main__":
    main()

