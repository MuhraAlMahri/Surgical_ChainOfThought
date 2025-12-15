#!/usr/bin/env python3
"""
Create Non-Reordered Baseline from TAMSET Qwen3 Output

This script takes the reordered TAMSET QA pairs and creates a non-reordered
baseline version by randomizing the QA order within each video/frame.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def create_non_reordered_baseline(reordered_data: Dict, seed: int = 42) -> Dict:
    """
    Create non-reordered baseline by shuffling QA pairs within each video.
    
    Args:
        reordered_data: Dict with video_id -> {qa_pairs, metadata}
        seed: Random seed for reproducibility
    
    Returns:
        Non-reordered version with shuffled QA pairs
    """
    random.seed(seed)
    non_reordered = {}
    
    for video_id, video_data in reordered_data.items():
        # Copy metadata
        non_reordered[video_id] = {
            'video_id': video_data.get('video_id', video_id),
            'metadata': video_data.get('metadata', {}),
            'qa_pairs': []
        }
        
        # Shuffle QA pairs
        qa_pairs = video_data.get('qa_pairs', [])
        shuffled_qa = qa_pairs.copy()
        random.shuffle(shuffled_qa)
        
        # Assign new order
        for idx, qa in enumerate(shuffled_qa):
            qa_copy = qa.copy()
            qa_copy['order'] = idx + 1  # 1-indexed
            qa_copy['is_reordered'] = False
            non_reordered[video_id]['qa_pairs'].append(qa_copy)
    
    return non_reordered


def main():
    parser = argparse.ArgumentParser(description='Create non-reordered baseline from TAMSET')
    parser.add_argument('--input', required=True, help='Input reordered JSON file')
    parser.add_argument('--output', required=True, help='Output non-reordered JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Loading reordered data from: {args.input}")
    with open(args.input, 'r') as f:
        reordered_data = json.load(f)
    
    print(f"Creating non-reordered baseline (seed={args.seed})...")
    non_reordered_data = create_non_reordered_baseline(reordered_data, args.seed)
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving non-reordered data to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(non_reordered_data, f, indent=2)
    
    # Statistics
    total_videos = len(non_reordered_data)
    total_qa = sum(len(v['qa_pairs']) for v in non_reordered_data.values())
    
    print(f"\n✅ Non-reordered baseline created!")
    print(f"   - Videos: {total_videos}")
    print(f"   - Total QA pairs: {total_qa}")
    print(f"   - Average QA per video: {total_qa/total_videos:.1f}")


if __name__ == '__main__':
    main()

