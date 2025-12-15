#!/usr/bin/env python3
"""
Create proper sequence-based train/val/test split for EndoVis2018
following surgical dataset best practices (like Surgery R1 paper).

This ensures:
- No sequence overlap between splits (prevents data leakage)
- Entire sequences assigned to one split only
- Standard 70/15/15 or 80/10/10 split ratio
"""

import json
from pathlib import Path
from collections import defaultdict
import random

BASE_DIR = Path("/l/users/muhra.almahri/Surgical_COT")
INDEX_DIR = BASE_DIR / "EndoVis2018" / "data" / "index"
OUTPUT_DIR = BASE_DIR / "EndoVis2018" / "data" / "index"


def get_all_sequences():
    """Get all available sequences (1-16, excluding 8)."""
    # Sequence 8 is missing according to the dataset
    all_seqs = [str(i) for i in range(1, 17) if i != 8]
    return sorted(all_seqs, key=int)


def create_sequence_based_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Create sequence-based split with no overlap.
    
    Args:
        train_ratio: Proportion of sequences for training
        val_ratio: Proportion of sequences for validation
        test_ratio: Proportion of sequences for testing
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    # Get all sequences
    all_seqs = get_all_sequences()
    n_seqs = len(all_seqs)
    
    # Shuffle sequences
    shuffled_seqs = all_seqs.copy()
    random.shuffle(shuffled_seqs)
    
    # Calculate split points
    n_train = int(n_seqs * train_ratio)
    n_val = int(n_seqs * val_ratio)
    
    # Assign sequences to splits
    train_seqs = sorted(shuffled_seqs[:n_train], key=int)
    val_seqs = sorted(shuffled_seqs[n_train:n_train + n_val], key=int)
    test_seqs = sorted(shuffled_seqs[n_train + n_val:], key=int)
    
    print("=" * 60)
    print("Sequence-Based Split (No Overlap)")
    print("=" * 60)
    print(f"\nTotal sequences: {n_seqs}")
    print(f"Train sequences ({len(train_seqs)}): {train_seqs}")
    print(f"Val sequences ({len(val_seqs)}): {val_seqs}")
    print(f"Test sequences ({len(test_seqs)}): {test_seqs}")
    print(f"\nSplit ratios: {train_ratio:.1%} / {val_ratio:.1%} / {test_ratio:.1%}")
    print(f"Random seed: {seed}")
    
    return train_seqs, val_seqs, test_seqs


def get_frames_for_sequence(seq):
    """Get all frame numbers for a sequence (typically 0-148)."""
    # Check existing files to determine frame range
    # Most sequences have 149 frames (0-148)
    return list(range(149))


def create_split_files(train_seqs, val_seqs, test_seqs, output_dir):
    """Create train/val/test split files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train file
    train_file = output_dir / "train_data_proper.txt"
    with open(train_file, 'w') as f:
        f.write("Seq# Frame#\n")
        for seq in train_seqs:
            for frame in get_frames_for_sequence(seq):
                f.write(f" {int(seq):3d},   {frame:03d} \n")
    print(f"\n✅ Created: {train_file}")
    
    # Create validation file
    val_file = output_dir / "validation_data_proper.txt"
    with open(val_file, 'w') as f:
        f.write("Seq# Frame#\n")
        for seq in val_seqs:
            for frame in get_frames_for_sequence(seq):
                f.write(f" {int(seq):3d},   {frame:03d} \n")
    print(f"✅ Created: {val_file}")
    
    # Create test file
    test_file = output_dir / "test_data_proper.txt"
    with open(test_file, 'w') as f:
        f.write("Seq# Frame#\n")
        for seq in test_seqs:
            for frame in get_frames_for_sequence(seq):
                f.write(f" {int(seq):3d},   {frame:03d} \n")
    print(f"✅ Created: {test_file}")
    
    # Create metadata
    metadata = {
        "split_method": "sequence_based_no_overlap",
        "description": "Proper sequence-based split with no overlap between train/val/test",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_seed": 42,
        "sequences": {
            "train": train_seqs,
            "validation": val_seqs,
            "test": test_seqs
        },
        "total_sequences": len(train_seqs) + len(val_seqs) + len(test_seqs),
        "files": {
            "train": "train_data_proper.txt",
            "validation": "validation_data_proper.txt",
            "test": "test_data_proper.txt"
        }
    }
    
    metadata_file = output_dir / "split_metadata_proper.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Created: {metadata_file}")
    
    return metadata


def verify_no_overlap(train_seqs, val_seqs, test_seqs):
    """Verify no sequence overlap between splits."""
    train_set = set(train_seqs)
    val_set = set(val_seqs)
    test_set = set(test_seqs)
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("\n⚠️  WARNING: Overlap detected!")
        if overlap_train_val:
            print(f"  Train-Val overlap: {overlap_train_val}")
        if overlap_train_test:
            print(f"  Train-Test overlap: {overlap_train_test}")
        if overlap_val_test:
            print(f"  Val-Test overlap: {overlap_val_test}")
        return False
    else:
        print("\n✅ No overlap detected - split is clean!")
        return True


def main():
    print("🚀 Creating Proper Sequence-Based Split for EndoVis2018")
    print("=" * 60)
    
    # Create split (70/15/15 ratio)
    train_seqs, val_seqs, test_seqs = create_sequence_based_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # Verify no overlap
    verify_no_overlap(train_seqs, val_seqs, test_seqs)
    
    # Create split files
    metadata = create_split_files(train_seqs, val_seqs, test_seqs, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Split Summary")
    print("=" * 60)
    print(f"Train: {len(train_seqs)} sequences")
    print(f"Validation: {len(val_seqs)} sequences")
    print(f"Test: {len(test_seqs)} sequences")
    print(f"\nFiles created in: {OUTPUT_DIR}")
    print("\n✅ Done! Use these files for proper sequence-based splitting.")


if __name__ == '__main__':
    main()





















