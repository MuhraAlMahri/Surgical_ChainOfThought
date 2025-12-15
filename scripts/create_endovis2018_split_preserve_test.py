#!/usr/bin/env python3
"""
Create sequence-based train/val/test split for EndoVis2018
preserving the original test split (sequences 1-4) and properly
splitting remaining sequences for train/val.

This follows surgical dataset best practices:
- Test sequences fixed (as in original dataset/paper)
- Train/Val split from remaining sequences with no overlap
"""

import json
from pathlib import Path
import random

BASE_DIR = Path("/l/users/muhra.almahri/Surgical_COT")
INDEX_DIR = BASE_DIR / "EndoVis2018" / "data" / "index"
OUTPUT_DIR = BASE_DIR / "EndoVis2018" / "data" / "index"


def get_all_sequences():
    """Get all available sequences (1-16, excluding 8)."""
    all_seqs = [str(i) for i in range(1, 17) if i != 8]
    return sorted(all_seqs, key=int)


def create_split_preserving_test(test_seqs=['1', '2', '3', '4'], val_ratio=0.18, seed=42):
    """
    Create split preserving original test sequences and splitting rest.
    
    Args:
        test_seqs: Sequences to use for test (default: 1-4 as in original)
        val_ratio: Proportion of remaining sequences for validation
        seed: Random seed
    """
    random.seed(seed)
    
    # Get all sequences
    all_seqs = get_all_sequences()
    test_seqs = [str(s) for s in test_seqs]
    
    # Get remaining sequences for train/val
    remaining_seqs = [s for s in all_seqs if s not in test_seqs]
    random.shuffle(remaining_seqs)
    
    # Split remaining sequences
    # Ensure at least 2 sequences for validation (better for evaluation)
    n_val = max(2, int(len(remaining_seqs) * val_ratio))
    val_seqs = sorted(remaining_seqs[:n_val], key=int)
    train_seqs = sorted(remaining_seqs[n_val:], key=int)
    
    print("=" * 60)
    print("Sequence-Based Split (Preserving Original Test)")
    print("=" * 60)
    print(f"\nTotal sequences: {len(all_seqs)}")
    print(f"Test sequences (preserved): {sorted(test_seqs, key=int)}")
    print(f"Train sequences ({len(train_seqs)}): {train_seqs}")
    print(f"Val sequences ({len(val_seqs)}): {val_seqs}")
    print(f"\nSplit ratios: ~{len(train_seqs)/len(all_seqs):.1%} train / {len(val_seqs)/len(all_seqs):.1%} val / {len(test_seqs)/len(all_seqs):.1%} test")
    print(f"Random seed: {seed}")
    
    return train_seqs, val_seqs, test_seqs


def get_frames_for_sequence(seq):
    """Get all frame numbers for a sequence (typically 0-148)."""
    return list(range(149))


def create_split_files(train_seqs, val_seqs, test_seqs, output_dir):
    """Create train/val/test split files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train file
    train_file = output_dir / "train_data_sequence_based.txt"
    with open(train_file, 'w') as f:
        f.write("Seq# Frame#\n")
        for seq in train_seqs:
            for frame in get_frames_for_sequence(seq):
                f.write(f" {int(seq):3d},   {frame:03d} \n")
    print(f"\n✅ Created: {train_file}")
    
    # Create validation file
    val_file = output_dir / "validation_data_sequence_based.txt"
    with open(val_file, 'w') as f:
        f.write("Seq# Frame#\n")
        for seq in val_seqs:
            for frame in get_frames_for_sequence(seq):
                f.write(f" {int(seq):3d},   {frame:03d} \n")
    print(f"✅ Created: {val_file}")
    
    # Create test file (use original test_data_final.txt format)
    test_file = output_dir / "test_data_sequence_based.txt"
    with open(test_file, 'w') as f:
        f.write("Seq# Frame#\n")
        for seq in test_seqs:
            for frame in get_frames_for_sequence(seq):
                f.write(f" {int(seq):3d},   {frame:03d} \n")
    print(f"✅ Created: {test_file}")
    
    # Create metadata
    metadata = {
        "split_method": "sequence_based_preserve_test",
        "description": "Sequence-based split preserving original test sequences (1-4), proper train/val split from remaining sequences",
        "test_sequences_preserved": test_seqs,
        "val_ratio": 0.18,
        "random_seed": 42,
        "sequences": {
            "train": train_seqs,
            "validation": val_seqs,
            "test": test_seqs
        },
        "total_sequences": len(train_seqs) + len(val_seqs) + len(test_seqs),
        "files": {
            "train": "train_data_sequence_based.txt",
            "validation": "validation_data_sequence_based.txt",
            "test": "test_data_sequence_based.txt"
        }
    }
    
    metadata_file = output_dir / "split_metadata_sequence_based.json"
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
    print("🚀 Creating Sequence-Based Split for EndoVis2018")
    print("(Preserving original test sequences 1-4)")
    print("=" * 60)
    
    # Create split preserving test sequences 1-4
    # Using ~0.18 ratio to get 2 sequences for validation (better than 1)
    train_seqs, val_seqs, test_seqs = create_split_preserving_test(
        test_seqs=['1', '2', '3', '4'],
        val_ratio=0.18,  # Gives 2 val sequences from 11 remaining
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
    print(f"Train: {len(train_seqs)} sequences - {train_seqs}")
    print(f"Validation: {len(val_seqs)} sequences - {val_seqs}")
    print(f"Test: {len(test_seqs)} sequences - {test_seqs} (preserved from original)")
    print(f"\nFiles created in: {OUTPUT_DIR}")
    print("\n✅ Done! This split follows surgical dataset best practices.")


if __name__ == '__main__':
    main()

