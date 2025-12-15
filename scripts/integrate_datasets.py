#!/usr/bin/env python3
"""
Integrate Kvasir-VQA and TAMSET datasets.
Creates combined datasets for both reordered and non-reordered versions.
"""

import json
import os
from typing import Dict, List, Tuple
from pathlib import Path
import random

def integrate_reordered_datasets():
    """Integrate Kvasir-VQA and TAMSET reordered datasets."""
    
    print("=" * 80)
    print("INTEGRATING REORDERED DATASETS (3-Stage Clinical Flow)")
    print("=" * 80)
    
    # Paths
    kvasir_train = Path("/l/users/muhra.almahri/Surgical_COT/Kvasir-pilot/outputs/kvasir_train.json")
    kvasir_val = Path("/l/users/muhra.almahri/Surgical_COT/Kvasir-pilot/outputs/kvasir_val.json")
    kvasir_test = Path("/l/users/muhra.almahri/Surgical_COT/Kvasir-pilot/outputs/kvasir_test.json")
    tamset_file = Path("/l/users/muhra.almahri/Surgical_COT/temset/tamset_qa_qwen3_full_reordered.json")
    
    output_dir = Path("/l/users/muhra.almahri/Surgical_COT/datasets")
    output_dir.mkdir(exist_ok=True)
    
    # Check TAMSET availability
    if not tamset_file.exists():
        print(f"\n⚠️  TAMSET file not found: {tamset_file}")
        print("   Will create Kvasir-only datasets for now.")
        print("   Re-run this script after TAMSET Qwen3 generation completes.\n")
        tamset_data = None
    else:
        print(f"\n✅ TAMSET file found: {tamset_file}")
        with open(tamset_file, 'r') as f:
            tamset_data = json.load(f)
        print(f"   TAMSET videos: {len(tamset_data)}")
    
    # Load Kvasir-VQA
    print(f"\n📂 Loading Kvasir-VQA datasets...")
    with open(kvasir_train, 'r') as f:
        kvasir_train_data = json.load(f)
    with open(kvasir_val, 'r') as f:
        kvasir_val_data = json.load(f)
    with open(kvasir_test, 'r') as f:
        kvasir_test_data = json.load(f)
    
    print(f"   Train: {len(kvasir_train_data)} items")
    print(f"   Val: {len(kvasir_val_data)} items")
    print(f"   Test: {len(kvasir_test_data)} items")
    
    # Convert TAMSET dict to list and split
    if tamset_data:
        print(f"\n📂 Processing TAMSET data...")
        tamset_list = []
        for video_id, video_data in tamset_data.items():
            item = {
                'video_id': video_id,
                'source': 'TAMSET',
                'total_qa_pairs': video_data.get('total_qa_pairs', 0),
                'clinical_flow_stages': video_data.get('clinical_flow_stages', {})
            }
            tamset_list.append(item)
        
        # Shuffle and split TAMSET: 80% train, 10% val, 10% test
        random.seed(42)
        random.shuffle(tamset_list)
        
        n_videos = len(tamset_list)
        train_end = int(0.8 * n_videos)
        val_end = int(0.9 * n_videos)
        
        tamset_train = tamset_list[:train_end]
        tamset_val = tamset_list[train_end:val_end]
        tamset_test = tamset_list[val_end:]
        
        print(f"   TAMSET split:")
        print(f"   Train: {len(tamset_train)} videos")
        print(f"   Val: {len(tamset_val)} videos")
        print(f"   Test: {len(tamset_test)} videos")
    else:
        tamset_train = []
        tamset_val = []
        tamset_test = []
    
    # Add source tag to Kvasir items
    for item in kvasir_train_data:
        item['source'] = 'Kvasir-VQA'
    for item in kvasir_val_data:
        item['source'] = 'Kvasir-VQA'
    for item in kvasir_test_data:
        item['source'] = 'Kvasir-VQA'
    
    # Combine datasets
    print(f"\n🔗 Combining datasets...")
    
    integrated_train = kvasir_train_data + tamset_train
    integrated_val = kvasir_val_data + tamset_val
    integrated_test = kvasir_test_data + tamset_test
    
    # Shuffle combined datasets
    random.shuffle(integrated_train)
    random.shuffle(integrated_val)
    random.shuffle(integrated_test)
    
    # Calculate stats
    def count_qa_pairs(dataset):
        total = 0
        for item in dataset:
            if 'total_qa_pairs' in item:
                total += item['total_qa_pairs']
            elif 'clinical_flow_stages' in item:
                stages = item['clinical_flow_stages']
                total += len(stages.get('Stage-1: Initial Assessment', []))
                stage2 = stages.get('Stage-2: Findings Identification', {})
                total += sum(len(v) for v in stage2.values() if isinstance(v, list))
                total += len(stages.get('Stage-3: Relationships/Context', []))
        return total
    
    train_qa = count_qa_pairs(integrated_train)
    val_qa = count_qa_pairs(integrated_val)
    test_qa = count_qa_pairs(integrated_test)
    
    # Save integrated datasets
    print(f"\n💾 Saving integrated reordered datasets...")
    
    train_output = output_dir / "integrated_train_reordered.json"
    val_output = output_dir / "integrated_val_reordered.json"
    test_output = output_dir / "integrated_test_reordered.json"
    
    with open(train_output, 'w') as f:
        json.dump(integrated_train, f, indent=2)
    print(f"   ✅ {train_output}")
    
    with open(val_output, 'w') as f:
        json.dump(integrated_val, f, indent=2)
    print(f"   ✅ {val_output}")
    
    with open(test_output, 'w') as f:
        json.dump(integrated_test, f, indent=2)
    print(f"   ✅ {test_output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("REORDERED INTEGRATION COMPLETE")
    print("=" * 80)
    print(f"\n{'Split':<10} {'Items':<10} {'QA Pairs':<15} {'Kvasir':<10} {'TAMSET':<10}")
    print("-" * 80)
    print(f"{'Train':<10} {len(integrated_train):<10} {train_qa:<15,} {len(kvasir_train_data):<10} {len(tamset_train):<10}")
    print(f"{'Val':<10} {len(integrated_val):<10} {val_qa:<15,} {len(kvasir_val_data):<10} {len(tamset_val):<10}")
    print(f"{'Test':<10} {len(integrated_test):<10} {test_qa:<15,} {len(kvasir_test_data):<10} {len(tamset_test):<10}")
    print("-" * 80)
    print(f"{'Total':<10} {len(integrated_train) + len(integrated_val) + len(integrated_test):<10} {train_qa + val_qa + test_qa:<15,}")
    print("=" * 80)
    
    return {
        'train': {'items': len(integrated_train), 'qa_pairs': train_qa},
        'val': {'items': len(integrated_val), 'qa_pairs': val_qa},
        'test': {'items': len(integrated_test), 'qa_pairs': test_qa}
    }


def integrate_non_reordered_datasets():
    """Integrate Kvasir-VQA and TAMSET non-reordered datasets."""
    
    print("\n" + "=" * 80)
    print("INTEGRATING NON-REORDERED DATASETS (Baseline)")
    print("=" * 80)
    
    # Paths
    kvasir_dir = Path("/l/users/muhra.almahri/Surgical_COT/datasets")
    kvasir_train = kvasir_dir / "kvasir_train_non_reordered.json"
    kvasir_val = kvasir_dir / "kvasir_val_non_reordered.json"
    kvasir_test = kvasir_dir / "kvasir_test_non_reordered.json"
    tamset_file = kvasir_dir / "tamset_full_non_reordered.json"
    
    output_dir = Path("/l/users/muhra.almahri/Surgical_COT/datasets")
    
    # Check if non-reordered files exist
    if not kvasir_train.exists():
        print(f"\n⚠️  Kvasir non-reordered files not found!")
        print("   Run create_non_reordered_datasets.py first.")
        return None
    
    # Check TAMSET availability
    if not tamset_file.exists():
        print(f"\n⚠️  TAMSET non-reordered file not found: {tamset_file}")
        print("   Will create Kvasir-only datasets for now.")
        tamset_data = None
    else:
        print(f"\n✅ TAMSET file found: {tamset_file}")
        with open(tamset_file, 'r') as f:
            tamset_data = json.load(f)
        print(f"   TAMSET videos: {len(tamset_data)}")
    
    # Load Kvasir-VQA
    print(f"\n📂 Loading Kvasir-VQA non-reordered datasets...")
    with open(kvasir_train, 'r') as f:
        kvasir_train_data = json.load(f)
    with open(kvasir_val, 'r') as f:
        kvasir_val_data = json.load(f)
    with open(kvasir_test, 'r') as f:
        kvasir_test_data = json.load(f)
    
    print(f"   Train: {len(kvasir_train_data)} items")
    print(f"   Val: {len(kvasir_val_data)} items")
    print(f"   Test: {len(kvasir_test_data)} items")
    
    # Convert TAMSET dict to list and split
    if tamset_data:
        print(f"\n📂 Processing TAMSET data...")
        tamset_list = []
        for video_id, video_data in tamset_data.items():
            item = {
                'video_id': video_id,
                'source': 'TAMSET',
                'total_qa_pairs': video_data.get('total_qa_pairs', 0),
                'qa_pairs': video_data.get('qa_pairs', [])
            }
            tamset_list.append(item)
        
        # Split TAMSET with same seed as reordered
        random.seed(42)
        random.shuffle(tamset_list)
        
        n_videos = len(tamset_list)
        train_end = int(0.8 * n_videos)
        val_end = int(0.9 * n_videos)
        
        tamset_train = tamset_list[:train_end]
        tamset_val = tamset_list[train_end:val_end]
        tamset_test = tamset_list[val_end:]
        
        print(f"   TAMSET split:")
        print(f"   Train: {len(tamset_train)} videos")
        print(f"   Val: {len(tamset_val)} videos")
        print(f"   Test: {len(tamset_test)} videos")
    else:
        tamset_train = []
        tamset_val = []
        tamset_test = []
    
    # Add source tag to Kvasir items
    for item in kvasir_train_data:
        item['source'] = 'Kvasir-VQA'
    for item in kvasir_val_data:
        item['source'] = 'Kvasir-VQA'
    for item in kvasir_test_data:
        item['source'] = 'Kvasir-VQA'
    
    # Combine datasets
    print(f"\n🔗 Combining datasets...")
    
    integrated_train = kvasir_train_data + tamset_train
    integrated_val = kvasir_val_data + tamset_val
    integrated_test = kvasir_test_data + tamset_test
    
    # Shuffle with same seed for consistency
    random.seed(42)
    random.shuffle(integrated_train)
    random.shuffle(integrated_val)
    random.shuffle(integrated_test)
    
    # Calculate stats
    train_qa = sum(item.get('total_qa_pairs', 0) for item in integrated_train)
    val_qa = sum(item.get('total_qa_pairs', 0) for item in integrated_val)
    test_qa = sum(item.get('total_qa_pairs', 0) for item in integrated_test)
    
    # Save integrated datasets
    print(f"\n💾 Saving integrated non-reordered datasets...")
    
    train_output = output_dir / "integrated_train_non_reordered.json"
    val_output = output_dir / "integrated_val_non_reordered.json"
    test_output = output_dir / "integrated_test_non_reordered.json"
    
    with open(train_output, 'w') as f:
        json.dump(integrated_train, f, indent=2)
    print(f"   ✅ {train_output}")
    
    with open(val_output, 'w') as f:
        json.dump(integrated_val, f, indent=2)
    print(f"   ✅ {val_output}")
    
    with open(test_output, 'w') as f:
        json.dump(integrated_test, f, indent=2)
    print(f"   ✅ {test_output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("NON-REORDERED INTEGRATION COMPLETE")
    print("=" * 80)
    print(f"\n{'Split':<10} {'Items':<10} {'QA Pairs':<15} {'Kvasir':<10} {'TAMSET':<10}")
    print("-" * 80)
    print(f"{'Train':<10} {len(integrated_train):<10} {train_qa:<15,} {len(kvasir_train_data):<10} {len(tamset_train):<10}")
    print(f"{'Val':<10} {len(integrated_val):<10} {val_qa:<15,} {len(kvasir_val_data):<10} {len(tamset_val):<10}")
    print(f"{'Test':<10} {len(integrated_test):<10} {test_qa:<15,} {len(kvasir_test_data):<10} {len(tamset_test):<10}")
    print("-" * 80)
    print(f"{'Total':<10} {len(integrated_train) + len(integrated_val) + len(integrated_test):<10} {train_qa + val_qa + test_qa:<15,}")
    print("=" * 80)
    
    return {
        'train': {'items': len(integrated_train), 'qa_pairs': train_qa},
        'val': {'items': len(integrated_val), 'qa_pairs': val_qa},
        'test': {'items': len(integrated_test), 'qa_pairs': test_qa}
    }


def main():
    """Main execution."""
    print("\n🔗 DATASET INTEGRATION: KVASIR-VQA + TAMSET")
    print("=" * 80)
    print("Creating combined datasets for training and evaluation")
    print("=" * 80)
    
    # Integrate reordered datasets
    reordered_stats = integrate_reordered_datasets()
    
    # Integrate non-reordered datasets
    non_reordered_stats = integrate_non_reordered_datasets()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 INTEGRATION COMPLETE")
    print("=" * 80)
    
    if reordered_stats:
        total_reordered = sum(s['qa_pairs'] for s in reordered_stats.values())
        print(f"\n✅ Reordered datasets: {total_reordered:,} total QA pairs")
    
    if non_reordered_stats:
        total_non_reordered = sum(s['qa_pairs'] for s in non_reordered_stats.values())
        print(f"✅ Non-reordered datasets: {total_non_reordered:,} total QA pairs")
    
    print("\n📁 All datasets saved to: /l/users/muhra.almahri/Surgical_COT/datasets/")
    print("🚀 Ready for model training!")
    print("=" * 80)


if __name__ == "__main__":
    main()

