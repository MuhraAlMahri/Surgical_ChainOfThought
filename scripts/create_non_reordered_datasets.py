#!/usr/bin/env python3
"""
Convert reordered (3-stage clinical flow) datasets to non-reordered (flat) format.
Creates baseline datasets with same QA pairs but without stage organization.
"""

import json
import os
from typing import Dict, List
from pathlib import Path

def convert_reordered_to_flat(reordered_data: Dict) -> Dict:
    """
    Convert reordered dataset to flat structure.
    
    Args:
        reordered_data: Dataset with clinical_flow_stages structure
    
    Returns:
        Dataset with flat qa_pairs list
    """
    flat_data = {}
    
    # Handle list format (Kvasir-VQA)
    if isinstance(reordered_data, list):
        flat_list = []
        
        for item in reordered_data:
            qa_pairs = []
            stages = item.get('clinical_flow_stages', {})
            
            # Extract all QA pairs from all stages (flatten)
            # Stage 1
            for qa in stages.get('Stage-1: Initial Assessment', []):
                qa_pairs.append({
                    'question': qa['question'],
                    'answer': qa['answer']
                })
            
            # Stage 2
            stage2 = stages.get('Stage-2: Findings Identification', {})
            for finding_type in ['instruments', 'anatomical_landmarks', 'procedures', 'abnormalities']:
                for qa in stage2.get(finding_type, []):
                    qa_pairs.append({
                        'question': qa['question'],
                        'answer': qa['answer']
                    })
            
            # Stage 3
            for qa in stages.get('Stage-3: Relationships/Context', []):
                qa_pairs.append({
                    'question': qa['question'],
                    'answer': qa['answer']
                })
            
            # Create flat item
            flat_item = {
                'image_id': item.get('image_id', 'unknown'),
                'qa_pairs': qa_pairs,
                'total_qa_pairs': len(qa_pairs)
            }
            
            # Preserve metadata if present
            if 'image_path' in item:
                flat_item['image_path'] = item['image_path']
            if 'image_name' in item:
                flat_item['image_name'] = item['image_name']
            
            flat_list.append(flat_item)
        
        return flat_list
    
    # Handle dict format (TAMSET)
    elif isinstance(reordered_data, dict):
        flat_dict = {}
        
        for video_id, video_data in reordered_data.items():
            qa_pairs = []
            stages = video_data.get('clinical_flow_stages', {})
            
            # Extract all QA pairs (flatten)
            # Stage 1
            for qa in stages.get('Stage-1: Initial Assessment', []):
                qa_pairs.append({
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'label': qa.get('label'),
                    'start': qa.get('start'),
                    'video': qa.get('video')
                })
            
            # Stage 2
            stage2 = stages.get('Stage-2: Findings Identification', {})
            for finding_type in ['instruments', 'anatomical_landmarks', 'procedures', 'abnormalities']:
                for qa in stage2.get(finding_type, []):
                    qa_pairs.append({
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'label': qa.get('label'),
                        'start': qa.get('start'),
                        'video': qa.get('video')
                    })
            
            # Stage 3
            for qa in stages.get('Stage-3: Relationships/Context', []):
                qa_pairs.append({
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'label': qa.get('label'),
                    'start': qa.get('start'),
                    'video': qa.get('video')
                })
            
            flat_dict[video_id] = {
                'qa_pairs': qa_pairs,
                'total_qa_pairs': len(qa_pairs)
            }
        
        return flat_dict
    
    return reordered_data


def process_kvasir_to_non_reordered():
    """Convert Kvasir-VQA reordered datasets to non-reordered format."""
    
    print("=" * 80)
    print("Converting Kvasir-VQA to Non-Reordered Format")
    print("=" * 80)
    
    base_path = Path("/l/users/muhra.almahri/Surgical_COT/Kvasir-pilot/outputs")
    output_path = Path("/l/users/muhra.almahri/Surgical_COT/datasets")
    output_path.mkdir(exist_ok=True)
    
    splits = ['train', 'val', 'test']
    total_stats = {}
    
    for split in splits:
        input_file = base_path / f"kvasir_{split}.json"
        output_file = output_path / f"kvasir_{split}_non_reordered.json"
        
        print(f"\n📂 Processing {split} split...")
        print(f"   Input: {input_file}")
        
        # Load reordered data
        with open(input_file, 'r') as f:
            reordered_data = json.load(f)
        
        # Convert to flat
        flat_data = convert_reordered_to_flat(reordered_data)
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(flat_data, f, indent=2)
        
        # Stats
        if isinstance(flat_data, list):
            total_qa = sum(item['total_qa_pairs'] for item in flat_data)
            total_items = len(flat_data)
        else:
            total_qa = sum(item['total_qa_pairs'] for item in flat_data.values())
            total_items = len(flat_data)
        
        total_stats[split] = {
            'items': total_items,
            'qa_pairs': total_qa
        }
        
        print(f"   ✅ Saved: {output_file}")
        print(f"   Items: {total_items:,}")
        print(f"   QA pairs: {total_qa:,}")
    
    print("\n" + "=" * 80)
    print("Kvasir-VQA Non-Reordered Conversion Complete")
    print("=" * 80)
    print(f"\nTrain: {total_stats['train']['items']:,} items, {total_stats['train']['qa_pairs']:,} QA pairs")
    print(f"Val:   {total_stats['val']['items']:,} items, {total_stats['val']['qa_pairs']:,} QA pairs")
    print(f"Test:  {total_stats['test']['items']:,} items, {total_stats['test']['qa_pairs']:,} QA pairs")
    
    return total_stats


def process_tamset_to_non_reordered():
    """Convert TAMSET reordered dataset to non-reordered format."""
    
    print("\n" + "=" * 80)
    print("Converting TAMSET to Non-Reordered Format")
    print("=" * 80)
    
    input_file = Path("/l/users/muhra.almahri/Surgical_COT/temset/tamset_qa_qwen3_full_reordered.json")
    output_file = Path("/l/users/muhra.almahri/Surgical_COT/datasets/tamset_full_non_reordered.json")
    
    if not input_file.exists():
        print(f"⚠️  Input file not found: {input_file}")
        print("   TAMSET Qwen3 generation must complete first!")
        return None
    
    print(f"\n📂 Processing TAMSET dataset...")
    print(f"   Input: {input_file}")
    
    # Load reordered data
    with open(input_file, 'r') as f:
        reordered_data = json.load(f)
    
    # Convert to flat
    flat_data = convert_reordered_to_flat(reordered_data)
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(flat_data, f, indent=2)
    
    # Stats
    if isinstance(flat_data, list):
        total_qa = sum(item['total_qa_pairs'] for item in flat_data)
        total_items = len(flat_data)
    else:
        total_qa = sum(item['total_qa_pairs'] for item in flat_data.values())
        total_items = len(flat_data)
    
    print(f"   ✅ Saved: {output_file}")
    print(f"   Videos: {total_items:,}")
    print(f"   QA pairs: {total_qa:,}")
    
    print("\n" + "=" * 80)
    print("TAMSET Non-Reordered Conversion Complete")
    print("=" * 80)
    
    return {'videos': total_items, 'qa_pairs': total_qa}


def main():
    """Main execution."""
    print("\n🔄 DATASET CONVERSION: REORDERED → NON-REORDERED")
    print("=" * 80)
    print("Creating baseline datasets without 3-stage organization")
    print("=" * 80)
    
    # Process Kvasir-VQA
    kvasir_stats = process_kvasir_to_non_reordered()
    
    # Process TAMSET
    tamset_stats = process_tamset_to_non_reordered()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 ALL CONVERSIONS COMPLETE")
    print("=" * 80)
    
    print("\n📊 KVASIR-VQA:")
    if kvasir_stats:
        total_kvasir_qa = sum(s['qa_pairs'] for s in kvasir_stats.values())
        print(f"   Total QA pairs: {total_kvasir_qa:,}")
    
    print("\n📊 TAMSET:")
    if tamset_stats:
        print(f"   Total QA pairs: {tamset_stats['qa_pairs']:,}")
    else:
        print("   ⚠️  Not yet available (Qwen3 generation pending)")
    
    print("\n✅ Non-reordered datasets ready for baseline model training!")
    print("=" * 80)


if __name__ == "__main__":
    main()
