#!/usr/bin/env python3
"""
Create stage-specific train/val/test splits for experiments 3 and 4.
Uses the 'stage' field from the dataset to filter by stage 1, 2, or 3.
"""

import json
import os
from pathlib import Path


def create_stage_splits(base_dir: str, output_base_dir: str):
    """
    Create stage-specific splits from the main dataset.
    
    Args:
        base_dir: Path to kvasir_raw_6500_image_level_70_15_15 directory
        output_base_dir: Base directory for output stage splits
    """
    splits = ['train', 'val', 'test']
    
    for split in splits:
        input_file = os.path.join(base_dir, f'{split}.json')
        
        if not os.path.exists(input_file):
            print(f'⚠️  Skipping {split} - file not found: {input_file}')
            continue
        
        print(f'\n📊 Processing {split}...')
        
        # Read data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        print(f'  Loaded {len(data)} total samples')
        
        # Group by stage
        stage_data = {1: [], 2: [], 3: []}
        
        for item in data:
            # Use the stage field from the dataset
            stage = item.get('stage', 2)  # Default to 2 if missing
            if stage in [1, 2, 3]:
                stage_data[stage].append(item)
            else:
                # If stage is not 1-3, categorize by question content
                q = item.get('question', '').lower()
                if '[stage-1' in q or 'initial assessment' in q or 'quality control' in q:
                    stage_data[1].append(item)
                elif '[stage-3' in q or 'clinical context' in q or 'diagnosis' in q:
                    stage_data[3].append(item)
                else:
                    stage_data[2].append(item)
        
        # Write stage splits
        for stage_num in [1, 2, 3]:
            stage_dir = os.path.join(output_base_dir, f'kvasir_stage_splits_stage{stage_num}')
            os.makedirs(stage_dir, exist_ok=True)
            
            output_file = os.path.join(stage_dir, f'{split}.json')
            with open(output_file, 'w') as f:
                json.dump(stage_data[stage_num], f, indent=2)
            
            print(f'  Stage {stage_num}: {len(stage_data[stage_num])} samples')
    
    print('\n✅ Done creating 3-stage splits!')


if __name__ == '__main__':
    # Paths relative to this script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_project = os.path.join(script_dir, '..', '..')
    
    dataset_dir = os.path.join(base_project, 'datasets', 'kvasir_raw_6500_image_level_70_15_15')
    output_dir = os.path.join(base_project, 'datasets')
    
    print(f"📁 Input directory: {dataset_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    create_stage_splits(dataset_dir, output_dir)

