#!/usr/bin/env python3
"""
Data Splitting Script for Surgical CoT Dataset

This script reads the data_manifest.json and creates proper train/val/test splits
respecting original splits or creating stratified splits if they don't exist.
"""

import json
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataSplitter:
    """Handles creation of train/val/test splits for the surgical dataset."""
    
    def __init__(self, manifest_path: str, config_path: str):
        """Initialize with manifest and config paths."""
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['output']['standardized_dir'])
        
        # Split ratios
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Ensure ratios sum to 1
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    def get_dataset_splits(self) -> Dict[str, List[Dict]]:
        """Group samples by dataset for stratified splitting."""
        dataset_samples = defaultdict(list)
        
        for sample in self.manifest['samples']:
            dataset_name = sample['dataset_name']
            dataset_samples[dataset_name].append(sample)
        
        return dict(dataset_samples)
    
    def create_stratified_splits(self, samples: List[Dict], random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create stratified train/val/test splits."""
        if len(samples) < 3:
            # If too few samples, put all in train
            return samples, [], []
        
        # First split: train vs (val + test)
        train_samples, temp_samples = train_test_split(
            samples, 
            test_size=(self.val_ratio + self.test_ratio),
            random_state=random_state,
            stratify=None  # We'll handle stratification at dataset level
        )
        
        # Second split: val vs test
        if len(temp_samples) < 2:
            val_samples = temp_samples
            test_samples = []
        else:
            val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_samples, test_samples = train_test_split(
                temp_samples,
                test_size=(1 - val_ratio_adjusted),
                random_state=random_state
            )
        
        return train_samples, val_samples, test_samples
    
    def update_sample_splits(self, dataset_name: str, train_samples: List[Dict], 
                           val_samples: List[Dict], test_samples: List[Dict]) -> None:
        """Update the split field for samples and move files to appropriate directories."""
        
        # Update train samples
        for sample in train_samples:
            sample['split'] = 'train'
            self._move_sample_to_split(sample, 'train')
        
        # Update val samples
        for sample in val_samples:
            sample['split'] = 'val'
            self._move_sample_to_split(sample, 'val')
        
        # Update test samples
        for sample in test_samples:
            sample['split'] = 'test'
            self._move_sample_to_split(sample, 'test')
        
        logger.info(f"{dataset_name}: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    
    def _move_sample_to_split(self, sample: Dict, split: str) -> None:
        """Move sample files to the appropriate split directory."""
        dataset_name = sample['dataset_name']
        sample_id = sample['sample_id']
        
        # Current path (should be in train/)
        current_dir = self.output_dir / "train" / dataset_name / sample_id
        
        # New path
        new_dir = self.output_dir / split / dataset_name / sample_id
        
        if current_dir.exists() and current_dir != new_dir:
            # Create new directory
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # Move all files
            for file_path in current_dir.iterdir():
                if file_path.is_file():
                    shutil.move(str(file_path), str(new_dir / file_path.name))
            
            # Remove old directory if empty
            try:
                current_dir.rmdir()
            except OSError:
                pass  # Directory not empty or other error
        
        # Update paths in sample
        sample['image_path'] = str(new_dir / f"{sample_id}.jpg")
        if sample.get('video_path'):
            sample['video_path'] = str(new_dir / f"{sample_id}.mp4")
        sample['report_path'] = str(new_dir / "report.txt")
    
    def create_splits(self) -> None:
        """Create train/val/test splits for all datasets."""
        logger.info("Creating data splits...")
        
        dataset_samples = self.get_dataset_splits()
        
        total_train = 0
        total_val = 0
        total_test = 0
        
        for dataset_name, samples in dataset_samples.items():
            logger.info(f"Processing {dataset_name} with {len(samples)} samples")
            
            # Create splits
            train_samples, val_samples, test_samples = self.create_stratified_splits(samples)
            
            # Update sample splits and move files
            self.update_sample_splits(dataset_name, train_samples, val_samples, test_samples)
            
            total_train += len(train_samples)
            total_val += len(val_samples)
            total_test += len(test_samples)
        
        # Update manifest metadata
        self.manifest['metadata']['splits'] = {
            'train': total_train,
            'val': total_val,
            'test': total_test,
            'total': total_train + total_val + total_test
        }
        
        # Save updated manifest
        manifest_path = self.output_dir / "data_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        logger.info(f"Split creation complete!")
        logger.info(f"Train: {total_train}, Val: {total_val}, Test: {total_test}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATA SPLIT SUMMARY")
        print("="*50)
        print(f"{'Split':<10} {'Count':<8} {'Percentage':<10}")
        print("-" * 50)
        total = total_train + total_val + total_test
        print(f"{'Train':<10} {total_train:<8} {total_train/total*100:.1f}%")
        print(f"{'Val':<10} {total_val:<8} {total_val/total*100:.1f}%")
        print(f"{'Test':<10} {total_test:<8} {total_test/total*100:.1f}%")
        print(f"{'Total':<10} {total:<8} {100.0:.1f}%")
        print("="*50)

def main():
    """Main function to run data splitting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--manifest", default="data/standardized/data_manifest.json",
                       help="Path to data manifest file")
    parser.add_argument("--config", default="configs/paths.yaml",
                       help="Path to configuration file")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test set ratio (default: 0.15)")
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Initialize splitter
    splitter = DataSplitter(args.manifest, args.config)
    
    # Update ratios
    splitter.train_ratio = args.train_ratio
    splitter.val_ratio = args.val_ratio
    splitter.test_ratio = args.test_ratio
    
    # Create splits
    splitter.create_splits()

if __name__ == "__main__":
    import shutil
    main()
