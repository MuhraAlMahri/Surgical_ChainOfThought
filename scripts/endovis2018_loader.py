#!/usr/bin/env python3
"""
EndoVis2018 Dataset Loader and Converter

This script helps load and convert the EndoVis2018 surgical scene segmentation dataset
for use in the Surgical COT project. The EndoVis2018 dataset contains:
- Surgical scene images from multiple sequences
- Segmentation labels for 12 classes (instruments, organs, etc.)
- Train/validation/test splits

Usage:
    python scripts/endovis2018_loader.py --action list
    python scripts/endovis2018_loader.py --action convert --output datasets/endovis2018
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import shutil

class EndoVis2018Loader:
    """Loader for EndoVis2018 dataset."""
    
    def __init__(self, data_root: str):
        """
        Initialize the loader.
        
        Args:
            data_root: Path to EndoVis2018 repository root
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "data" / "images"
        self.index_dir = self.data_root / "data" / "index"
        self.labels_file = self.data_root / "data" / "labels.json"
        
        # Load class labels
        with open(self.labels_file, 'r') as f:
            self.classes = json.load(f)
        
        self.class_map = {cls['classid']: cls['name'] for cls in self.classes}
        self.class_names = [cls['name'] for cls in sorted(self.classes, key=lambda x: x['classid'])]
    
    def get_class_info(self) -> Dict:
        """Get information about the segmentation classes."""
        return {
            'total_classes': len(self.classes),
            'classes': self.classes,
            'class_map': self.class_map,
            'class_names': self.class_names
        }
    
    def parse_index_file(self, index_file: str) -> List[Dict]:
        """
        Parse an index file (train_data.txt, validation_data.txt, etc.)
        
        Returns:
            List of dicts with 'seq' and 'frame' keys
        """
        index_path = self.index_dir / index_file
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        samples = []
        with open(index_path, 'r') as f:
            lines = f.readlines()
            # Skip header line
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    seq = parts[0].strip()
                    frame = parts[1].strip().zfill(3)
                    samples.append({
                        'seq': seq,
                        'frame': frame,
                        'image_path': f"images/seq_{seq}/left_frames/frame{frame}.png",
                        'label_path': f"images/seq_{seq}/labels/frame{frame}.png"
                    })
        
        return samples
    
    def list_available_splits(self) -> Dict[str, List[str]]:
        """List all available data splits."""
        splits = {
            'train': [],
            'validation': [],
            'test': []
        }
        
        for file in self.index_dir.glob("*.txt"):
            filename = file.name
            if 'train' in filename:
                splits['train'].append(filename)
            elif 'validation' in filename or 'val' in filename:
                splits['validation'].append(filename)
            elif 'test' in filename:
                splits['test'].append(filename)
        
        return splits
    
    def get_statistics(self) -> Dict:
        """Get statistics about the dataset."""
        stats = {
            'classes': self.get_class_info(),
            'splits': {},
            'sequences': []
        }
        
        # Count samples in each split
        splits = self.list_available_splits()
        for split_type, files in splits.items():
            stats['splits'][split_type] = {}
            for file in files:
                try:
                    samples = self.parse_index_file(file)
                    stats['splits'][split_type][file] = len(samples)
                except Exception as e:
                    stats['splits'][split_type][file] = f"Error: {str(e)}"
        
        # List available sequences
        if self.images_dir.exists():
            seq_dirs = sorted([d for d in self.images_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('seq_')])
            for seq_dir in seq_dirs:
                seq_info = {
                    'name': seq_dir.name,
                    'has_left_frames': (seq_dir / 'left_frames').exists(),
                    'has_right_frames': (seq_dir / 'right_frames').exists(),
                    'has_labels': (seq_dir / 'labels').exists(),
                    'has_calibration': (seq_dir / 'camera_calibration.txt').exists()
                }
                
                # Count frames if directory exists
                if seq_info['has_left_frames']:
                    frames = list((seq_dir / 'left_frames').glob('*.png'))
                    seq_info['frame_count'] = len(frames)
                
                stats['sequences'].append(seq_info)
        
        return stats
    
    def convert_to_vqa_format(self, output_dir: str, 
                                 split_files: Optional[Dict[str, str]] = None,
                                 generate_questions: bool = True) -> Dict:
        """
        Convert EndoVis2018 data to VQA format compatible with the project.
        
        Args:
            output_dir: Output directory for converted data
            split_files: Dict mapping split names to index files
                        e.g., {'train': 'train_data.txt', 'test': 'test_data.txt'}
            generate_questions: Whether to generate VQA questions from segmentation labels
        
        Returns:
            Dict with conversion statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Default split files if not provided
        if split_files is None:
            split_files = {
                'train': 'train_data.txt',
                'validation': 'validation_data.txt',
                'test': 'test_data_final.txt'
            }
        
        results = {
            'output_dir': str(output_path),
            'converted_splits': {},
            'total_samples': 0
        }
        
        # Convert each split
        for split_name, index_file in split_files.items():
            if not (self.index_dir / index_file).exists():
                print(f"⚠️  Skipping {split_name}: {index_file} not found")
                continue
            
            print(f"\n📊 Converting {split_name} split from {index_file}...")
            samples = self.parse_index_file(index_file)
            
            vqa_samples = []
            for sample in samples:
                image_path = self.data_root / "data" / sample['image_path']
                label_path = self.data_root / "data" / sample['label_path']
                
                # Check if files exist
                if not image_path.exists():
                    continue
                
                # Create VQA sample
                vqa_sample = {
                    'image_id': f"endovis_seq{sample['seq']}_frame{sample['frame']}",
                    'image_path': str(image_path.relative_to(self.data_root / "data")),
                    'sequence': sample['seq'],
                    'frame': sample['frame'],
                    'dataset': 'EndoVis2018'
                }
                
                # Generate questions if requested
                if generate_questions and label_path.exists():
                    questions = self._generate_questions_from_segmentation(
                        label_path, sample['seq'], sample['frame']
                    )
                    vqa_sample['qa_pairs'] = questions
                else:
                    vqa_sample['qa_pairs'] = []
                
                vqa_samples.append(vqa_sample)
            
            # Save converted split
            output_file = output_path / f"{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(vqa_samples, f, indent=2)
            
            results['converted_splits'][split_name] = {
                'file': str(output_file),
                'samples': len(vqa_samples)
            }
            results['total_samples'] += len(vqa_samples)
            
            print(f"✅ Converted {len(vqa_samples)} samples to {output_file}")
        
        # Save metadata
        metadata = {
            'dataset': 'EndoVis2018',
            'source': 'https://github.com/mli0603/EndoVis2018',
            'description': 'Surgical scene segmentation dataset converted for VQA',
            'classes': self.classes,
            'conversion_info': results
        }
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Conversion complete! Metadata saved to {metadata_file}")
        return results
    
    def _generate_questions_from_segmentation(self, label_path: Path, 
                                             seq: str, frame: str) -> List[Dict]:
        """
        Generate VQA questions based on segmentation labels.
        
        This is a placeholder - you can enhance this to actually read the
        segmentation masks and generate questions about what's visible.
        """
        questions = [
            {
                'question': 'What type of surgical procedure is shown in this image?',
                'answer': 'Robotic surgery',
                'stage': 1,
                'stage_name': 'Initial Assessment'
            },
            {
                'question': 'Are there any surgical instruments visible in the image?',
                'answer': 'Yes, surgical instruments are visible',
                'stage': 2,
                'stage_name': 'Findings'
            },
            {
                'question': 'What anatomical structures are present in this surgical scene?',
                'answer': 'Kidney and small intestine are visible',
                'stage': 3,
                'stage_name': 'Clinical Context'
            }
        ]
        
        return questions


def main():
    parser = argparse.ArgumentParser(description='EndoVis2018 Dataset Loader')
    parser.add_argument('--data-root', type=str, 
                       default='/l/users/muhra.almahri/Surgical_COT/EndoVis2018',
                       help='Path to EndoVis2018 repository root')
    parser.add_argument('--action', type=str, required=True,
                       choices=['list', 'stats', 'convert', 'info'],
                       help='Action to perform')
    parser.add_argument('--output', type=str,
                       default='datasets/endovis2018',
                       help='Output directory for conversion')
    parser.add_argument('--split-files', type=str, nargs='+',
                       help='Custom split files (format: split_name:filename)')
    parser.add_argument('--no-questions', action='store_true',
                       help='Skip question generation during conversion')
    
    args = parser.parse_args()
    
    loader = EndoVis2018Loader(args.data_root)
    
    if args.action == 'list':
        print("📋 Available Data Splits:")
        print("=" * 60)
        splits = loader.list_available_splits()
        for split_type, files in splits.items():
            print(f"\n{split_type.upper()}:")
            for file in files:
                print(f"  - {file}")
    
    elif args.action == 'info':
        print("📊 Class Information:")
        print("=" * 60)
        class_info = loader.get_class_info()
        print(f"\nTotal Classes: {class_info['total_classes']}")
        print("\nClasses:")
        for cls in class_info['classes']:
            print(f"  {cls['classid']:2d}: {cls['name']:30s} (RGB: {cls['color']})")
    
    elif args.action == 'stats':
        print("📊 Dataset Statistics:")
        print("=" * 60)
        stats = loader.get_statistics()
        
        print(f"\nClasses: {stats['classes']['total_classes']}")
        print("\nSplits:")
        for split_type, files in stats['splits'].items():
            print(f"\n  {split_type.upper()}:")
            for file, count in files.items():
                print(f"    {file}: {count} samples")
        
        print(f"\nSequences: {len(stats['sequences'])}")
        for seq in stats['sequences'][:5]:  # Show first 5
            print(f"  {seq['name']}: frames={seq.get('frame_count', 'N/A')}, "
                  f"labels={'✓' if seq['has_labels'] else '✗'}")
        if len(stats['sequences']) > 5:
            print(f"  ... and {len(stats['sequences']) - 5} more")
    
    elif args.action == 'convert':
        # Parse custom split files if provided
        split_files = None
        if args.split_files:
            split_files = {}
            for item in args.split_files:
                if ':' in item:
                    split_name, filename = item.split(':', 1)
                    split_files[split_name] = filename
                else:
                    print(f"⚠️  Invalid format for split file: {item} (expected split_name:filename)")
        
        results = loader.convert_to_vqa_format(
            args.output,
            split_files=split_files,
            generate_questions=not args.no_questions
        )
        
        print("\n" + "=" * 60)
        print("✅ Conversion Summary:")
        print("=" * 60)
        print(f"Total samples: {results['total_samples']}")
        for split_name, info in results['converted_splits'].items():
            print(f"  {split_name}: {info['samples']} samples -> {info['file']}")


if __name__ == '__main__':
    main()

