#!/usr/bin/env python3
"""
Data Standardization Script for Surgical CoT Dataset

This script processes all 6 surgical datasets into a unified structure:
- Cataract-1K: Ophthalmic surgery videos
- EgoSurgery: Egocentric open surgery images  
- HeiChole: Laparoscopic cholecystectomy videos
- Kvasir-VQA: Endoscopy images with VQA pairs
- PitVis: Endoscopic neurosurgery videos
- Surg-396k: General endoscopic surgery images

Output: Unified structure in data/standardized/ with data_manifest.json
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetStandardizer:
    """Handles standardization of multiple surgical datasets into unified format."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.datasets = self.config['datasets']
        self.output_dir = Path(self.config['output']['standardized_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported file extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        self.text_extensions = {'.txt', '.json', '.xml', '.csv'}
        
        # Initialize manifest
        self.manifest = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_samples': 0,
                'datasets': {}
            },
            'samples': []
        }
    
    def find_files_recursive(self, directory: Path, extensions: set) -> List[Path]:
        """Recursively find files with specified extensions."""
        files = []
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return files
            
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))
            files.extend(directory.rglob(f"*{ext.upper()}"))
        return files
    
    def extract_frame_from_video(self, video_path: Path, output_dir: Path, frame_number: int = 0) -> Optional[Path]:
        """Extract a frame from video for processing."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_path = output_dir / f"{video_path.stem}_frame_{frame_number:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                return frame_path
        except ImportError:
            logger.warning("OpenCV not available, skipping video frame extraction")
        except Exception as e:
            logger.error(f"Error extracting frame from {video_path}: {e}")
        return None
    
    def process_cataract1k(self) -> List[Dict]:
        """Process Cataract-1K dataset (ophthalmic surgery videos)."""
        logger.info("Processing Cataract-1K dataset...")
        samples = []
        dataset_dir = Path(self.datasets['cataract1k_dir'])
        
        if not dataset_dir.exists():
            logger.warning(f"Cataract-1K directory not found: {dataset_dir}")
            return samples
        
        # Find all video files
        video_files = self.find_files_recursive(dataset_dir, self.video_extensions)
        
        for video_path in video_files:
            sample_id = f"cataract1k_{video_path.stem}"
            output_sample_dir = self.output_dir / "train" / "cataract1k" / sample_id
            output_sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy video file
            video_dest = output_sample_dir / f"{sample_id}.mp4"
            shutil.copy2(video_path, video_dest)
            
            # Extract frame for processing
            frame_path = self.extract_frame_from_video(video_path, output_sample_dir)
            
            # Create basic report
            report_path = output_sample_dir / "report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Ophthalmic surgery video from Cataract-1K dataset.\n")
                f.write(f"Original path: {video_path}\n")
                f.write(f"Video duration: [To be extracted]\n")
                f.write(f"Procedure type: Cataract surgery\n")
            
            samples.append({
                'dataset_name': 'cataract1k',
                'sample_id': sample_id,
                'split': 'train',  # Will be updated by split_data.py
                'image_path': str(frame_path) if frame_path else str(video_dest),
                'video_path': str(video_dest),
                'report_path': str(report_path),
                'original_path': str(video_path)
            })
        
        logger.info(f"Processed {len(samples)} samples from Cataract-1K")
        return samples
    
    def process_egosurgery(self) -> List[Dict]:
        """Process EgoSurgery dataset (egocentric open surgery images)."""
        logger.info("Processing EgoSurgery dataset...")
        samples = []
        dataset_dir = Path(self.datasets['egosurgery_dir'])
        
        if not dataset_dir.exists():
            logger.warning(f"EgoSurgery directory not found: {dataset_dir}")
            return samples
        
        # Find all image files
        image_files = self.find_files_recursive(dataset_dir, self.image_extensions)
        
        for img_path in image_files:
            sample_id = f"egosurgery_{img_path.stem}"
            output_sample_dir = self.output_dir / "train" / "egosurgery" / sample_id
            output_sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image file
            img_dest = output_sample_dir / f"{sample_id}.jpg"
            shutil.copy2(img_path, img_dest)
            
            # Create basic report
            report_path = output_sample_dir / "report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Egocentric open surgery image from EgoSurgery dataset.\n")
                f.write(f"Original path: {img_path}\n")
                f.write(f"Procedure type: Open surgery (egocentric view)\n")
                f.write(f"View type: First-person perspective\n")
            
            samples.append({
                'dataset_name': 'egosurgery',
                'sample_id': sample_id,
                'split': 'train',
                'image_path': str(img_dest),
                'video_path': None,
                'report_path': str(report_path),
                'original_path': str(img_path)
            })
        
        logger.info(f"Processed {len(samples)} samples from EgoSurgery")
        return samples
    
    def process_heichole(self) -> List[Dict]:
        """Process HeiChole dataset (laparoscopic cholecystectomy videos)."""
        logger.info("Processing HeiChole dataset...")
        samples = []
        dataset_dir = Path(self.datasets['heichole_dir'])
        
        if not dataset_dir.exists():
            logger.warning(f"HeiChole directory not found: {dataset_dir}")
            return samples
        
        # Find all video files
        video_files = self.find_files_recursive(dataset_dir, self.video_extensions)
        
        for video_path in video_files:
            sample_id = f"heichole_{video_path.stem}"
            output_sample_dir = self.output_dir / "train" / "heichole" / sample_id
            output_sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy video file
            video_dest = output_sample_dir / f"{sample_id}.mp4"
            shutil.copy2(video_path, video_dest)
            
            # Extract frame for processing
            frame_path = self.extract_frame_from_video(video_path, output_sample_dir)
            
            # Create basic report
            report_path = output_sample_dir / "report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Laparoscopic cholecystectomy video from HeiChole dataset.\n")
                f.write(f"Original path: {video_path}\n")
                f.write(f"Procedure type: Laparoscopic cholecystectomy\n")
                f.write(f"Anatomy: Gallbladder and surrounding structures\n")
            
            samples.append({
                'dataset_name': 'heichole',
                'sample_id': sample_id,
                'split': 'train',
                'image_path': str(frame_path) if frame_path else str(video_dest),
                'video_path': str(video_dest),
                'report_path': str(report_path),
                'original_path': str(video_path)
            })
        
        logger.info(f"Processed {len(samples)} samples from HeiChole")
        return samples
    
    def process_kvasirvqa(self) -> List[Dict]:
        """Process Kvasir-VQA dataset (endoscopy images with VQA pairs)."""
        logger.info("Processing Kvasir-VQA dataset...")
        samples = []
        dataset_dir = Path(self.datasets['kvasirvqa_dir'])
        
        if not dataset_dir.exists():
            logger.warning(f"Kvasir-VQA directory not found: {dataset_dir}")
            return samples
        
        # Find all image files
        image_files = self.find_files_recursive(dataset_dir, self.image_extensions)
        
        # Look for VQA annotation files
        annotation_files = self.find_files_recursive(dataset_dir, {'.json', '.csv'})
        vqa_data = {}
        
        # Load VQA annotations if available
        for ann_file in annotation_files:
            try:
                if ann_file.suffix == '.json':
                    with open(ann_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if 'image' in item and 'question' in item:
                                    vqa_data[item['image']] = item
                        elif isinstance(data, dict):
                            vqa_data.update(data)
                elif ann_file.suffix == '.csv':
                    import pandas as pd
                    df = pd.read_csv(ann_file)
                    for _, row in df.iterrows():
                        if 'image' in row and 'question' in row:
                            vqa_data[row['image']] = row.to_dict()
            except Exception as e:
                logger.warning(f"Could not load annotation file {ann_file}: {e}")
        
        for img_path in image_files:
            sample_id = f"kvasirvqa_{img_path.stem}"
            output_sample_dir = self.output_dir / "train" / "kvasirvqa" / sample_id
            output_sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image file
            img_dest = output_sample_dir / f"{sample_id}.jpg"
            shutil.copy2(img_path, img_dest)
            
            # Create report with VQA data if available
            report_path = output_sample_dir / "report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Endoscopy image from Kvasir-VQA dataset.\n")
                f.write(f"Original path: {img_path}\n")
                f.write(f"Procedure type: Endoscopic examination\n")
                
                # Add VQA data if available
                img_name = img_path.name
                if img_name in vqa_data:
                    vqa = vqa_data[img_name]
                    f.write(f"\nVQA Data:\n")
                    f.write(f"Question: {vqa.get('question', 'N/A')}\n")
                    f.write(f"Answer: {vqa.get('answer', 'N/A')}\n")
                    if 'category' in vqa:
                        f.write(f"Category: {vqa['category']}\n")
            
            samples.append({
                'dataset_name': 'kvasirvqa',
                'sample_id': sample_id,
                'split': 'train',
                'image_path': str(img_dest),
                'video_path': None,
                'report_path': str(report_path),
                'original_path': str(img_path),
                'vqa_question': vqa_data.get(img_name, {}).get('question'),
                'vqa_answer': vqa_data.get(img_name, {}).get('answer')
            })
        
        logger.info(f"Processed {len(samples)} samples from Kvasir-VQA")
        return samples
    
    def process_pitvis(self) -> List[Dict]:
        """Process PitVis dataset (endoscopic neurosurgery videos)."""
        logger.info("Processing PitVis dataset...")
        samples = []
        dataset_dir = Path(self.datasets['pitvis_dir'])
        
        if not dataset_dir.exists():
            logger.warning(f"PitVis directory not found: {dataset_dir}")
            return samples
        
        # Find all video files
        video_files = self.find_files_recursive(dataset_dir, self.video_extensions)
        
        for video_path in video_files:
            sample_id = f"pitvis_{video_path.stem}"
            output_sample_dir = self.output_dir / "train" / "pitvis" / sample_id
            output_sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy video file
            video_dest = output_sample_dir / f"{sample_id}.mp4"
            shutil.copy2(video_path, video_dest)
            
            # Extract frame for processing
            frame_path = self.extract_frame_from_video(video_path, output_sample_dir)
            
            # Create basic report
            report_path = output_sample_dir / "report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Endoscopic neurosurgery video from PitVis dataset.\n")
                f.write(f"Original path: {video_path}\n")
                f.write(f"Procedure type: Pituitary surgery (endoscopic)\n")
                f.write(f"Anatomy: Pituitary gland and surrounding structures\n")
            
            samples.append({
                'dataset_name': 'pitvis',
                'sample_id': sample_id,
                'split': 'train',
                'image_path': str(frame_path) if frame_path else str(video_dest),
                'video_path': str(video_dest),
                'report_path': str(report_path),
                'original_path': str(video_path)
            })
        
        logger.info(f"Processed {len(samples)} samples from PitVis")
        return samples
    
    def process_surg396k(self) -> List[Dict]:
        """Process Surg-396k dataset (general endoscopic surgery images)."""
        logger.info("Processing Surg-396k dataset...")
        samples = []
        dataset_dir = Path(self.datasets['surg396k_dir'])
        
        if not dataset_dir.exists():
            logger.warning(f"Surg-396k directory not found: {dataset_dir}")
            return samples
        
        # Find all image files
        image_files = self.find_files_recursive(dataset_dir, self.image_extensions)
        
        for img_path in image_files:
            sample_id = f"surg396k_{img_path.stem}"
            output_sample_dir = self.output_dir / "train" / "surg396k" / sample_id
            output_sample_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image file
            img_dest = output_sample_dir / f"{sample_id}.jpg"
            shutil.copy2(img_path, img_dest)
            
            # Create basic report
            report_path = output_sample_dir / "report.txt"
            with open(report_path, 'w') as f:
                f.write(f"General endoscopic surgery image from Surg-396k dataset.\n")
                f.write(f"Original path: {img_path}\n")
                f.write(f"Procedure type: Endoscopic surgery\n")
                f.write(f"View type: Laparoscopic/endoscopic\n")
            
            samples.append({
                'dataset_name': 'surg396k',
                'sample_id': sample_id,
                'split': 'train',
                'image_path': str(img_dest),
                'video_path': None,
                'report_path': str(report_path),
                'original_path': str(img_path)
            })
        
        logger.info(f"Processed {len(samples)} samples from Surg-396k")
        return samples
    
    def process_all_datasets(self) -> None:
        """Process all datasets and create unified manifest."""
        logger.info("Starting dataset standardization process...")
        logger.info("This addresses the 'Limited Dataset Availability' limitation in current SOTA papers")
        
        all_samples = []
        
        # Process each dataset
        dataset_processors = {
            'cataract1k': self.process_cataract1k,
            'egosurgery': self.process_egosurgery,
            'heichole': self.process_heichole,
            'kvasirvqa': self.process_kvasirvqa,
            'pitvis': self.process_pitvis,
            'surg396k': self.process_surg396k
        }
        
        successful_datasets = 0
        for dataset_name, processor in dataset_processors.items():
            try:
                logger.info(f"Processing {dataset_name}...")
                samples = processor()
                all_samples.extend(samples)
                self.manifest['metadata']['datasets'][dataset_name] = len(samples)
                successful_datasets += 1
                logger.info(f"✓ {dataset_name}: {len(samples)} samples processed")
            except Exception as e:
                logger.error(f"✗ Error processing {dataset_name}: {e}")
                self.manifest['metadata']['datasets'][dataset_name] = 0
        
        # Update manifest
        self.manifest['samples'] = all_samples
        self.manifest['metadata']['total_samples'] = len(all_samples)
        self.manifest['metadata']['successful_datasets'] = successful_datasets
        self.manifest['metadata']['failed_datasets'] = len(dataset_processors) - successful_datasets
        
        # Save manifest
        manifest_path = self.output_dir / "data_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        logger.info(f"Standardization complete! Processed {len(all_samples)} samples total.")
        logger.info(f"Manifest saved to: {manifest_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("SURGICAL CoT DATASET STANDARDIZATION SUMMARY")
        print("="*60)
        print("Addressing 'Limited Dataset Availability' in current SOTA papers")
        print("-" * 60)
        for dataset_name, count in self.manifest['metadata']['datasets'].items():
            status = "✓" if count > 0 else "✗"
            print(f"{status} {dataset_name:15}: {count:6} samples")
        print("-" * 60)
        print(f"✓ Successful datasets: {successful_datasets}/{len(dataset_processors)}")
        print(f"✓ Total samples: {len(all_samples):,}")
        print(f"✓ Manifest file: {manifest_path}")
        print("="*60)
        
        if successful_datasets == len(dataset_processors):
            print("🎉 ALL DATASETS SUCCESSFULLY PROCESSED!")
            print("Ready for CoT generation with Qwen2.5-VL-72B-Instruct")
        else:
            print(f"⚠️  {len(dataset_processors) - successful_datasets} datasets failed - check logs")

def main():
    """Main function to run dataset standardization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize surgical datasets")
    parser.add_argument("--config", default="configs/paths.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--datasets", nargs="+", 
                       choices=['cataract1k', 'egosurgery', 'heichole', 'kvasirvqa', 'pitvis', 'surg396k'],
                       help="Specific datasets to process (default: all)")
    
    args = parser.parse_args()
    
    # Initialize standardizer
    standardizer = DatasetStandardizer(args.config)
    
    # Process datasets
    if args.datasets:
        # Process specific datasets only
        for dataset_name in args.datasets:
            processor = getattr(standardizer, f"process_{dataset_name}")
            processor()
    else:
        # Process all datasets
        standardizer.process_all_datasets()

if __name__ == "__main__":
    main()
