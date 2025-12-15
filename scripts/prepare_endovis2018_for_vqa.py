#!/usr/bin/env python3
"""
Prepare EndoVis2018 dataset for VQA experiments
Converts segmentation dataset to VQA format (JSONL) similar to Kvasir-VQA

This script:
1. Uses the organized EndoVis2018 images
2. Generates VQA questions from segmentation masks
3. Creates train/val/test JSONL files in the same format as Kvasir-VQA
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from collections import defaultdict

BASE_DIR = Path("/l/users/muhra.almahri/Surgical_COT")
ENDOVIS_DIR = BASE_DIR / "EndoVis2018" / "data"
ORGANIZED_DIR = BASE_DIR / "datasets" / "EndoVis2018" / "raw"
INDEX_DIR = ENDOVIS_DIR / "index"


def load_segmentation_mask(mask_path: Path) -> np.ndarray:
    """Load segmentation mask image."""
    if not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert('RGB')
    return np.array(mask)


def get_present_classes(mask: np.ndarray, class_colors: Dict) -> List[str]:
    """Detect which classes are present in the segmentation mask."""
    if mask is None:
        return []
    
    present_classes = []
    mask_flat = mask.reshape(-1, 3)
    
    for class_name, color in class_colors.items():
        # Check if this color exists in the mask
        color_array = np.array(color)
        matches = np.all(mask_flat == color_array, axis=1)
        if np.any(matches):
            present_classes.append(class_name)
    
    return present_classes


def get_instruction_template(category: str, question: str, question_type: str) -> str:
    """Generate instruction template based on category."""
    # General instruction (same for all)
    general_instruction = """You are a surgical image analysis assistant analysing a robotic surgical scene image.

Instructions:
- Select your answer(s) ONLY from the provided candidate list
- For multi-label questions: Select ALL applicable items, separated by semicolons (;)
- For single-choice questions: Select EXACTLY one option
- Output format: item1; item2; item3 (for multi-label) or item1 (for single-choice)

"""
    
    # Category-specific templates
    if category == 'INSTRUMENT_DETECTION':
        candidates = ['instrument-shaft', 'instrument-clasper', 'instrument-wrist', 'basket', 'biopsy forceps', 'catheter', 'dilator', 'grasping forceps', 'hemoclip', 'injection needle', 'knife', 'metal clip', 'polyp snare', 'retrieval net', 'scissors', 'suction-instrument', 'suturing-needle', 'thread', 'clamps', 'ultrasound-probe', 'none']
        return general_instruction + f"Question: {question}\nQuestion Type: {question_type}\nCandidates: {candidates}\nAnswer:"
    
    elif category == 'ANATOMY_DETECTION':
        candidates = ['kidney-parenchyma', 'covered-kidney', 'small-intestine', 'background-tissue', 'none']
        return general_instruction + f"Question: {question}\nQuestion Type: {question_type}\nCandidates: {candidates}\nAnswer:"
    
    elif category == 'INSTRUMENT_COUNT':
        candidates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
        return general_instruction + f"Question: {question}\nQuestion Type: {question_type}\nCandidates: {candidates}\nAnswer:"
    
    elif category == 'PROCEDURE_TYPE':
        candidates = ['Robotic surgery', 'Laparoscopic surgery', 'Endoscopic surgery', 'Open surgery', 'Minimally invasive surgery']
        return general_instruction + f"Question: {question}\nQuestion Type: {question_type}\nCandidates: {candidates}\nAnswer:"
    
    else:
        # Fallback
        return general_instruction + f"Question: {question}\nAnswer:"


def generate_vqa_questions(present_classes: List[str], seq: str, frame: str) -> List[Dict]:
    """Generate VQA questions based on present classes."""
    questions = []
    
    # Question 1: Instrument detection
    instruments = [c for c in present_classes if 'instrument' in c.lower()]
    if instruments:
        instrument_names = [c.replace('instrument-', '').replace('-', ' ') for c in instruments]
        answer = '; '.join(instrument_names) if len(instrument_names) > 1 else instrument_names[0] if instrument_names else 'none'
    else:
        answer = 'none'
    
    question_text = 'What surgical instruments are visible in this image?'
    instruction = get_instruction_template('INSTRUMENT_DETECTION', question_text, 'multi_label')
    questions.append({
        'question': question_text,
        'answer': answer,
        'instruction': instruction,
        'question_type': 'instrument_detection',
        'category': 'INSTRUMENT_DETECTION',
        'is_multi_label': True
    })
    
    # Question 2: Organ/anatomy detection
    organs = [c for c in present_classes if any(org in c.lower() for org in ['kidney', 'intestine', 'tissue'])]
    if organs:
        organ_names = [c.replace('-', ' ') for c in organs]
        answer = '; '.join(organ_names) if len(organ_names) > 1 else organ_names[0]
    else:
        answer = 'none'
    
    question_text = 'What anatomical structures are visible in this surgical scene?'
    instruction = get_instruction_template('ANATOMY_DETECTION', question_text, 'multi_label')
    questions.append({
        'question': question_text,
        'answer': answer,
        'instruction': instruction,
        'question_type': 'anatomy_detection',
        'category': 'ANATOMY_DETECTION',
        'is_multi_label': True
    })
    
    # Question 3: Instrument count
    question_text = 'How many surgical instruments are visible?'
    instruction = get_instruction_template('INSTRUMENT_COUNT', question_text, 'single_choice')
    questions.append({
        'question': question_text,
        'answer': str(len(instruments)),
        'instruction': instruction,
        'question_type': 'instrument_count',
        'category': 'INSTRUMENT_COUNT',
        'is_multi_label': False
    })
    
    # Question 4: Procedure type
    question_text = 'What type of surgical procedure is shown in this image?'
    instruction = get_instruction_template('PROCEDURE_TYPE', question_text, 'single_choice')
    questions.append({
        'question': question_text,
        'answer': 'Robotic surgery',
        'instruction': instruction,
        'question_type': 'procedure_type',
        'category': 'PROCEDURE_TYPE',
        'is_multi_label': False
    })
    
    return questions


def load_class_colors():
    """Load class color mappings from labels.json."""
    labels_file = ENDOVIS_DIR / "labels.json"
    if not labels_file.exists():
        # Default colors if file doesn't exist
        return {
            'background-tissue': [0, 0, 0],
            'instrument-shaft': [0, 0, 255],
            'instrument-clasper': [0, 255, 0],
            'instrument-wrist': [255, 0, 0],
            'kidney-parenchyma': [255, 255, 0],
            'covered-kidney': [255, 0, 255],
            'thread': [0, 255, 255],
            'clamps': [128, 128, 128],
            'suturing-needle': [255, 128, 0],
            'suction-instrument': [128, 255, 0],
            'small-intestine': [0, 128, 255],
            'ultrasound-probe': [255, 0, 128]
        }
    
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    class_colors = {}
    for label in labels:
        class_colors[label['name']] = label['color']
    
    return class_colors


def create_vqa_samples(split_name: str, split_file: str, image_dir: Path, class_colors: Dict) -> List[Dict]:
    """Create VQA samples for a split."""
    samples = []
    
    # Load split index
    index_path = INDEX_DIR / split_file
    if not index_path.exists():
        print(f"⚠️  Split file not found: {index_path}")
        return samples
    
    print(f"\n📊 Processing {split_name} split from {split_file}...")
    
    with open(index_path, 'r') as f:
        lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            
            seq = parts[0].strip()
            frame = parts[1].strip().zfill(3)
            
            # Image path - try both naming conventions
            image_filename1 = f"endovis_seq{seq}_frame{frame}.png"  # endovis_seq1_frame000.png
            image_filename2 = f"endovis_seq_{seq}_frame{frame}.png"  # endovis_seq_1_frame000.png
            image_path = image_dir / image_filename1
            if not image_path.exists():
                image_path = image_dir / image_filename2
                image_filename = image_filename2
            else:
                image_filename = image_filename1
            
            if not image_path.exists():
                continue
            
            # Label path (if available) - try both possible paths
            label_path = ENDOVIS_DIR / "images" / "miccai_challenge_release_2" / f"seq_{seq}" / "labels" / f"frame{frame}.png"
            if not label_path.exists():
                # Fallback to alternative path structure
                label_path = ENDOVIS_DIR / "images" / f"seq_{seq}" / "labels" / f"frame{frame}.png"
            
            # Load segmentation mask and detect classes
            mask = load_segmentation_mask(label_path)
            present_classes = get_present_classes(mask, class_colors) if mask is not None else []
            
            # Generate VQA questions
            qa_pairs = generate_vqa_questions(present_classes, seq, frame)
            
            # Create one sample per question
            for qa in qa_pairs:
                sample = {
                    'image_id': f"endovis_seq{seq}_frame{frame}",
                    'image_filename': image_filename,
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'instruction': qa.get('instruction', qa['question']),  # Use instruction if available
                    'question_type': qa['question_type'],
                    'category': qa['category'],
                    'is_multi_label': qa.get('is_multi_label', False),
                    'sequence': seq,
                    'frame': frame,
                    'dataset': 'EndoVis2018'
                }
                samples.append(sample)
    
    print(f"✅ Created {len(samples)} VQA samples for {split_name}")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Prepare EndoVis2018 for VQA experiments')
    parser.add_argument('--output_dir', type=str, 
                       default='corrected_1-5_experiments/datasets/endovis2018_vqa',
                       help='Output directory for JSONL files')
    parser.add_argument('--image_dir', type=str,
                       default='datasets/EndoVis2018/raw/images',
                       help='Directory with organized images')
    parser.add_argument('--use_proper_split', action='store_true', default=True,
                       help='Use proper sequence-based split')
    
    args = parser.parse_args()
    
    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_dir = BASE_DIR / args.image_dir
    
    print("=" * 60)
    print("Preparing EndoVis2018 for VQA Experiments")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Image directory: {image_dir}")
    print()
    
    # Load class colors
    class_colors = load_class_colors()
    print(f"Loaded {len(class_colors)} class definitions")
    
    # Determine split files
    if args.use_proper_split:
        split_files = {
            'train': 'train_data_sequence_based.txt',
            'validation': 'validation_data_sequence_based.txt',
            'test': 'test_data_sequence_based.txt'
        }
        print("Using proper sequence-based split (no overlap)")
    else:
        split_files = {
            'train': 'train_data.txt',
            'validation': 'validation_data.txt',
            'test': 'test_data_final.txt'
        }
        print("Using original split files")
    
    # Process each split
    all_samples = {}
    for split_name, split_file in split_files.items():
        samples = create_vqa_samples(split_name, split_file, image_dir, class_colors)
        all_samples[split_name] = samples
        
        # Save as JSONL
        output_file = output_dir / f"{split_name}.jsonl"
        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"✅ Saved {len(samples)} samples to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = sum(len(s) for s in all_samples.values())
    print(f"Total VQA samples: {total:,}")
    for split_name, samples in all_samples.items():
        print(f"  {split_name}: {len(samples):,} samples")
    print(f"\n✅ All files saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()


