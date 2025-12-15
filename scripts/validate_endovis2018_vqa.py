#!/usr/bin/env python3
"""
Validate EndoVis2018 VQA dataset against ground truth segmentation masks.

This script:
1. Loads Q&A pairs from the dataset
2. For each Q&A pair, loads the corresponding segmentation mask
3. Detects classes in the mask (ground truth)
4. Generates expected answers based on mask content
5. Compares expected vs actual answers
6. Reports accuracy and errors
"""

import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

BASE_DIR = Path("/l/users/muhra.almahri/Surgical_COT")
ENDOVIS_DIR = BASE_DIR / "EndoVis2018" / "data"


def load_class_colors() -> Dict[str, List[int]]:
    """Load class color mappings from labels.json."""
    labels_file = ENDOVIS_DIR / "labels.json"
    if not labels_file.exists():
        raise FileNotFoundError(f"labels.json not found at {labels_file}")
    
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    class_colors = {}
    for label in labels:
        class_colors[label['name']] = label['color']
    
    return class_colors


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
        color_array = np.array(color)
        matches = np.all(mask_flat == color_array, axis=1)
        if np.any(matches):
            present_classes.append(class_name)
    
    return present_classes


def generate_expected_answer(present_classes: List[str], category: str) -> str:
    """Generate expected answer based on present classes and category."""
    if category == 'INSTRUMENT_DETECTION':
        instruments = [c for c in present_classes if 'instrument' in c.lower()]
        if instruments:
            instrument_names = [c.replace('instrument-', '').replace('-', ' ') for c in instruments]
            return '; '.join(instrument_names) if len(instrument_names) > 1 else instrument_names[0]
        else:
            return 'none'
    
    elif category == 'ANATOMY_DETECTION':
        organs = [c for c in present_classes if any(org in c.lower() for org in ['kidney', 'intestine', 'tissue'])]
        if organs:
            organ_names = [c.replace('-', ' ') for c in organs]
            return '; '.join(organ_names) if len(organ_names) > 1 else organ_names[0]
        else:
            return 'none'
    
    elif category == 'INSTRUMENT_COUNT':
        instruments = [c for c in present_classes if 'instrument' in c.lower()]
        return str(len(instruments))
    
    elif category == 'PROCEDURE_TYPE':
        return 'Robotic surgery'
    
    return 'unknown'


def find_mask_path(seq: str, frame: str) -> Path:
    """Find segmentation mask path, trying multiple possible locations."""
    # Try primary path
    mask_path = ENDOVIS_DIR / "images" / "miccai_challenge_release_2" / f"seq_{seq}" / "labels" / f"frame{frame}.png"
    if mask_path.exists():
        return mask_path
    
    # Try alternative path
    mask_path = ENDOVIS_DIR / "images" / f"seq_{seq}" / "labels" / f"frame{frame}.png"
    if mask_path.exists():
        return mask_path
    
    return None


def validate_sample(sample: Dict, class_colors: Dict) -> Tuple[bool, str, str, List[str]]:
    """Validate a single Q&A sample against ground truth mask."""
    seq = sample.get('sequence', '')
    frame = sample.get('frame', '')
    category = sample.get('category', '')
    actual_answer = sample.get('answer', '')
    
    # Find mask
    mask_path = find_mask_path(seq, frame)
    if mask_path is None:
        return None, f"Mask not found", actual_answer, []
    
    # Load mask and detect classes
    mask = load_segmentation_mask(mask_path)
    if mask is None:
        return None, f"Could not load mask", actual_answer, []
    
    present_classes = get_present_classes(mask, class_colors)
    
    # Generate expected answer
    expected_answer = generate_expected_answer(present_classes, category)
    
    # Compare
    is_correct = (expected_answer == actual_answer)
    
    return is_correct, expected_answer, actual_answer, present_classes


def validate_dataset(qa_file: Path, sample_size: int = None, verbose: bool = False):
    """Validate entire dataset against ground truth."""
    print("=" * 80)
    print("EndoVis2018 VQA Dataset Validation")
    print("=" * 80)
    print()
    
    # Load class colors (ground truth definitions)
    class_colors = load_class_colors()
    print(f"✅ Loaded {len(class_colors)} ground truth class definitions")
    print()
    
    # Load Q&A samples
    with open(qa_file, 'r') as f:
        samples = [json.loads(line) for line in f.readlines()]
    
    print(f"✅ Loaded {len(samples)} Q&A pairs from {qa_file.name}")
    print()
    
    # Sample if requested
    if sample_size and sample_size < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, sample_size)
        print(f"📊 Validating random sample of {sample_size} Q&A pairs")
        print()
    
    # Validate
    results = {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'missing_mask': 0,
        'by_category': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'errors': []
    }
    
    print("Validating samples...")
    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")
        
        category = sample.get('category', 'unknown')
        result = validate_sample(sample, class_colors)
        
        if result[0] is None:
            # Missing mask
            results['missing_mask'] += 1
            if verbose:
                results['errors'].append({
                    'sample': sample.get('image_id', 'unknown'),
                    'category': category,
                    'error': result[1]
                })
        else:
            is_correct, expected, actual, present_classes = result
            results['total'] += 1
            results['by_category'][category]['total'] += 1
            
            if is_correct:
                results['correct'] += 1
                results['by_category'][category]['correct'] += 1
            else:
                results['incorrect'] += 1
                results['errors'].append({
                    'sample': sample.get('image_id', 'unknown'),
                    'category': category,
                    'expected': expected,
                    'actual': actual,
                    'present_classes': present_classes
                })
    
    # Print results
    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()
    
    if results['total'] > 0:
        accuracy = (results['correct'] / results['total']) * 100
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"  Correct: {results['correct']:,} / {results['total']:,}")
        print(f"  Incorrect: {results['incorrect']:,}")
        print()
    
    if results['missing_mask'] > 0:
        print(f"⚠️  Samples with missing masks: {results['missing_mask']:,}")
        print("   (These frames may not have segmentation labels)")
        print()
    
    print("Accuracy by Category:")
    for category in sorted(results['by_category'].keys()):
        cat_results = results['by_category'][category]
        if cat_results['total'] > 0:
            cat_accuracy = (cat_results['correct'] / cat_results['total']) * 100
            print(f"  {category:25s}: {cat_accuracy:6.2f}% ({cat_results['correct']}/{cat_results['total']})")
    print()
    
    # Show errors
    if results['errors']:
        print(f"Errors found: {len(results['errors'])}")
        print()
        print("First 10 errors:")
        for error in results['errors'][:10]:
            if 'expected' in error:
                print(f"  {error['sample']} [{error['category']}]")
                print(f"    Expected: '{error['expected']}'")
                print(f"    Actual:   '{error['actual']}'")
                print(f"    Classes in mask: {error['present_classes']}")
            else:
                print(f"  {error['sample']} [{error['category']}]: {error['error']}")
        
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    else:
        print("✅ No errors found! All validated samples are correct.")
    
    print()
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate EndoVis2018 VQA dataset')
    parser.add_argument('--qa_file', type=str,
                       default='corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl',
                       help='Path to Q&A JSONL file')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Validate only a random sample of this size (for quick check)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed error information')
    
    args = parser.parse_args()
    
    qa_file = BASE_DIR / args.qa_file
    if not qa_file.exists():
        print(f"❌ Q&A file not found: {qa_file}")
        return
    
    validate_dataset(qa_file, args.sample_size, args.verbose)


if __name__ == '__main__':
    main()
















