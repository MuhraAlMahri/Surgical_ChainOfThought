#!/usr/bin/env python3
"""
Preprocess Kvasir surgical QA data to add proper instruction templates and question types.
This will transform your existing data to work better with instruction fine-tuning.
"""

import json
import os
import re
from typing import Dict, List, Tuple
from pathlib import Path


# Question type classification patterns
QUESTION_TYPE_PATTERNS = {
    'binary': [
        r'^(is|are|does|do|can|has|have|was|were|will)\s',
        r'\b(yes|no)\b.*\?$',
    ],
    'numeric': [
        r'^how many\s',
        r'^\s*\d+\s*$',  # Answer is just a number
    ],
    'mcq': [
        r'check all that',
        r'select all',
        r'choose.*option',
        r'\(a\)|\(b\)|\(c\)|\(d\)',
    ],
    'color': [
        r'what.*color',
        r'what.*colour',
    ],
    'size': [
        r'what.*size',
        r'how.*big',
        r'how.*large',
        r'diameter',
    ],
}


def classify_question_type(question: str, answer: str) -> str:
    """
    Classify question type based on question text and answer format.
    
    Returns:
        - 'binary': Yes/No questions
        - 'numeric': Number questions
        - 'mcq': Multiple choice
        - 'color': Color questions
        - 'size': Size/measurement questions
        - 'open_short': Short open-ended (1-3 words)
        - 'open_long': Long descriptive answers
    """
    question_lower = question.lower().strip()
    answer_lower = answer.lower().strip()
    
    # Check binary
    if answer_lower in ['yes', 'no']:
        return 'binary'
    
    # Check numeric
    if re.match(r'^\d+$', answer_lower):
        return 'numeric'
    
    # Check by patterns
    for qtype, patterns in QUESTION_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question_lower):
                return qtype
    
    # Check answer length for remaining
    word_count = len(answer.split())
    if word_count <= 3:
        return 'open_short'
    else:
        return 'open_long'


def get_instruction_template(question_type: str, question: str) -> str:
    """
    Get the appropriate instruction template based on question type.
    These templates tell the model EXACTLY how to format its response.
    """
    templates = {
        'binary': (
            "You are a surgical image analysis assistant. "
            "Answer the following question about the surgical/endoscopic image with ONLY 'yes' or 'no'. "
            "Do not provide any explanations, reasoning, or additional text.\n\n"
            f"Question: {question}\n"
            "Answer with only 'yes' or 'no':"
        ),
        
        'numeric': (
            "You are a surgical image analysis assistant. "
            "Answer the following question about the surgical/endoscopic image with ONLY a number. "
            "Do not include units, explanations, or additional text.\n\n"
            f"Question: {question}\n"
            "Answer with only the number:"
        ),
        
        'color': (
            "You are a surgical image analysis assistant. "
            "Answer the following question about the surgical/endoscopic image with ONLY the color name. "
            "Provide a single word color (e.g., 'red', 'pink', 'white', 'brown'). "
            "Do not provide explanations or additional text.\n\n"
            f"Question: {question}\n"
            "Answer with only the color:"
        ),
        
        'size': (
            "You are a surgical image analysis assistant. "
            "Answer the following question about the surgical/endoscopic image with ONLY the size or measurement. "
            "Use the format specified in the question (e.g., '5-10mm', '2cm'). "
            "Do not provide explanations or additional text.\n\n"
            f"Question: {question}\n"
            "Answer with only the size:"
        ),
        
        'mcq': (
            "You are a surgical image analysis assistant. "
            "Answer the following multiple choice question about the surgical/endoscopic image. "
            "Provide ONLY the answer term(s), separated by commas if multiple apply. "
            "Do not include explanations.\n\n"
            f"Question: {question}\n"
            "Answer:"
        ),
        
        'open_short': (
            "You are a surgical image analysis assistant. "
            "Answer the following question about the surgical/endoscopic image concisely. "
            "Provide a brief answer (1-3 words maximum). "
            "Do not provide unnecessary explanations.\n\n"
            f"Question: {question}\n"
            "Answer:"
        ),
        
        'open_long': (
            "You are a surgical image analysis assistant. "
            "Answer the following question about the surgical/endoscopic image. "
            "Provide a clear and accurate answer.\n\n"
            f"Question: {question}\n"
            "Answer:"
        ),
    }
    
    return templates.get(question_type, templates['open_short'])


def process_single_item(item: Dict) -> Dict:
    """
    Process a single QA item, adding instruction template and question type.
    """
    question = item['question']
    answer = item['answer']
    
    # Classify question type
    question_type = classify_question_type(question, answer)
    
    # Get instruction template
    instruction = get_instruction_template(question_type, question)
    
    # Create enhanced item
    enhanced_item = {
        **item,  # Keep all original fields
        'question_type': question_type,
        'instruction': instruction,
        'full_text': f"{instruction}\n{answer}"  # For training
    }
    
    return enhanced_item


def process_dataset_file(input_file: str, output_file: str):
    """
    Process entire dataset file, adding instruction templates and question types.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"ERROR: Expected list format, got {type(data)}")
        return
    
    print(f"Loaded {len(data)} items")
    
    # Process each item
    processed_data = []
    question_type_counts = {}
    
    for idx, item in enumerate(data):
        if idx % 1000 == 0:
            print(f"Processing item {idx}/{len(data)}...")
        
        enhanced_item = process_single_item(item)
        processed_data.append(enhanced_item)
        
        # Count question types
        qtype = enhanced_item['question_type']
        question_type_counts[qtype] = question_type_counts.get(qtype, 0) + 1
    
    # Save processed data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"\n✓ Saved {len(processed_data)} processed items to: {output_file}")
    print(f"\nQuestion Type Distribution:")
    for qtype, count in sorted(question_type_counts.items()):
        percentage = (count / len(processed_data)) * 100
        print(f"  {qtype:15s}: {count:5d} ({percentage:5.1f}%)")
    
    # Show a sample
    print(f"\n{'='*60}")
    print("SAMPLE PROCESSED ITEM:")
    print(f"{'='*60}")
    sample = processed_data[0]
    print(f"Original Question: {sample['question']}")
    print(f"Question Type: {sample['question_type']}")
    print(f"Answer: {sample['answer']}")
    print(f"\nInstruction Template:\n{sample['instruction']}")
    print(f"{'='*60}\n")


def main():
    """
    Main function to process train, validation, and test sets.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Add instruction templates to Kvasir surgical QA data")
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing train.json, val.json, test.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        default=['train.json', 'val.json', 'test.json'],
        help='List of files to process (default: train.json val.json test.json)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("KVASIR SURGICAL QA - INSTRUCTION PREPROCESSING")
    print("="*60)
    
    # Process each file
    for filename in args.files:
        input_file = os.path.join(args.input_dir, filename)
        
        if not os.path.exists(input_file):
            print(f"\n⚠ WARNING: File not found: {input_file}")
            continue
        
        # Create output filename with '_instructed' suffix
        base_name = filename.replace('.json', '')
        output_filename = f"{base_name}_instructed.json"
        output_file = os.path.join(args.output_dir, output_filename)
        
        process_dataset_file(input_file, output_file)
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nProcessed files saved in: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review the sample outputs above")
    print("2. Use the *_instructed.json files for training")
    print("3. Update your training script to use the 'instruction' field")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
