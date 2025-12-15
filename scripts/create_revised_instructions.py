#!/usr/bin/env python3
"""
REDESIGNED Instruction Templates for Kvasir-VQA
Following advisor feedback to fix instruction-GT mismatches and add explicit candidates.

Key Changes:
1. Separate close-ended (with candidates) vs open-ended (with constraints)
2. Explicit candidate lists for ALL close-ended questions
3. Proper multi-label support for "check all that apply"
4. Controlled vocabulary/length for open-ended questions
5. Output format specification matching GT format
"""

import json
import re
from typing import Dict, List, Optional, Set
from pathlib import Path


# ============================================================================
# FIXED CANDIDATE LISTS (Based on Dataset Analysis)
# ============================================================================

CANDIDATE_LISTS = {
    # Abnormality types (multi-label - "check all that apply")
    'abnormality': [
        'polyp',
        'ulcerative colitis', 
        'oesophagitis',
        'barretts',
        'hemorrhoids',
        'short-segment barretts',
        'erosion',
        'normal'
    ],
    
    # Instruments (multi-label - "check all that apply")
    'instrument': [
        'biopsy forceps',
        'metal clip',
        'polyp snare',
        'injection needle',
        'tube',
        'none'
    ],
    
    # Anatomical landmarks (multi-label - "check all that apply")
    'anatomical_landmark': [
        'cecum',
        'ileum',
        'pylorus',
        'z-line',
        'none'
    ],
    
    # Procedure type (single choice)
    'procedure': [
        'colonoscopy',
        'gastroscopy',
        'capsule endoscopy'
    ],
    
    # Polyp type - Paris classification (single choice or multi-label)
    'polyp_type': [
        'none',
        'paris iia',
        'paris ip',
        'paris is'
    ],
    
    # Size categories (can be multi-label if multiple polyps)
    'size': [
        '<5mm',
        '5-10mm',
        '11-20mm',
        '>20mm'
    ],
    
    # Detection difficulty (single choice with "not relevant" option)
    'detection_difficulty': [
        'yes',
        'no',
        'not relevant'
    ],
    
    # Polyp removal status (single choice with "not relevant" option)
    'removal_status': [
        'yes',
        'no',
        'not relevant'
    ],
    
    # Binary questions (true binary - yes/no only)
    'binary': [
        'yes',
        'no'
    ],
    
    # Numeric counts
    'numeric_count': [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16'
    ],
    
    # Base colors (for color questions - comprehensive list from data)
    'color': [
        'red',
        'pink',
        'white',
        'brown',
        'yellow',
        'black',
        'grey',
        'blue',
        'green',
        'purple',
        'orange',
        'flesh',
        'none'
    ],
    
    # Simplified locations (instead of 196+ combinations)
    'location_zone': [
        'upper-left',
        'upper-center', 
        'upper-right',
        'center-left',
        'center',
        'center-right',
        'lower-left',
        'lower-center',
        'lower-right'
    ]
}


# ============================================================================
# QUESTION TYPE IDENTIFICATION & MAPPING
# ============================================================================

QUESTION_MAPPING = {
    # Close-ended Multi-label (choose all that apply)
    'are there any abnormalities in the image? check all that are present': {
        'type': 'multi_label',
        'category': 'abnormality',
        'candidates': CANDIDATE_LISTS['abnormality'],
        'output_format': 'list'
    },
    
    'are there any instruments in the image? check all that are present': {
        'type': 'multi_label',
        'category': 'instrument',
        'candidates': CANDIDATE_LISTS['instrument'],
        'output_format': 'list'
    },
    
    'are there any anatomical landmarks in the image? check all that are present': {
        'type': 'multi_label',
        'category': 'anatomical_landmark',
        'candidates': CANDIDATE_LISTS['anatomical_landmark'],
        'output_format': 'list'
    },
    
    # Close-ended Single Choice
    'what type of procedure is the image taken from': {
        'type': 'single_choice',
        'category': 'procedure',
        'candidates': CANDIDATE_LISTS['procedure'],
        'output_format': 'single'
    },
    
    'what type of polyp is present': {
        'type': 'single_choice',
        'category': 'polyp_type',
        'candidates': CANDIDATE_LISTS['polyp_type'],
        'output_format': 'single'
    },
    
    'is this finding easy to detect': {
        'type': 'single_choice',
        'category': 'detection_difficulty',
        'candidates': CANDIDATE_LISTS['detection_difficulty'],
        'output_format': 'single'
    },
    
    'have all polyps been removed': {
        'type': 'single_choice',
        'category': 'removal_status',
        'candidates': CANDIDATE_LISTS['removal_status'],
        'output_format': 'single'
    },
    
    'what is the size of the polyp': {
        'type': 'multi_label',  # Can have multiple sizes
        'category': 'size',
        'candidates': CANDIDATE_LISTS['size'],
        'output_format': 'list'
    },
    
    # True Binary (yes/no only)
    'is there text': {
        'type': 'binary',
        'category': 'binary',
        'candidates': CANDIDATE_LISTS['binary'],
        'output_format': 'single'
    },
    
    'is there a green/black box artefact': {
        'type': 'binary',
        'category': 'binary',
        'candidates': CANDIDATE_LISTS['binary'],
        'output_format': 'single'
    },
    
    'does this image contain any finding': {
        'type': 'binary',
        'category': 'binary',
        'candidates': CANDIDATE_LISTS['binary'],
        'output_format': 'single'
    },
    
    # Numeric (count questions)
    'how many polyps are in the image': {
        'type': 'numeric',
        'category': 'numeric_count',
        'candidates': CANDIDATE_LISTS['numeric_count'],
        'output_format': 'single'
    },
    
    'how many instrumnets are in the image': {
        'type': 'numeric',
        'category': 'numeric_count',
        'candidates': CANDIDATE_LISTS['numeric_count'],
        'output_format': 'single'
    },
    
    'how many findings are present': {
        'type': 'numeric',
        'category': 'numeric_count',
        'candidates': CANDIDATE_LISTS['numeric_count'],
        'output_format': 'single'
    },
    
    # Open-ended with controlled vocabulary (colors)
    'what color is the abnormality? if more than one separate with': {
        'type': 'open_constrained',
        'category': 'color',
        'vocabulary': CANDIDATE_LISTS['color'],
        'max_selections': 3,
        'output_format': 'list'
    },
    
    'what color is the anatomical landmark? if more than one separate with': {
        'type': 'open_constrained',
        'category': 'color',
        'vocabulary': CANDIDATE_LISTS['color'],
        'max_selections': 3,
        'output_format': 'list'
    },
    
    # Handle the "none" questions
    'none': {
        'type': 'open_constrained',
        'category': 'general',
        'vocabulary': ['none', 'n/a', 'not applicable'],
        'max_selections': 1,
        'output_format': 'single'
    },
    
    # Open-ended with controlled vocabulary (locations)
    'where in the image is the abnormality': {
        'type': 'open_constrained',
        'category': 'location_zone',
        'vocabulary': CANDIDATE_LISTS['location_zone'],
        'max_selections': 3,
        'output_format': 'list'
    },
    
    'where in the image is the instrument': {
        'type': 'open_constrained',
        'category': 'location_zone',
        'vocabulary': CANDIDATE_LISTS['location_zone'],
        'max_selections': 3,
        'output_format': 'list'
    },
    
    'where in the image is the anatomical landmark': {
        'type': 'open_constrained',
        'category': 'location_zone',
        'vocabulary': CANDIDATE_LISTS['location_zone'],
        'max_selections': 3,
        'output_format': 'list'
    },
}


# ============================================================================
# INSTRUCTION TEMPLATE GENERATORS
# ============================================================================

def generate_multi_label_instruction(question: str, candidates: List[str], category: str) -> str:
    """Generate instruction for multi-label classification (choose all that apply)."""
    candidates_str = ", ".join([f"'{c}'" for c in candidates])
    
    instruction = f"""You are a surgical image analysis assistant analyzing an endoscopic image.

Question: {question}

Task: Select ALL applicable options from the candidate list below.
Candidates: [{candidates_str}]

Instructions:
- Choose ALL options that apply (this is multi-label classification)
- You MUST select from the candidate list only
- If multiple items apply, list them separated by semicolons
- If none apply, select 'none' or 'normal' if available
- Do not generate any text outside the candidate list

Output Format: item1; item2; item3
Example: polyp; ulcerative colitis

Your answer:"""
    
    return instruction


def generate_single_choice_instruction(question: str, candidates: List[str], category: str) -> str:
    """Generate instruction for single choice classification."""
    candidates_str = ", ".join([f"'{c}'" for c in candidates])
    
    instruction = f"""You are a surgical image analysis assistant analyzing an endoscopic image.

Question: {question}

Task: Select ONE option from the candidate list below.
Candidates: [{candidates_str}]

Instructions:
- Choose ONLY ONE option from the candidate list
- Output the exact text of your choice
- Do not add explanations or additional text

Output Format: selected_option
Example: colonoscopy

Your answer:"""
    
    return instruction


def generate_binary_instruction(question: str) -> str:
    """Generate instruction for true binary questions."""
    instruction = f"""You are a surgical image analysis assistant analyzing an endoscopic image.

Question: {question}

Task: Answer with 'yes' or 'no' only.
Candidates: ['yes', 'no']

Instructions:
- Answer with ONLY 'yes' or 'no'
- Do not add explanations or additional text

Output Format: yes
OR
Output Format: no

Your answer:"""
    
    return instruction


def generate_numeric_instruction(question: str, candidates: List[str]) -> str:
    """Generate instruction for numeric count questions."""
    candidates_str = ", ".join([f"'{c}'" for c in candidates])
    
    instruction = f"""You are a surgical image analysis assistant analyzing an endoscopic image.

Question: {question}

Task: Count and provide the numeric answer.
Valid Answers: [{candidates_str}]

Instructions:
- Provide ONLY the number as your answer
- Choose from the valid answers listed above
- Do not include units or explanations

Output Format: 2
Example: 0

Your answer:"""
    
    return instruction


def generate_open_constrained_instruction(question: str, vocabulary: List[str], 
                                         max_selections: int, category: str) -> str:
    """Generate instruction for open-ended questions with controlled vocabulary."""
    vocab_str = ", ".join([f"'{v}'" for v in vocabulary])
    
    instruction = f"""You are a surgical image analysis assistant analyzing an endoscopic image.

Question: {question}

Task: Provide a brief answer using ONLY words from the controlled vocabulary below.
Vocabulary: [{vocab_str}]

Instructions:
- Use ONLY words from the vocabulary list
- Maximum {max_selections} selections
- Separate multiple items with semicolons
- Be specific and accurate

Output Format: word1; word2
Example: red; pink

Your answer:"""
    
    return instruction


# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================

def create_revised_instruction(item: Dict) -> Dict:
    """
    Create revised instruction with proper candidate lists and format.
    
    Args:
        item: Original data item with question and answer
        
    Returns:
        Enhanced item with proper instruction template
    """
    question = item.get('question', '').lower().strip()
    answer = item.get('answer', '')
    
    # Normalize question for lookup
    question_normalized = question.rstrip('?.;,').strip()
    
    # Find matching question mapping
    if question_normalized in QUESTION_MAPPING:
        mapping = QUESTION_MAPPING[question_normalized]
        qtype = mapping['type']
        category = mapping['category']
        
        # Generate appropriate instruction
        if qtype == 'multi_label':
            instruction = generate_multi_label_instruction(
                question, 
                mapping['candidates'],
                category
            )
        elif qtype == 'single_choice':
            instruction = generate_single_choice_instruction(
                question,
                mapping['candidates'],
                category
            )
        elif qtype == 'binary':
            instruction = generate_binary_instruction(question)
        elif qtype == 'numeric':
            instruction = generate_numeric_instruction(
                question,
                mapping['candidates']
            )
        elif qtype == 'open_constrained':
            instruction = generate_open_constrained_instruction(
                question,
                mapping['vocabulary'],
                mapping.get('max_selections', 3),
                category
            )
        else:
            # Fallback
            instruction = f"Question: {question}\n\nProvide a concise, accurate answer."
        
        # Create enhanced item
        enhanced_item = {
            **item,  # Keep all original fields
            'instruction': instruction,
            'question_type': qtype,
            'category': category,
            'candidates': mapping.get('candidates') or mapping.get('vocabulary'),
            'output_format': mapping['output_format']
        }
        
        return enhanced_item
    
    else:
        # Unknown question - mark for review
        print(f"WARNING: Unknown question pattern: {question}")
        enhanced_item = {
            **item,
            'instruction': f"Question: {question}\n\nProvide a brief, accurate answer (max 15 words).",
            'question_type': 'unknown',
            'category': 'unknown',
            'candidates': None,
            'output_format': 'single'
        }
        return enhanced_item


def process_dataset(input_file: str, output_file: str):
    """Process entire dataset with revised instructions."""
    print(f"\nProcessing: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    stats = {
        'multi_label': 0,
        'single_choice': 0,
        'binary': 0,
        'numeric': 0,
        'open_constrained': 0,
        'unknown': 0
    }
    
    for item in data:
        enhanced_item = create_revised_instruction(item)
        processed_data.append(enhanced_item)
        
        qtype = enhanced_item.get('question_type', 'unknown')
        stats[qtype] = stats.get(qtype, 0) + 1
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"✓ Saved {len(processed_data)} items to: {output_file}")
    print(f"\nQuestion Type Distribution:")
    for qtype, count in sorted(stats.items()):
        if count > 0:
            percentage = (count / len(processed_data)) * 100
            print(f"  {qtype:20s}: {count:5d} ({percentage:5.1f}%)")
    
    return processed_data, stats


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create REVISED instruction templates")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    print("="*80)
    print("CREATING REVISED INSTRUCTION TEMPLATES")
    print("="*80)
    print("\nKey Changes:")
    print("1. ✅ Explicit candidate lists for ALL close-ended questions")
    print("2. ✅ Proper multi-label support for 'check all that apply'")
    print("3. ✅ Separate templates for close vs open questions")
    print("4. ✅ Controlled vocabulary for open-ended questions")
    print("5. ✅ Output format matching GT format")
    print("="*80)
    
    # Process all splits
    for split in ['train.json', 'val.json', 'test.json']:
        input_file = f"{args.input_dir}/{split}"
        output_file = f"{args.output_dir}/{split.replace('.json', '_REVISED.json')}"
        
        if Path(input_file).exists():
            process_dataset(input_file, output_file)
    
    print("\n" + "="*80)
    print("✓ REVISED INSTRUCTION TEMPLATES CREATED")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review sample instructions in output files")
    print("2. Submit to advisor for approval")
    print("3. Update evaluation metrics for each question type")
    print("4. Retrain model with revised instructions")
    print("="*80)


if __name__ == "__main__":
    main()
