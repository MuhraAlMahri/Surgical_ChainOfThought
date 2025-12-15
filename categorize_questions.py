#!/usr/bin/env python3
"""
Categorize questions using LLM semantic classification.

Input: Questions from Kvasir and EndoVis datasets
Output: JSON mapping {question: category}

Categories:
- "abnormality_detection": presence/absence questions
- "characteristics": property questions (color, type, location, count)
- "treatment": diagnosis/clinical significance questions

Example:
{
  "Is there a polyp?": "abnormality_detection",
  "What is the color?": "characteristics",
  "What is the diagnosis?": "treatment"
}
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
from data.question_categorizer import QuestionCategorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_questions_from_json(file_path: str) -> List[Dict]:
    """Load questions from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Try common keys
        if 'questions' in data:
            return data['questions']
        elif 'qa_pairs' in data:
            return data['qa_pairs']
        elif 'data' in data:
            return data['data']
        else:
            # Flatten dict values
            return [item for sublist in data.values() if isinstance(sublist, list) for item in sublist]
    else:
        raise ValueError(f"Unexpected JSON structure in {file_path}")


def extract_unique_questions(data: List[Dict]) -> List[str]:
    """Extract unique questions from data."""
    questions = set()
    for item in data:
        if 'question' in item:
            questions.add(item['question'])
        elif isinstance(item, str):
            questions.add(item)
    
    return sorted(list(questions))


def categorize_dataset_questions(
    categorizer: QuestionCategorizer,
    data: List[Dict],
    dataset_name: str
) -> Dict[str, str]:
    """
    Categorize all questions in a dataset.
    
    Returns:
        Dictionary mapping question -> category name
    """
    question_to_category = {}
    
    logger.info(f"Categorizing {len(data)} items from {dataset_name}...")
    
    for item in data:
        question = item.get('question', '')
        if not question:
            continue
        
        # Classify question
        result = categorizer.classify_question(
            question=question,
            category=item.get('category') or item.get('question_type'),
            dataset=dataset_name
        )
        
        # Map stage number to category name
        stage = result['stage']
        if stage == 1:
            category_name = "abnormality_detection"
        elif stage == 2:
            category_name = "characteristics"
        elif stage == 3:
            category_name = "treatment"
        else:
            category_name = result.get('category_name', 'abnormality_detection')
        
        question_to_category[question] = category_name
    
    logger.info(f"Categorized {len(question_to_category)} questions")
    return question_to_category


def main():
    parser = argparse.ArgumentParser(
        description="Categorize surgical VQA questions into clinical stages"
    )
    parser.add_argument(
        "--kvasir_path",
        type=str,
        help="Path to Kvasir dataset JSON file"
    )
    parser.add_argument(
        "--endovis_path",
        type=str,
        help="Path to EndoVis dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="question_categories.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model for classification"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    args = parser.parse_args()
    
    # Initialize categorizer
    categorizer = QuestionCategorizer(
        model_name=args.model,
        use_cache=not args.no_cache
    )
    
    # Collect all questions
    all_categories = {}
    
    # Process Kvasir if provided
    if args.kvasir_path:
        logger.info(f"Processing Kvasir dataset: {args.kvasir_path}")
        kvasir_data = load_questions_from_json(args.kvasir_path)
        kvasir_categories = categorize_dataset_questions(
            categorizer,
            kvasir_data,
            "kvasir"
        )
        all_categories["kvasir"] = kvasir_categories
        logger.info(f"Kvasir: {len(kvasir_categories)} questions categorized")
    
    # Process EndoVis if provided
    if args.endovis_path:
        logger.info(f"Processing EndoVis dataset: {args.endovis_path}")
        endovis_data = load_questions_from_json(args.endovis_path)
        endovis_categories = categorize_dataset_questions(
            categorizer,
            endovis_data,
            "endovis"
        )
        all_categories["endovis"] = endovis_categories
        logger.info(f"EndoVis: {len(endovis_categories)} questions categorized")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_categories, f, indent=2)
    
    logger.info(f"Saved categories to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CATEGORIZATION SUMMARY")
    print("="*60)
    for dataset_name, categories in all_categories.items():
        print(f"\n{dataset_name.upper()}:")
        category_counts = {}
        for question, category in categories.items():
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} questions")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()













