#!/usr/bin/env python3
"""
Evaluation script for multi-head CoT models.

Compares:
- Baseline (no CoT): Your existing results
- Multi-head CoT: New results

Outputs comparison table.
"""

import torch
import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
import logging
import argparse

from multihead_model import create_multihead_model
from cot_prompts import build_cot_prompt, format_prompt_for_model
from data.vqa_data_loader import create_data_loader
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_question_categories(categories_file: str) -> Dict[str, Dict[str, str]]:
    """Load question categories mapping."""
    with open(categories_file, 'r') as f:
        return json.load(f)


def evaluate(
    model,
    test_loader,
    question_categories: Dict[str, str],
    dataset_name: str,
    processor,
    device: str
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Multi-head model
        test_loader: Test data loader
        question_categories: Question category mapping
        dataset_name: Dataset name
        processor: Model processor
        device: Device
        
    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    
    correct = 0
    total = 0
    
    # Per-category tracking
    category_correct = {
        'abnormality_detection': 0,
        'characteristics': 0,
        'treatment': 0
    }
    category_total = {
        'abnormality_detection': 0,
        'characteristics': 0,
        'treatment': 0
    }
    
    all_predictions = []
    all_answers = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
            images = batch.get('images', [])
            questions = batch.get('questions', [])
            answers = batch.get('answers', [])
            categories = batch.get('categories', [])
            
            for i in range(len(questions)):
                if i >= len(images) or images[i] is None:
                    continue
                
                question = questions[i]
                answer = answers[i]
                category = categories[i] if i < len(categories) else question_categories.get(question, 'abnormality_detection')
                
                # Normalize category
                if category == 1 or category == "1":
                    category = 'abnormality_detection'
                elif category == 2 or category == "2":
                    category = 'characteristics'
                elif category == 3 or category == "3":
                    category = 'treatment'
                
                try:
                    # Build prompt
                    prompt = build_cot_prompt(question, category, model_type="qwen3vl")
                    formatted_prompt = format_prompt_for_model(prompt, model_type="qwen3vl", processor=processor)
                    
                    # Process inputs
                    inputs = processor(formatted_prompt, images[i], return_tensors="pt").to(device)
                    
                    # Generate answer
                    result = model.generate(
                        images=[images[i]],
                        prompt=question,
                        category=category,
                        max_new_tokens=256,
                        temperature=0.7
                    )
                    
                    prediction = result.get('answer', '').strip()
                    
                    # Check if answer is correct
                    is_correct = check_correctness(prediction, answer)
                    
                    if is_correct:
                        correct += 1
                        category_correct[category] += 1
                    
                    total += 1
                    category_total[category] += 1
                    
                    all_predictions.append(prediction)
                    all_answers.append(answer)
                    
                except Exception as e:
                    logger.warning(f"Error evaluating sample {i}: {e}")
                    continue
    
    # Calculate accuracies
    overall_accuracy = correct / total if total > 0 else 0.0
    
    category_accuracies = {
        cat: category_correct[cat] / max(category_total[cat], 1)
        for cat in category_correct
    }
    
    return {
        'overall_accuracy': overall_accuracy,
        'category_accuracies': category_accuracies,
        'correct': correct,
        'total': total,
        'predictions': all_predictions,
        'answers': all_answers
    }


def check_correctness(prediction: str, answer: str) -> bool:
    """Check if prediction matches answer."""
    pred_lower = prediction.lower().strip()
    ans_lower = answer.lower().strip()
    
    # Exact match
    if pred_lower == ans_lower:
        return True
    
    # Check if answer is contained in prediction (for longer answers)
    if ans_lower in pred_lower or pred_lower in ans_lower:
        return True
    
    # For multi-label answers (semicolon-separated)
    if ';' in ans_lower:
        ans_parts = [a.strip() for a in ans_lower.split(';')]
        # Check if all answer parts are in prediction
        return all(part in pred_lower for part in ans_parts)
    
    return False


def print_comparison_table(baseline_results: Dict[str, float], cot_results: Dict[str, float]):
    """
    Print comparison table.
    
    Args:
        baseline_results: Dictionary mapping dataset -> accuracy
        cot_results: Dictionary mapping dataset -> accuracy
    """
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    datasets = set(list(baseline_results.keys()) + list(cot_results.keys()))
    
    print(f"\n{'Dataset':<20} {'Baseline (no CoT)':<20} {'Multi-head CoT':<20} {'Improvement':<15}")
    print("-" * 80)
    
    for dataset in sorted(datasets):
        baseline_acc = baseline_results.get(dataset, 0.0)
        cot_acc = cot_results.get(dataset, 0.0)
        improvement = cot_acc - baseline_acc
        
        print(f"{dataset:<20} {baseline_acc:<20.2%} {cot_acc:<20.2%} {improvement:+.2%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-head CoT model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", choices=["qwen3vl", "medgemma", "llava_med"], required=True)
    parser.add_argument("--test-data", required=True, help="Path to test data JSON")
    parser.add_argument("--image-base-path", required=True, help="Base path for images")
    parser.add_argument("--question-categories", default="question_categories.json", help="Question categories file")
    parser.add_argument("--dataset", choices=["kvasir", "endovis"], required=True)
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--baseline-results", help="Path to baseline results JSON")
    
    args = parser.parse_args()
    
    # Load question categories
    all_categories = load_question_categories(args.question_categories)
    question_categories = all_categories.get(args.dataset, {})
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = create_multihead_model(
        base_checkpoint=args.checkpoint,
        model_type=args.model_type,
        freeze_base=True
    )
    
    # Get processor
    if args.model_type == "qwen3vl":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
    elif args.model_type == "medgemma":
        processor = AutoProcessor.from_pretrained("google/medgemma-4b", trust_remote_code=True)
    else:  # llava_med
        processor = AutoProcessor.from_pretrained("microsoft/llava-med-v1.5-mistral-7b", trust_remote_code=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create test loader
    test_loader = create_data_loader(
        data_file=args.test_data,
        image_base_path=args.image_base_path,
        batch_size=1,
        shuffle=False,
        is_temporal=(args.dataset == "endovis")
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    results = evaluate(
        model=model,
        test_loader=test_loader,
        question_categories=question_categories,
        dataset_name=args.dataset,
        processor=processor,
        device=device
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"Correct: {results['correct']} / {results['total']}")
    print("\nPer-Category Accuracies:")
    for category, acc in results['category_accuracies'].items():
        print(f"  {category}: {acc:.2%}")
    
    # Compare with baselines if provided
    if args.baseline_results:
        with open(args.baseline_results, 'r') as f:
            baseline_results = json.load(f)
        
        # Create comparison
        dataset_key = f"{args.model_type}_{args.dataset}"
        baseline_acc = baseline_results.get(dataset_key, 0.0)
        cot_acc = results['overall_accuracy']
        
        print_comparison_table(
            {dataset_key: baseline_acc},
            {dataset_key: cot_acc}
        )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"evaluation_results_{args.model_type}_{args.dataset}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {results_file}")


if __name__ == "__main__":
    main()













